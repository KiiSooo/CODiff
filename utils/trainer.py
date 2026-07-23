import glob
import os
from pathlib import Path
import wandb
import numpy as np
import torch
from accelerate import Accelerator
from utils.logger_utils import create_url_shortcut_of_wandb, create_logger_of_wandb
from utils.train_utils import SmoothedValue, set_random_seed
from utils.import_utils import fill_args_from_dict
import torch.nn.functional as F
import matplotlib.pyplot as plt
from model.train_val_forward import modification_train_val_forward
import re
from collections import OrderedDict

def exists(x):
    return x is not None

def cal_mae(gt, res, thresholding, save_to=None, n=None):
    if res.shape != gt.shape:
        res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
    if thresholding:
        res = (res > 0.5).float()
    else:
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
    res = res.cpu().numpy().squeeze()
    if save_to is not None:
        plt.imsave(os.path.join(save_to, n), res, cmap='gray')
    return np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])


class Trainer(object):
    def __init__(
            self,
            model,
            train_loader: torch.utils.data.DataLoader,
            test_loader: torch.utils.data.DataLoader = None,
            train_val_forward_fn=modification_train_val_forward,
            gradient_accumulate_every=1,
            optimizer=None, scheduler=None,
            train_num_epoch=100,
            results_folder='./results_folder/',
            amp=False,
            fp16=False,
            split_batches=True,
            log_with='wandb',
            cfg=None,
    ):
        super().__init__()
        from accelerate import DistributedDataParallelKwargs
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision='fp16' if fp16 else 'no',
            log_with='wandb' if log_with else None,
            gradient_accumulation_steps=gradient_accumulate_every,
            kwargs_handlers=[ddp_kwargs]
        )
        project_name = getattr(cfg, "project_name", 'CODiff')
        self.accelerator.init_trackers(project_name, config=cfg)
        create_url_shortcut_of_wandb(accelerator=self.accelerator)
        self.logger = create_logger_of_wandb(accelerator=self.accelerator, rank=not self.accelerator.is_main_process)
        self.accelerator.native_amp = amp
        self.model = model
        self.train_val_forward_fn = train_val_forward_fn
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_epoch = train_num_epoch
        self.opt = optimizer

        if self.accelerator.is_main_process:
            self.results_folder = Path(results_folder)
            self.results_folder.mkdir(exist_ok=True)
        self.cur_epoch = 0

        self.model, self.opt, self.scheduler, self.train_loader, self.test_loader \
            = self.accelerator.prepare(self.model, self.opt, scheduler, self.train_loader, self.test_loader)

    def save(self, epoch, max_to_keep=1, model=None):
        self.accelerator.wait_for_everyone()
        if not self.accelerator.is_local_main_process:
            return
        
        if model is None:
            model = self.accelerator.unwrap_model(self.model)

        ckpt_files = glob.glob(os.path.join(self.results_folder, 'model-[0-9]*.pt'))
        ckpt_files = sorted(ckpt_files, key=lambda x: int(x.split('-')[-1].split('.')[0]))
        ckpt_files_to_delete = ckpt_files[:-max_to_keep]
        for ckpt_file in ckpt_files_to_delete:
            os.remove(ckpt_file)
        
        data = {
            'epoch': self.cur_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.opt.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None
        }

        save_name = str(self.results_folder / f'model-{epoch}.pt')
        last_save_name = str(self.results_folder / f'model-{epoch}-last.pt')

        if os.path.exists(save_name):
            if os.path.exists(last_save_name):
                os.remove(last_save_name)
            os.rename(save_name, last_save_name)

        torch.save(data, save_name)

    def load(self, resume_path: str = None, pretrained_path: str = None):
        accelerator = self.accelerator
        device = accelerator.device
        if resume_path is not None:
            data = torch.load(resume_path, map_location=device)
            self.cur_epoch = data['epoch']
            try:
                self.opt.load_state_dict(data['optimizer_state_dict'])
            except ValueError:
                print("Optimizer state not loaded due to param group mismatch.")
            if self.scheduler and 'scheduler_state_dict' in data and data['scheduler_state_dict'] is not None:
                self.scheduler.load_state_dict(data['scheduler_state_dict'])
            if exists(self.accelerator.scaler) and 'scaler_state_dict' in data and data['scaler_state_dict'] is not None:
                self.accelerator.scaler.load_state_dict(data['scaler_state_dict'])

        elif pretrained_path is not None:
            data = torch.load(pretrained_path, map_location=device)
        else:
            raise ValueError('Must specify resume_path or pretrained_path.')
        if self.scheduler is not None:
            for _ in range(self.cur_epoch):
                self.scheduler.step(_)
        model = self.accelerator.unwrap_model(self.model)
        
        state_dict = data['model_state_dict']
        new_state_dict = OrderedDict(
            [(re.sub(r'^module\.', '', k), v) for k, v in state_dict.items()])

        model.load_state_dict(new_state_dict, strict=True)

    @torch.inference_mode()
    def val_time_ensemble(self, model, test_data_loader, accelerator, thresholding=False, save_to=None):
        if not hasattr(self, "_best_score"):
            self._best_score = float("inf")

        model.eval()
        model = accelerator.unwrap_model(model)
        device = next(model.parameters()).device
        ensemble_maes = []
        val_loss = []
        
        ensemble_maes = []

        model.eval()

        for data in test_data_loader:
            image, gt, name = data['image'], data['gt'], data['name']
            gt = [np.array(x, np.float32) for x in gt]
            gt = [(x > 0.5).astype(np.float32) for x in gt]
            image = image.to(device).squeeze(1)

            out = self.train_val_forward_fn(model, image=image, time_ensemble=True, gt_sizes=[g.shape for g in gt])
            ensem_res = out["pred"]

            ensemble_maes += [cal_mae(g, r, thresholding, save_to, n) for g, r, n in zip(gt, ensem_res, name)]

        accelerator.wait_for_everyone()
        val_loss = accelerator.gather(torch.tensor(ensemble_maes).mean().to(device)).mean().item()
        self._best_score = min(self._best_score, val_loss)

        return val_loss, self._best_score
        
    def train(self):     
        log_interval = 64
        accelerator = self.accelerator
        
        for epoch in range(self.cur_epoch, self.train_num_epoch):
            self.cur_epoch = epoch
            self.model.train()
            loss_sm = SmoothedValue(window_size=10) 

            for i, data in enumerate(self.train_loader):
                with accelerator.autocast(), accelerator.accumulate(self.model):
                    loss = fill_args_from_dict(self.train_val_forward_fn, data)(model=self.model)
                    accelerator.backward(loss)
                    accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.opt.step()
                    self.opt.zero_grad()
                loss_sm.update(loss.item())

                if i % log_interval == 0:
                    self.accelerator.log({
                        'loss': loss_sm.avg,
                        'lr': self.opt.param_groups[0]['lr']
                    })
                
            accelerator.wait_for_everyone()
            loss_sm_gather = accelerator.gather(torch.tensor([loss_sm.global_avg]).to(accelerator.device))
            loss_sm_avg = loss_sm_gather.mean().item()
            self.logger.info(f'Epoch:{epoch}/{self.train_num_epoch} loss: {loss_sm_avg:.4f}')

            # Val
            self.model.eval()
            if (epoch + 1) % 1 == 0 or (epoch >= self.train_num_epoch * 0.7):
                score, best_score = self.val_time_ensemble(self.model, self.test_loader, accelerator)
                self.logger.info(f'Epoch:{epoch}/{self.train_num_epoch} score: {score:.4f}({best_score:.4f})')
                accelerator.log({'score': score, 'best_score': best_score})
                if score == best_score:
                    self.save("best", model = self.model,max_to_keep=10)
                
                if epoch % 50 ==0:
                    self.save(epoch, model = self.model,max_to_keep=10)

            # Scheduler
            if self.scheduler is not None:
                self.scheduler.scheduler.step(epoch)
                
            lr = self.opt.param_groups[0]['lr']
            self.logger.info(f'Epoch:{epoch}/{self.train_num_epoch} lr: {lr:.6e}')

            with torch.inference_mode():
                if accelerator.is_main_process:
                    model = self.accelerator.unwrap_model(self.model)
                    for tracker in accelerator.trackers:
                        if tracker.name == "wandb":
                            out = fill_args_from_dict(self.train_val_forward_fn, data)(model=model,
                                                                                       verbose=False)
                            tracker.log(
                                {'pred-img-mask':
                                     [wandb.Image(o[0, :, :]) for o in out.values()]
                                 })

            accelerator.wait_for_everyone()
        self.logger.info('training complete')
        accelerator.end_training()


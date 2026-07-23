import argparse
import torch
from utils.collate_utils import collate, SampleDataset
from utils.import_utils import instantiate_from_config, recurse_instantiate_from_config, get_obj_from_str
from utils.init_utils import add_args, config_pretty
from utils.train_utils import set_random_seed
from torch.utils.data import DataLoader
from utils.trainer import Trainer
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

set_random_seed(42)

def get_loader(cfg):
    cod10k = instantiate_from_config(cfg.train_dataset.COD10K)
    camo = instantiate_from_config(cfg.train_dataset.CAMO)
    train_dataset = torch.utils.data.ConcatDataset([camo, cod10k])
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers
    )

    camo = instantiate_from_config(cfg.test_dataset.CAMO)
    cod10k = SampleDataset(full_dataset=instantiate_from_config(cfg.test_dataset.COD10K), interval=10)
    val_dataset = torch.utils.data.ConcatDataset([camo, cod10k])

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        collate_fn=collate
    )
    return train_loader, val_loader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--results_folder', type=str, default=None, help='None for saving in results folder.')
    parser.add_argument('--num_epoch', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--gradient_accumulate_every', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)

    cfg = add_args(parser)

    model = recurse_instantiate_from_config(cfg.model)
    diffusion_model = instantiate_from_config(cfg.diffusion_model,
                                              model=model)

    train_loader, val_loader = get_loader(cfg)

    optimizer = instantiate_from_config(cfg.optimizer, params=model.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.05)
    scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=150, cycle_mult=1.0, max_lr=1e-4, min_lr=2e-5, warmup_steps=0, gamma=1)
    trainer = Trainer(
        diffusion_model, train_loader, val_loader,
        train_val_forward_fn=get_obj_from_str(cfg.train_val_forward_fn),
        gradient_accumulate_every=cfg.gradient_accumulate_every,
        results_folder=cfg.results_folder,
        optimizer=optimizer, scheduler=scheduler,
        train_num_epoch=cfg.num_epoch,
        amp=cfg.fp16,
        log_with=None,
        cfg=cfg
    )

    if trainer.accelerator.is_main_process:
        config_pretty(cfg)

    if getattr(cfg, 'resume', None) or getattr(cfg, 'pretrained', None):
        trainer.load(resume_path=cfg.resume, pretrained_path=cfg.pretrained)
    trainer.train()
import torch
from einops import rearrange, pack, unpack
from torch import nn
from denoising_diffusion_pytorch.simple_diffusion import *

class CondGaussianDiffusion(GaussianDiffusion):
    def __init__(
            self,
            model: nn.Module,
            *,
            image_size,
            channels=1,
            extra_channels=0,
            cond_channels=3,
            pred_objective='x0',
            loss_type='l2',
            noise_schedule=logsnr_schedule_cosine,
            noise_d=None,
            noise_d_low=None,
            noise_d_high=None,
            num_sample_steps=1,
            clip_sample_denoised=True,
    ):
        super(CondGaussianDiffusion, self).__init__(model, image_size=image_size, channels=channels,
                                                    pred_objective=pred_objective, noise_schedule=noise_schedule,
                                                    noise_d=noise_d, noise_d_low=noise_d_low, noise_d_high=noise_d_high,
                                                    num_sample_steps=num_sample_steps,
                                                    clip_sample_denoised=clip_sample_denoised)
        if loss_type not in ['l2', 'l1', 'l1+l2', 'mean(l1, l2)']:
            try:
                from utils.import_utils import get_obj_from_str
                loss_type = get_obj_from_str(loss_type)
            except:
                raise NotImplementedError

        self.loss_type = loss_type
        self.extra_channels = extra_channels
        self.cond_channels = cond_channels

    def forward(self, img, cond_img, seg=None, extra_cond=None, *args, **kwargs):
        b, channels, h, w = img.shape
        cond_channels = cond_img.shape[1]
        assert channels == self.channels
        assert h == w == self.image_size
        assert cond_channels == self.cond_channels

        times = torch.zeros((img.shape[0],), device=self.device).float().uniform_(0, 1)
        return self.p_losses(img, times, cond_img, seg, extra_cond, *args, **kwargs)

    def p_losses(self, x_start, times, cond_img, seg=None, extra_cond=None, noise=None, *args, **kwargs):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x, log_snr = self.q_sample(x_start=x_start, times=times, noise=noise)

        model_out, loss_net = self.model(x, log_snr, cond_img, x_start)

        if self.pred_objective == 'v':
            padded_log_snr = right_pad_dims_to(x, log_snr)
            alpha, sigma = padded_log_snr.sigmoid().sqrt(), (-padded_log_snr).sigmoid().sqrt()
            target = alpha * noise - sigma * x_start

        elif self.pred_objective == 'eps':
            target = noise

        elif self.pred_objective == 'x0':
            target = x_start

        return self.loss_type(model_out, target, loss_net)

    @torch.no_grad()
    def sample(self, cond_img, extra_cond=None):
        b, c, h, w = cond_img.shape
        return self.p_sample_loop((b, self.channels, self.image_size, self.image_size),
                                  cond_img,
                                  extra_cond=extra_cond)

    def p_sample_loop(self, shape, cond_img, extra_cond):
        self.history = []
        img = torch.randn(shape, device=self.device)
        steps = torch.linspace(1., 0., self.num_sample_steps + 1, device=self.device)

        for i in range(self.num_sample_steps):
            times = steps[i]
            times_next = steps[i + 1]
            img = self.p_sample(img, cond_img, extra_cond, times, times_next)

        img.clamp_(0., 1.)
        return img

    @torch.no_grad()
    def p_sample(self, x, cond, extra_cond, time, time_next):
        batch, *_, device = *x.shape, x.device

        model_mean, model_variance = self.p_mean_variance(x=x, cond=cond, extra_cond=extra_cond, time=time,
                                                          time_next=time_next)
        if time_next == 0:
            return model_mean

        noise = torch.randn_like(x)
        return model_mean + sqrt(model_variance) * noise

    @torch.no_grad()
    def p_mean_variance(self, x, cond, extra_cond, time, time_next):
        log_snr = self.log_snr(time)
        log_snr_next = self.log_snr(time_next)
        c = -expm1(log_snr - log_snr_next)

        squared_alpha, squared_alpha_next = log_snr.sigmoid(), log_snr_next.sigmoid()
        squared_sigma, squared_sigma_next = (-log_snr).sigmoid(), (-log_snr_next).sigmoid()

        alpha, sigma, alpha_next = map(sqrt, (squared_alpha, squared_sigma, squared_alpha_next))

        batch_log_snr = repeat(log_snr, ' -> b', b=x.shape[0])
        pred = self.model.sample_net(x, batch_log_snr, cond)

        if self.pred_objective == 'v':
            x_start = alpha * x - sigma * pred

        elif self.pred_objective == 'eps':
            x_start = (x - sigma * pred) / alpha

        elif self.pred_objective == 'x0':
            x_start = pred

        x_start.clamp_(0., 1.)
        
        self.history.append(x_start.detach().to(self.device))

        model_mean = alpha_next * (x * (1 - c) / alpha + c * x_start)

        posterior_variance = squared_sigma_next * c

        return model_mean, posterior_variance

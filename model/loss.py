import torch
import torch.nn.functional as F
from pytorch_msssim import ssim

def ssim_loss(pred, mask):
    return 1.0 - ssim(pred, mask.float(), data_range=1.0, size_average=True)

def structure_loss(pred, mask, loss_net):
    with torch.autocast(device_type='cuda', enabled=False):
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        wbce = F.binary_cross_entropy(pred, mask, reduction='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
        inter = ((pred * mask) * weit).sum(dim=(2, 3))
        union = ((pred + mask) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)
        mae_loss = (pred - mask).abs().mean(dim=(2, 3))
        ssim= ssim_loss(pred, mask)
        
        return (wbce + wiou + mae_loss+ ssim).mean() + 0.75*loss_net

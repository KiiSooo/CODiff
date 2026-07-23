from torch import nn
import torch.nn.functional as F

def modification_train_val_forward(model: nn.Module, gt=None, image=None, **kwargs):
    if model.training:
        assert gt is not None and image is not None
        return model(gt, image, **kwargs)
    else:
        time_ensemble = kwargs.pop('time_ensemble') if 'time_ensemble' in kwargs else False
        gt_sizes = kwargs.pop('gt_sizes') if time_ensemble else None
        pred = model.sample(image, **kwargs)
        
        def process(p, gt_size):
            p = F.interpolate(p.unsqueeze(0) , size=gt_size[-2:], mode='bilinear', align_corners=False)
            return p
        pred = [process(p, gt_size) for (p, gt_size) in zip(pred, gt_sizes)]
  
        return {
            "image": image,
            "pred": pred,
            "gt": gt if gt is not None else None,
        }
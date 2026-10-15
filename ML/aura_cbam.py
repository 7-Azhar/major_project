"""CBAM insertion at trainer construction time, compatible with Ultralytics 8.4.127.

Keep this module importable alongside any checkpoints trained with it.
This follows the notebook's average-pooling channel-attention variant.
"""
from copy import deepcopy
import torch
from torch import nn
from ultralytics.models.yolo.detect import DetectionTrainer


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(), nn.Conv2d(hidden, channels, 1, bias=False), nn.Sigmoid())
        self.spatial_attention = nn.Sequential(nn.Conv2d(2, 1, 7, padding=3, bias=False), nn.Sigmoid())

    def forward(self, x):
        x = x * self.channel_attention(x)
        pooled = torch.cat([x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)], dim=1)
        return x * self.spatial_attention(pooled)


def cbam_count(model):
    return sum(isinstance(module, CBAM) for module in model.modules())


def insert_cbam(model, indices=(2, 5, 8)):
    if cbam_count(model):
        raise ValueError('Model already contains CBAM; refusing duplicate insertion')
    for idx in indices:
        old = model.model[idx]
        if hasattr(old, 'cv2') and hasattr(old.cv2, 'conv'):
            channels = old.cv2.conv.out_channels
        elif hasattr(old, 'conv'):
            channels = old.conv.out_channels
        else:
            raise TypeError(f'Unsupported layer {idx}: {type(old).__name__}')
        wrapper = nn.Sequential(old, CBAM(channels))
        # YOLO routes tensors using metadata on each top-level layer.
        for attr in ['i', 'f']:
            setattr(wrapper, attr, getattr(old, attr))
        wrapper.type = type(old).__name__ + '+CBAM'
        wrapper.np = sum(p.numel() for p in wrapper.parameters())
        model.model[idx] = wrapper
    assert cbam_count(model) == len(indices)
    return model


class CBAMTrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        if isinstance(weights, nn.Module) and cbam_count(weights):
            model = deepcopy(weights).float()
            if model.model[-1].nc != self.data['nc']:
                raise ValueError('CBAM checkpoint class count differs from dataset')
        else:
            model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
            model = insert_cbam(model)
        if cbam_count(model) != 3:
            raise RuntimeError('Expected exactly three CBAM blocks before training')
        return model


def verify_training_model(trainer):
    model = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
    if cbam_count(model) != 3:
        raise RuntimeError('CBAM missing from active training model; stopping before training')
    optimizer_ids = {id(p) for group in trainer.optimizer.param_groups for p in group['params']}
    attention_parameters = [p for module in model.modules() if isinstance(module, CBAM)
                            for p in module.parameters()]
    if not attention_parameters or not all(id(p) in optimizer_ids for p in attention_parameters):
        raise RuntimeError('CBAM parameters are missing from optimizer')
    print('AURA verified: 3 CBAM blocks present and included in optimizer', flush=True)

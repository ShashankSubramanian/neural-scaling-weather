import os, sys
import logging
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import math


def set_scheduler(cfg, opt):
    """set the lr scheduler"""
    total_steps = cfg.optimizer.max_iterations
    if cfg.optimizer.scheduler == "reducelr":
        scheduler = lr_scheduler.ReduceLROnPlateau(
            opt,
            "min",
            patience=cfg.optimizer.patience,
            verbose=True,
            min_lr=1e-3 * 1e-5,
            factor=0.2,
        )
    elif cfg.optimizer.scheduler == "cosine":
        scheduler = lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    elif cfg.optimizer.scheduler == "cosine_warmup":
        lr_scale = lambda x: min(
            (x + 1) / cfg.optimizer.warmup_steps,
            0.5 * (1 + np.cos(np.pi * x / total_steps)),
        )
        scheduler = lr_scheduler.LambdaLR(opt, lr_scale)
    elif cfg.optimizer.scheduler == "cooldown":
        if cfg.optimizer.cooldown_fraction is not None:
            cooldown_steps = int(
                cfg.optimizer.cooldown_to_iter * cfg.optimizer.cooldown_fraction
            )
            steps = cfg.optimizer.cooldown_to_iter - cfg.optimizer.cooldown_from_iter
            constant_steps = steps - cooldown_steps
            assert (
                constant_steps >= 0
            ), "cooldown fraction too large; must cooldown from an earlier iteration"
        else:
            constant_steps = 0
            cooldown_steps = cfg.optimizer.cooldown_to_iter - cfg.optimizer.cooldown_from_iter

        def lr_scale(x):
            if x <= constant_steps:
                return 1.0
            else:
                cooldown_step = x - constant_steps
                return 1 - math.sqrt(cooldown_step / cooldown_steps)

        scheduler = lr_scheduler.LambdaLR(opt, lr_scale)
    elif cfg.optimizer.scheduler == "fixed_warmup":
        def lr_scale(x):
            if x < cfg.optimizer.warmup_steps:
                return (x + 1) / cfg.optimizer.warmup_steps
            else:
                return 1.0

        scheduler = lr_scheduler.LambdaLR(opt, lr_scale)
    else:
        scheduler = None
    return scheduler


def set_optimizer(cfg, net):
    """set the optimizer"""
    if cfg.optimizer.optimizer == "adam":
        optimizer = optim.Adam(
            net.parameters(), lr=cfg.optimizer.lr, fused=True, betas=(0.9, 0.95)
        )
    elif cfg.optimizer.optimizer == "adamw":
        optimizer = optim.AdamW(
            net.parameters(), 
            lr=cfg.optimizer.lr, 
            fused=True, 
            betas=(0.9, 0.95), 
            weight_decay=cfg.optimizer.weight_decay,
        )
    elif cfg.optimizer.optimizer == "sgd":
        optimizer = optim.SGD(net.parameters(), lr=cfg.optimizer.lr, momentum=0.9)
    return optimizer

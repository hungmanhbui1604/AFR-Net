import math

import torch
from torch.optim.lr_scheduler import LambdaLR


def cosine_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_iters: int,
    total_iters: int,
    min_lr: float = 1e-6,
) -> LambdaLR:

    def get_lr_lambda(base_lr: float):
        def _lr_lambda(step: int) -> float:
            if base_lr == 0.0:
                return 0.0

            if step < warmup_iters:
                warmup_progress = (step + 1) / max(warmup_iters, 1)
                lr = min_lr + (base_lr - min_lr) * warmup_progress
                return lr / base_lr

            progress = (step - warmup_iters) / max(total_iters - warmup_iters, 1)
            progress = min(progress, 1.0)

            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            lr = min_lr + (base_lr - min_lr) * cosine_decay

            return lr / base_lr

        return _lr_lambda

    lr_lambdas = [get_lr_lambda(pg["lr"]) for pg in optimizer.param_groups]

    return LambdaLR(
        optimizer,
        lr_lambda=lr_lambdas,
    )

def polynomial_scheduler(
    optimizer: torch.optim.Optimizer,
    total_iters: int,
    lr_min: float,
    power: float,
) -> LambdaLR:
    """lr(t) = (base_lr - lr_min) * (1 - t / total_steps) ** power + lr_min"""
    def get_lr_lambda(base_lr: float):
        def _lr_lambda(t: int) -> float:
            if t >= total_iters:
                return lr_min / base_lr
            decay = (1 - t / total_iters) ** power
            return (base_lr - lr_min) * decay / base_lr + lr_min / base_lr
        return _lr_lambda

    lr_lambdas = [get_lr_lambda(pg["lr"]) for pg in optimizer.param_groups]

    return LambdaLR(optimizer, lr_lambda=lr_lambdas)


def get_scheduler(sched_name: str, optimizer: torch.optim.Optimizer, iters: int, epochs: int, sched_cfg: dict):
    total_iters = iters * epochs
    warmup_iters = iters * sched_cfg["warmup_epochs"]

    if sched_name == "cosine":
        return cosine_warmup_scheduler(optimizer=optimizer, total_iters=total_iters, warmup_iters=warmup_iters, min_lr=sched_cfg["min_lr"])
    elif sched_name == "polynomial":
        return polynomial_scheduler(
            optimizer=optimizer,
            total_iters=total_iters,
            lr_min=sched_cfg["min_lr"],
            power=sched_cfg["power"],
        )
    else:
        raise ValueError("Unknown scheduler: " + sched_name)

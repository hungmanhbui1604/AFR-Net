import argparse
import os
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from data import AuthenticationEvaluationDataset, RecogTrainingDataset, UniqueFingerprintDataset
from losses import ArcFaceLoss
from metrics import compute_authentication_metrics
from models import get_model
from schedulers import get_scheduler
from transforms import get_transforms

# ---------------------------------------------------------------------------
# DDP Utilities
# ---------------------------------------------------------------------------


def setup_ddp() -> tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl", device_id=local_rank)
        return local_rank, dist.get_world_size()
    return local_rank, 1


def cleanup_ddp() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _unwrap(module):
    return module.module if isinstance(module, DDP) else module


# ---------------------------------------------------------------------------
# Optimizer Selection
# ---------------------------------------------------------------------------


def get_optimizer(opt_name: str, parameters: list, opt_cfg: dict):
    if opt_name == "adamw":
        return torch.optim.AdamW(
            parameters, lr=opt_cfg["lr"], weight_decay=opt_cfg["weight_decay"]
        )
    elif opt_name == "adam":
        return torch.optim.Adam(
            parameters, lr=opt_cfg["lr"], weight_decay=opt_cfg["weight_decay"]
        )
    else:
        raise ValueError("Unknown optimizer: " + opt_name)


# ---------------------------------------------------------------------------
# Embedding Extraction & Evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def get_embeddings(
    model: torch.nn.Module,
    val_unique_loader: DataLoader,
    device: torch.device,
    epoch: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()

    embed_dim_c = _unwrap(model).zc_linear.out_features
    embed_dim_a = _unwrap(model).attention_head.vit_feature.out_features
    n_unique_images = len(val_unique_loader.dataset)

    local_embeddings_c = torch.zeros((n_unique_images, embed_dim_c), device=device)
    local_embeddings_a = torch.zeros((n_unique_images, embed_dim_a), device=device)
    local_features = None
    mask = torch.zeros(n_unique_images, device=device, dtype=torch.bool)

    pbar = tqdm(
        val_unique_loader,
        desc=f"Epoch {epoch:03d} [val extract]",
        leave=False,
        unit="batch",
        disable=not is_main(),
    )

    for idxs, imgs in pbar:
        imgs = imgs.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda"):
            zc, za, l_feats = model(imgs)
            
        if local_features is None:
            local_features = torch.zeros((n_unique_images, *l_feats.shape[1:]), device=device, dtype=torch.float32)
        
        zc = F.normalize(zc, dim=1).float()
        za = F.normalize(za, dim=1).float()

        local_embeddings_c[idxs] = zc
        local_embeddings_a[idxs] = za
        local_features[idxs] = l_feats.float()
        mask[idxs] = True

    counts = torch.zeros(n_unique_images, device=device)
    counts[mask] = 1.0

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(local_embeddings_c, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_embeddings_a, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_features, op=dist.ReduceOp.SUM)
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)

    global_embeddings_c = local_embeddings_c / counts.clamp_min(1.0).unsqueeze(1)
    global_embeddings_a = local_embeddings_a / counts.clamp_min(1.0).unsqueeze(1)
    global_features = local_features / counts.clamp_min(1.0).view(-1, 1, 1, 1)
        
    return global_embeddings_c, global_embeddings_a, global_features


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    unique_dataset: UniqueFingerprintDataset,
    global_embeddings_c: torch.Tensor,
    global_embeddings_a: torch.Tensor,
    global_features: torch.Tensor,
    device: torch.device,
    epoch: int,
) -> tuple[float, float]:
    import refinement
    all_scores, all_labels = [], []

    pbar = tqdm(
        val_loader,
        desc=f"Epoch {epoch:03d} [val evaluate]",
        leave=False,
        unit="batch",
        disable=not is_main(),
    )

    w1, w2 = 0.5, 0.5 # Fusion weights for CNN and Attention embeddings

    for idx_a, idx_b, labels in pbar:
        idx_a = idx_a.to(device, non_blocking=True)
        idx_b = idx_b.to(device, non_blocking=True)

        emb_c_a = global_embeddings_c[idx_a]
        emb_c_b = global_embeddings_c[idx_b]
        
        emb_a_a = global_embeddings_a[idx_a]
        emb_a_b = global_embeddings_a[idx_b]

        cos_sim_c = (emb_c_a * emb_c_b).sum(dim=1)
        cos_sim_a = (emb_a_a * emb_a_b).sum(dim=1)
        
        cos_sim = w1 * cos_sim_c + w2 * cos_sim_a
        
        # Test-time refinement (Algorithm 1)
        # sl, sh = 0.3, 0.6
        # w3, w4 = 0.5, 0.5
        # for i in range(len(cos_sim)):
        #     s = cos_sim[i].item()
        #     if sl <= s <= sh:
        #         idx1 = idx_a[i].item()
        #         idx2 = idx_b[i].item()
        #         I1 = unique_dataset[idx1][1].unsqueeze(0).to(device)
        #         I2 = unique_dataset[idx2][1].unsqueeze(0).to(device)
                
        #         L1 = global_features[idx1]
        #         L2 = global_features[idx2]
                
        #         s_prime = refinement.apply_refinement(model, I1, I2, L1, L2, device)
        #         if s_prime is not None:
        #             cos_sim[i] = w3 * s + w4 * s_prime

        all_scores.append(cos_sim.cpu().numpy())
        all_labels.append(labels.numpy())

    metrics = compute_authentication_metrics(
        np.concatenate(all_scores), np.concatenate(all_labels)
    )

    return metrics


# ---------------------------------------------------------------------------
# Checkpoint Management
# ---------------------------------------------------------------------------


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    arcface_loss_c: torch.nn.Module,
    arcface_loss_a: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: torch.amp.GradScaler,
) -> tuple[int, float]:
    start_epoch = 1
    best_eer = float("inf")

    if os.path.isfile(path):
        if is_main():
            print(f"=> Loading checkpoint '{path}'")
        ckpt_dict = torch.load(path, map_location="cpu")
        _unwrap(model).load_state_dict(ckpt_dict["model"])
        _unwrap(arcface_loss_c).load_state_dict(ckpt_dict["arcface_c"])
        _unwrap(arcface_loss_a).load_state_dict(ckpt_dict["arcface_a"])
        if "optimizer" in ckpt_dict:
            optimizer.load_state_dict(ckpt_dict["optimizer"])
        if "scheduler" in ckpt_dict:
            scheduler.load_state_dict(ckpt_dict["scheduler"])
        if "scaler" in ckpt_dict:
            scaler.load_state_dict(ckpt_dict["scaler"])
        if "epoch" in ckpt_dict:
            start_epoch = ckpt_dict["epoch"] + 1
        if "eer" in ckpt_dict:
            best_eer = ckpt_dict["eer"]

        if is_main():
            print(f"=> Loaded checkpoint (epoch {start_epoch - 1})")
    else:
        if is_main():
            print(f"=> No checkpoint found at '{path}'")

    return start_epoch, best_eer


def save_checkpoint(
    path: str,
    epoch: int,
    model: torch.nn.Module,
    arcface_loss_c: torch.nn.Module,
    arcface_loss_a: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: torch.amp.GradScaler,
    eer: float,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model": _unwrap(model).state_dict(),
            "arcface_c": _unwrap(arcface_loss_c).state_dict(),
            "arcface_a": _unwrap(arcface_loss_a).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "eer": eer,
        },
        path,
    )
    tqdm.write(f"  [checkpoint] saved → {path}")

    if wandb.run is not None:
        artifact = wandb.Artifact(
            name=f"checkpoint-epoch{epoch:03d}",
            type="checkpoint",
            metadata={"epoch": epoch, "eer": eer},
        )
        artifact.add_file(path)
        wandb.log_artifact(artifact)
        tqdm.write("  [wandb] checkpoint artifact logged")


def save_best(
    ckpt_dir: str,
    best_name: str,
    epoch: int,
    model: torch.nn.Module,
    eer: float,
) -> None:
    path = os.path.join(ckpt_dir, best_name)
    torch.save(
        {
            "epoch": epoch,
            "model": _unwrap(model).state_dict(),
            "eer": eer,
        },
        path,
    )
    tqdm.write(f"  [best model] EER={eer:.2%} saved → {path}")

    if wandb.run is not None:
        artifact = wandb.Artifact(
            name="best-model",
            type="model",
            metadata={"epoch": epoch, "eer": eer},
        )
        artifact.add_file(path)
        wandb.log_artifact(artifact)
        wandb.run.summary["best_val_eer"] = eer
        wandb.run.summary["best_val_eer_epoch"] = epoch
        tqdm.write("  [wandb] best-model artifact logged")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: torch.nn.Module,
    arcface_loss_c: torch.nn.Module,
    arcface_loss_a: torch.nn.Module,
    train_loader: DataLoader,
    train_sampler: DistributedSampler,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    arcface_loss_c.train()
    arcface_loss_a.train()

    if train_sampler:
        train_sampler.set_epoch(epoch)

    total_loss = 0.0
    all_params = list(model.parameters()) + list(arcface_loss_c.parameters()) + list(arcface_loss_a.parameters())

    pbar = tqdm(
        train_loader,
        desc=f"[train] Epoch {epoch:03d}",
        leave=False,
        unit="batch",
        disable=not is_main(),
    )

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda"):
            zc, za, _ = model(images)
            
            loss_c, _ = arcface_loss_c(zc, labels)
            loss_a, _ = arcface_loss_a(za, labels)
            
            loss = loss_c + loss_a

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        new_scale = scaler.get_scale()
        if new_scale >= old_scale:
            scheduler.step()

        loss_val = loss.item()
        lr_val = scheduler.get_last_lr()[0]
        total_loss += loss_val

        pbar.set_postfix(loss=f"{loss_val:.4f}", lr=f"{lr_val:.2e}")

    return total_loss / len(train_loader)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(cfg: dict, no_wandb: bool = False, checkpoint: str = None) -> None:
    general_cfg = cfg["general"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    loss_cfg = cfg["loss"]
    opt_cfg = cfg["optimizer"]
    sched_cfg = cfg["scheduler"]
    output_cfg = cfg["output"]
    wandb_cfg = cfg["wandb"]
    eval_cfg = cfg["evaluation"]

    # ── DDP init ────────────────────────────────────────────────────────────
    local_rank, world_size = setup_ddp()
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    set_seed(general_cfg["seed"] + local_rank)

    if is_main():
        print(f"Device: {device}  |  world_size: {world_size}")

    # ── Wandb ─────────────────────────────────────────────────────────────
    if is_main() and not no_wandb and wandb_cfg.get("api_key"):
        wandb.login(key=wandb_cfg["api_key"])
        wandb.init(project=wandb_cfg["project"], config=cfg)

    # ── Transforms ────────────────────────────────────────────────────────
    train_transform, eval_transform, _ = get_transforms(data_cfg["transform_name"])

    # ── Datasets ──────────────────────────────────────────────────────────
    train_dataset = RecogTrainingDataset(
        split_path=data_cfg["split_path"],
        transform=train_transform,
    )
    val_dataset = AuthenticationEvaluationDataset(
        split_path=data_cfg["split_path"],
        split="val",
        n_genuine_impressions=data_cfg["n_genuine_impressions"],
        n_impostor_impressions=data_cfg["n_impostor_impressions"],
        impostor_mode=data_cfg["impostor_mode"],
        n_impostor_subset=data_cfg["n_impostor_subset"],
        seed=general_cfg["seed"],
    )
    unique_val_dataset = UniqueFingerprintDataset(
        idx_to_path=val_dataset.idx_to_path, transform=eval_transform
    )

    if is_main():
        print(f"\n{train_dataset}")
        print(f"{val_dataset}")

    # ── Dataloaders ───────────────────────────────────────────────────────
    global_batch_size = train_cfg["recog_batch_size"]
    local_batch_size = max(1, global_batch_size // world_size)
    
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=True,
        seed=general_cfg["seed"],
    ) if dist.is_available() and dist.is_initialized() else None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        sampler=train_sampler,
        num_workers=train_cfg["num_workers"],
        pin_memory=train_cfg["pin_memory"],
        shuffle=(train_sampler is None)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_cfg["recog_batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        pin_memory=train_cfg["pin_memory"],
    )

    unique_val_sampler = DistributedSampler(
        unique_val_dataset,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=False,
        drop_last=False,
    )
    
    unique_val_loader = DataLoader(
        unique_val_dataset,
        batch_size=local_batch_size,
        sampler=unique_val_sampler,
        num_workers=train_cfg["num_workers"],
        pin_memory=train_cfg["pin_memory"],
        shuffle=False
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = get_model(model_cfg["model_name"], model_cfg).to(device)
    
    if model_cfg.get("ckpt_path"):
        model.load_state_dict(
            torch.load(model_cfg["ckpt_path"], map_location="cpu")["model"]
        )
        tqdm.write(f"  [model] loaded from {model_cfg['ckpt_path']}")
        
    if dist.is_available() and dist.is_initialized():
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
        )

    if is_main():
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"[model] {model_cfg['model_name']} ({n_params:.2f}M params)")

    # ── Loss ──────────────────────────────────────────────────────────────
    n_ids = train_dataset.n_ids
    arcface_loss_c = ArcFaceLoss(
        embed_dim=384,
        num_classes=n_ids,
        margin=loss_cfg["margin"],
        scale=loss_cfg["scale"],
    ).to(device)
    
    arcface_loss_a = ArcFaceLoss(
        embed_dim=384,
        num_classes=n_ids,
        margin=loss_cfg["margin"],
        scale=loss_cfg["scale"],
    ).to(device)
    
    if dist.is_available() and dist.is_initialized():
        arcface_loss_c = DDP(arcface_loss_c, device_ids=[local_rank], output_device=local_rank)
        arcface_loss_a = DDP(arcface_loss_a, device_ids=[local_rank], output_device=local_rank)

    # ── Optimizer, Scheduler, Scaler ──────────────────────────────────────
    all_parameters = list(model.parameters()) + list(arcface_loss_c.parameters()) + list(arcface_loss_a.parameters())
    optimizer = get_optimizer(opt_cfg["opt_name"], all_parameters, opt_cfg)

    scheduler = get_scheduler(
        sched_cfg["sched_name"],
        optimizer,
        iters=len(train_loader),
        epochs=train_cfg["epochs"],
        sched_cfg=sched_cfg,
    )

    scaler = torch.amp.GradScaler("cuda") if torch.cuda.is_available() else torch.cuda.amp.GradScaler()

    # ── Training loop ─────────────────────────────────────────────────────
    start_epoch = 1
    best_eer = float("inf")

    if checkpoint is not None:
        start_epoch, best_eer = load_checkpoint(
            checkpoint, model, arcface_loss_c, arcface_loss_a, optimizer, scheduler, scaler
        )

    if is_main():
        if not wandb.run:
            history = {"epoch": [], "train_loss": [], "val_eer": []}

        os.makedirs(output_cfg["checkpoint_dir"], exist_ok=True)

        print("\n" + "=" * 60)
        print("Starting recognition training")
        print("=" * 60)

    epoch_pbar = tqdm(
        range(start_epoch, train_cfg["epochs"] + 1),
        desc="Training",
        unit="epoch",
        disable=not is_main(),
    )

    for epoch in epoch_pbar:
        avg_loss = train_one_epoch(
            model,
            arcface_loss_c,
            arcface_loss_a,
            train_loader,
            train_sampler,
            optimizer,
            scheduler,
            scaler,
            device,
            epoch,
        )

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        global_embeddings_c, global_embeddings_a, global_features = get_embeddings(
            _unwrap(model), unique_val_loader, device, epoch
        )

        if is_main():
            metrics = evaluate(model, val_loader, unique_val_dataset, global_embeddings_c, global_embeddings_a, global_features, device, epoch)

            epoch_pbar.set_postfix(
                loss=f"{avg_loss:.4f}",
                eer=f"{metrics['eer']:.2%}",
                thr=f"{metrics['eer_threshold']:.4f}",
            )
            tqdm.write(
                f"Epoch {epoch:03d} | avg loss: {avg_loss:.4f} | "
                f"val EER: {metrics['eer']:.2%}  (thr={metrics['eer_threshold']:.4f}) | "
                f"val TAR@FAR=0.1: {metrics['tar_at_far_0.1']:.2%} | "
                f"val TAR@FAR=0.01: {metrics['tar_at_far_0.01']:.2%} | "
                f"val TAR@FAR=0.001: {metrics['tar_at_far_0.001']:.2%}"
            )

            if wandb.run is not None:
                wandb.log(
                    {
                        "train/loss": avg_loss,
                        "val/eer": metrics["eer"],
                        "val/eer_threshold": metrics["eer_threshold"],
                        "val/tar_at_far_0.1": metrics["tar_at_far_0.1"],
                        "val/tar_at_far_0.01": metrics["tar_at_far_0.01"],
                        "val/tar_at_far_0.001": metrics["tar_at_far_0.001"],
                        "epoch": epoch,
                    }
                )
            else:
                history["epoch"].append(epoch)
                history["train_loss"].append(avg_loss)
                history["val_eer"].append(metrics["eer"])

            if metrics["eer"] < best_eer:
                best_eer = metrics["eer"]
                save_best(
                    output_cfg["checkpoint_dir"],
                    output_cfg["best_model_name"],
                    epoch,
                    model,
                    metrics["eer"],
                )

            if epoch % train_cfg["checkpoint_interval"] == 0:
                ckpt_path = os.path.join(
                    output_cfg["checkpoint_dir"], f"checkpoint_epoch{epoch:03d}.pth"
                )
                save_checkpoint(
                    ckpt_path,
                    epoch,
                    model,
                    arcface_loss_c,
                    arcface_loss_a,
                    optimizer,
                    scheduler,
                    scaler,
                    metrics["eer"],
                )

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    if is_main():
        print("=" * 60)
        print(f"Training complete. Best val EER: {best_eer:.2%}")
        print("=" * 60)

        if wandb.run is not None:
            wandb.finish()
        else:
            import matplotlib.pyplot as plt

            fig, ax1 = plt.subplots(figsize=(10, 5))
            ax2 = ax1.twinx()

            ax1.plot(history["epoch"], history["train_loss"], "g-", label="Train Loss")
            ax2.plot(history["epoch"], history["val_eer"], "b-", label="Val EER")

            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Train Loss", color="g")
            ax2.set_ylabel("Val EER", color="b")

            lines_1, labels_1 = ax1.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

            plt.title("Recognition Training History")
            plot_path = os.path.join(
                output_cfg["checkpoint_dir"], "training_history.png"
            )
            plt.savefig(plot_path)
            plt.close()
            print(f"Saved training history plot to {plot_path}")

    cleanup_ddp()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recognition Training")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    main(cfg, no_wandb=args.no_wandb, checkpoint=args.checkpoint)

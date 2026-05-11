import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import AuthenticationEvaluationDataset, IdentificationEvaluationDataset, UniqueFingerprintDataset
from metrics import compute_authentication_metrics, compute_identification_metrics
from models import get_model
from transforms import get_transforms


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def get_embeddings(
    model: torch.nn.Module,
    unique_loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()

    embed_dim_c = model.zc_linear.out_features
    embed_dim_a = model.attention_head.vit_feature.out_features
    n_unique_images = len(unique_loader.dataset)

    global_embeddings_c = torch.zeros((n_unique_images, embed_dim_c), device=device)
    global_embeddings_a = torch.zeros((n_unique_images, embed_dim_a), device=device)
    global_embeddings_l = None

    pbar = tqdm(
        unique_loader,
        desc="[extracting embeddings]",
        leave=False,
        unit="batch",
    )

    for idxs, imgs in pbar:
        imgs = imgs.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
            zc, za, local_features = model(imgs)
        
        zc = F.normalize(zc, dim=1).float()
        za = F.normalize(za, dim=1).float()

        if global_embeddings_l is None:
            _, C, H, W = local_features.shape
            global_embeddings_l = torch.zeros((n_unique_images, C, H, W), device="cpu", dtype=torch.float16)

        global_embeddings_c[idxs] = zc
        global_embeddings_a[idxs] = za
        global_embeddings_l[idxs] = local_features.to("cpu", dtype=torch.float16)

    return global_embeddings_c, global_embeddings_a, global_embeddings_l


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    unique_dataset: UniqueFingerprintDataset,
    global_embeddings_c: torch.Tensor,
    global_embeddings_a: torch.Tensor,
    global_embeddings_l: torch.Tensor,
    device: torch.device,
) -> tuple[float, float]:
    import refinement
    all_scores, all_labels = [], []

    pbar = tqdm(
        val_loader,
        desc="[evaluating pairs]",
        leave=False,
        unit="batch",
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
        sl, sh = 0.3, 0.6
        w3, w4 = 0.5, 0.5
        for i in range(len(cos_sim)):
            s = cos_sim[i].item()
            if sl <= s <= sh:
                idx1 = idx_a[i].item()
                idx2 = idx_b[i].item()
                I1 = unique_dataset[idx1][1].unsqueeze(0).to(device)
                I2 = unique_dataset[idx2][1].unsqueeze(0).to(device)
                
                L1 = global_embeddings_l[idx1].unsqueeze(0).to(device, dtype=torch.float32)
                L2 = global_embeddings_l[idx2].unsqueeze(0).to(device, dtype=torch.float32)
                
                s_prime = refinement.apply_refinement(model, I1, I2, L1, L2, device)
                if s_prime is not None:
                    cos_sim[i] = w3 * s + w4 * s_prime

        all_scores.append(cos_sim.cpu().numpy())
        all_labels.append(labels.numpy())

    metrics = compute_authentication_metrics(
        np.concatenate(all_scores), np.concatenate(all_labels)
    )

    return metrics


@torch.no_grad()
def evaluate_identification(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    dataset: torch.utils.data.Dataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()

    all_embs_c = []
    all_embs_a = []
    all_embs_l = []
    all_labels = []
    all_indices = []

    pbar = tqdm(
        loader,
        desc="[identification inference]",
        leave=False,
        unit="batch",
    )

    with torch.no_grad():
        for imgs, labels, idx in pbar:
            imgs = imgs.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                zc, za, l = model(imgs)

            zc = F.normalize(zc, dim=1).cpu()
            za = F.normalize(za, dim=1).cpu()
            l = l.cpu()

            all_embs_c.append(zc)
            all_embs_a.append(za)
            all_embs_l.append(l)
            all_labels.extend(labels.numpy())
            all_indices.extend(idx.numpy())

    all_embs_c = torch.cat(all_embs_c, dim=0).numpy()
    all_embs_a = torch.cat(all_embs_a, dim=0).numpy()
    all_embs_l = torch.cat(all_embs_l, dim=0)
    all_labels = np.array(all_labels)
    all_indices = np.array(all_indices)

    sort_order = np.argsort(all_indices)
    all_embs_c = all_embs_c[sort_order]
    all_embs_a = all_embs_a[sort_order]
    all_embs_l = all_embs_l[sort_order]
    all_labels = all_labels[sort_order]

    n_gal = dataset.n_gallery
    gallery_embs_c = all_embs_c[:n_gal]
    gallery_embs_a = all_embs_a[:n_gal]
    gallery_embs_l = all_embs_l[:n_gal]
    gallery_labels = all_labels[:n_gal]

    probe_embs_c = all_embs_c[n_gal:]
    probe_embs_a = all_embs_a[n_gal:]
    probe_embs_l = all_embs_l[n_gal:]
    probe_labels = all_labels[n_gal:]

    sim_mat_c = np.dot(probe_embs_c, gallery_embs_c.T)
    sim_mat_a = np.dot(probe_embs_a, gallery_embs_a.T)

    w1, w2 = 0.5, 0.5
    sim_mat = w1 * sim_mat_c + w2 * sim_mat_a

    # Test-time refinement (Algorithm 1)
    import refinement
    sl, sh = 0.3, 0.6
    w3, w4 = 0.5, 0.5
    
    for i in tqdm(range(dataset.n_probes), desc="[identification refinement]", leave=False):
        for j in range(dataset.n_gallery):
            s = sim_mat[i, j]
            if sl <= s <= sh:
                idx_probe = n_gal + i
                idx_gallery = j
                
                I1 = dataset[idx_probe][0].unsqueeze(0).to(device)
                I2 = dataset[idx_gallery][0].unsqueeze(0).to(device)
                
                L1 = probe_embs_l[i].unsqueeze(0).to(device)
                L2 = gallery_embs_l[j].unsqueeze(0).to(device)
                
                s_prime = refinement.apply_refinement(model, I1, I2, L1, L2, device)
                if s_prime is not None:
                    sim_mat[i, j] = w3 * s + w4 * s_prime

    return sim_mat, probe_labels, gallery_labels


def main(cfg: dict, checkpoint: str, split_path: str) -> None:
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    eval_cfg = cfg["evaluation"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Transforms ────────────────────────────────────────────────────────
    _, eval_transform, _ = get_transforms(data_cfg["transform_name"])

    # ── Datasets ──────────────────────────────────────────────────────────
    test_dataset = AuthenticationEvaluationDataset(
        split_path=split_path,
        split="test",
        n_genuine_impressions=data_cfg["n_genuine_impressions"],
        n_impostor_impressions=data_cfg["n_impostor_impressions"],
        impostor_mode=data_cfg["impostor_mode"],
        n_impostor_subset=data_cfg["n_impostor_subset"],
        seed=cfg["general"]["seed"],
    )
    unique_test_dataset = UniqueFingerprintDataset(
        idx_to_path=test_dataset.idx_to_path, transform=eval_transform
    )
    identification_dataset = IdentificationEvaluationDataset(
        split_path=split_path,
        split="test",
        gallery_per_id=data_cfg["gallery_per_id"],
        probe_per_id=data_cfg["probe_per_id"],
        transform=eval_transform,
        seed=cfg["general"]["seed"],
    )

    print(f"\n{test_dataset}")
    print(f"\n{identification_dataset}")

    # ── Dataloaders ───────────────────────────────────────────────────────
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_cfg["recog_batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=cfg["training"]["pin_memory"],
    )

    unique_test_loader = DataLoader(
        unique_test_dataset,
        batch_size=cfg["training"]["recog_batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=cfg["training"]["pin_memory"],
    )

    identification_loader = DataLoader(
        identification_dataset,
        batch_size=cfg["training"]["recog_batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=cfg["training"]["pin_memory"],
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = get_model(model_cfg["model_name"], model_cfg).to(device)
    
    if checkpoint and os.path.isfile(checkpoint):
        print(f"=> Loading checkpoint '{checkpoint}'")
        ckpt_dict = torch.load(checkpoint, map_location="cpu")
        
        # Load weights, ignore classification heads since we only need embeddings
        model.load_state_dict(ckpt_dict["model"], strict=False)
        print("=> Loaded checkpoint")
    else:
        print(f"=> WARNING: No checkpoint found at '{checkpoint}'")

    model.eval()

    # ── Evaluation loop ───────────────────────────────────────────────────
    global_embeddings_c, global_embeddings_a, global_embeddings_l = get_embeddings(
        model, unique_test_loader, device
    )

    auth_metrics = evaluate(model, test_loader, unique_test_dataset, global_embeddings_c, global_embeddings_a, global_embeddings_l, device)
    
    sim_mat, probe_labels, gallery_labels = evaluate_identification(
        model, identification_loader, device, identification_dataset
    )
    id_metrics = compute_identification_metrics(sim_mat, probe_labels, gallery_labels)

    print("\n" + "=" * 60)
    print("Authentication Metrics:")
    print("-" * 60)
    print(f"Test EER: {auth_metrics['eer']:.2%}  (thr={auth_metrics['eer_threshold']:.4f})")
    print(f"Test TAR@FAR=0.1: {auth_metrics['tar_at_far_0.1']:.2%}")
    print(f"Test TAR@FAR=0.01: {auth_metrics['tar_at_far_0.01']:.2%}")
    print(f"Test TAR@FAR=0.001: {auth_metrics['tar_at_far_0.001']:.2%}")
    print("=" * 60)
    print("Identification Metrics:")
    print("-" * 60)
    print(f"Identities: {identification_dataset.n_ids:,}")
    print(f"Gallery samples: {identification_dataset.n_gallery:,}, Probe samples: {identification_dataset.n_probes:,}")
    print("-" * 60)
    print(f"Rank-1 Accuracy : {id_metrics['rank_1']:.2%}")
    print(f"Rank-5 Accuracy : {id_metrics['rank_5']:.2%}")
    print(f"Rank-10 Accuracy: {id_metrics['rank_10']:.2%}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recognition Evaluation")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--split-path",
        type=str,
        required=True,
        help="Path to the split JSON file",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to checkpoint to evaluate",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    main(cfg, checkpoint=args.checkpoint_path, split_path=args.split_path)

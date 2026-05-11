import numpy as np
from sklearn.metrics import accuracy_score, auc, roc_curve


def compute_pad_metrics(probabilities: np.ndarray, labels: np.ndarray) -> dict:
    # Compute ROC
    fpr, tpr, thresholds = roc_curve(labels, probabilities)

    # Convert to PAD metrics
    apcer = 1 - tpr  # FNR
    bpcer = fpr  # FPR
    ace = (apcer + bpcer) / 2

    # Find best threshold (DISCRETE)
    idx = np.argmin(ace)
    threshold = thresholds[idx]

    # Compute accuracy at this threshold
    predictions = (probabilities >= threshold).astype(int)
    accuracy = accuracy_score(labels, predictions)

    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "ace": ace[idx],
        "apcer": apcer[idx],
        "bpcer": bpcer[idx],
    }


def compute_authentication_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
) -> dict:
    # ROC
    fmr, tar, thrs = roc_curve(labels, scores, pos_label=1)
    fnmr = 1.0 - tar

    # Remove sklearn's artificial first point
    if len(thrs) > 1 and np.isinf(thrs[0]):
        fmr = fmr[1:]
        tar = tar[1:]
        fnmr = fnmr[1:]
        thrs = thrs[1:]

    # EER (interpolation)
    diff = fmr - fnmr
    idx1_candidates = np.where(diff >= 0)[0]

    if len(idx1_candidates) == 0:
        # fallback: closest point
        eer_idx = int(np.argmin(np.abs(diff)))
        eer = (fmr[eer_idx] + fnmr[eer_idx]) / 2.0
        eer_thr = thrs[eer_idx]
    else:
        idx1 = idx1_candidates[0]
        idx0 = idx1 - 1 if idx1 > 0 else idx1

        x0, y0 = fmr[idx0], fnmr[idx0]
        x1, y1 = fmr[idx1], fnmr[idx1]

        if idx0 == idx1:
            eer = (x0 + y0) / 2.0
            eer_thr = thrs[idx0]
        else:
            den = (x1 - x0) - (y1 - y0)
            if abs(den) < 1e-12:
                eer = (x0 + y0) / 2.0
                eer_thr = thrs[idx0]
            else:
                t = (y0 - x0) / den
                eer = x0 + t * (x1 - x0)
                eer_thr = thrs[idx0] + t * (thrs[idx1] - thrs[idx0])

    # AUC
    auc_roc = float(auc(fmr, tar))

    # TAR@FAR (interpolation)
    def interp_tar_at_far(target_far: float) -> float:
        # exact match
        mask = (fmr == target_far)
        if mask.any():
            return float(tar[mask].max())

        idx = np.searchsorted(fmr, target_far, side="right")

        if idx == 0:
            return float(tar[0])
        if idx >= len(fmr):
            return float(tar[-1])

        x0, x1 = fmr[idx - 1], fmr[idx]
        y0, y1 = tar[idx - 1], tar[idx]

        # avoid division by zero
        if abs(x1 - x0) < 1e-12:
            return float(y0)

        t = (target_far - x0) / (x1 - x0)
        return float(y0 + t * (y1 - y0))

    tar_at_far = {
        0.1: interp_tar_at_far(0.1),
        0.01: interp_tar_at_far(0.01),
        0.001: interp_tar_at_far(0.001),
    }

    return {
        "eer": float(eer),
        "eer_threshold": float(eer_thr),
        "auc": auc_roc,
        "thresholds": thrs.tolist(),
        "fmr": fmr.tolist(),
        "tar": tar.tolist(),
        "fnmr": fnmr.tolist(),
        "tar_at_far_0.1": tar_at_far[0.1],
        "tar_at_far_0.01": tar_at_far[0.01],
        "tar_at_far_0.001": tar_at_far[0.001],
    }


def compute_identification_metrics(
    sim_mat: np.ndarray,
    probe_labels: np.ndarray,
    gallery_labels: np.ndarray,
    top_k: tuple = (1, 5, 10)
) -> dict:
    sorted_indices = np.argsort(-sim_mat, axis=1)

    pred_labels = gallery_labels[sorted_indices]

    matches = (pred_labels == probe_labels[:, None])

    metrics = {}
    for k in top_k:
        rank_k = matches[:, :k].any(axis=1).mean()
        metrics[f"rank_{k}"] = float(rank_k)

    return metrics
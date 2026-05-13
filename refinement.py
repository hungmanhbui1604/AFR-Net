import cv2
import numpy as np
import torch
import torch.nn.functional as F

def get_correspondences(L1, L2):
    """
    L1, L2: [1024, 14, 14] local feature maps for a single pair of images
    Returns: kp1, kp2 as Nx2 arrays of (x, y) coordinates
    """
    C, H, W = L1.shape
    
    L1_flat = L1.view(C, H*W).transpose(0, 1) # [196, 1024]
    L2_flat = L2.view(C, H*W).transpose(0, 1) # [196, 1024]
    
    L1_norm = F.normalize(L1_flat, dim=1)
    L2_norm = F.normalize(L2_flat, dim=1)
    
    sim = torch.mm(L1_norm, L2_norm.transpose(0, 1)) # [196, 196]
    
    # For each patch in L1, find the best matching patch in L2
    max_sim, max_idx = sim.max(dim=1) # [196]
    
    # Convert index 0..195 to (x, y) coordinates
    # The center of patch (y, x) where y = idx // 14, x = idx % 14
    # Image size is 224x224, patch size is 16x16
    # Center = (x * 16 + 8, y * 16 + 8)
    
    y1 = torch.arange(H * W) // W
    x1 = torch.arange(H * W) % W
    kp1 = torch.stack([x1, y1], dim=1).float() * 16.0 + 8.0 # [196, 2]
    
    y2 = max_idx // W
    x2 = max_idx % W
    kp2 = torch.stack([x2, y2], dim=1).float() * 16.0 + 8.0 # [196, 2]
    
    return kp1.cpu().numpy(), kp2.cpu().numpy()

def homography_ok(M):
    if M is None:
        return False
    # Check determinant of the affine part
    det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    if det < 0.2 or det > 5.0: # Extreme scaling or reflection
        return False
    # Check translation limits (e.g. shouldn't translate completely out of bounds)
    if abs(M[0, 2]) > 200 or abs(M[1, 2]) > 200:
        return False
    return True

def crop_overlap(I1_prime, I2):
    """
    I1_prime: [3, 224, 224] tensor, warped I1
    I2: [3, 224, 224] tensor
    Returns C1, C2: [1, 3, 224, 224] cropped and resized tensors
    """
    # Assuming pixels outside the valid warped area are exactly 0
    # Or we could have warped a mask of 1s
    # A simple proxy is summing over channels and checking for > 0
    # Note: image might have 0 pixels naturally, so it's an approximation
    # A more robust way is to just assume I2 is full size, and I1_prime has a valid boundary
    valid_mask = (I1_prime.abs().sum(dim=0) > 1e-4).float().cpu().numpy()
    
    # If the mask is completely empty or almost empty, fallback
    if valid_mask.sum() < 400: # less than 20x20 area
        return None, None
        
    rows = np.any(valid_mask, axis=1)
    cols = np.any(valid_mask, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        return None, None
        
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    
    # Add a small buffer but keep within [0, 224]
    ymin = max(0, ymin - 4)
    ymax = min(224, ymax + 4)
    xmin = max(0, xmin - 4)
    xmax = min(224, xmax + 4)
    
    # If the crop area is too small, fallback
    if (ymax - ymin) < 32 or (xmax - xmin) < 32:
        return None, None
        
    C1 = I1_prime[:, ymin:ymax, xmin:xmax].unsqueeze(0)
    C2 = I2[:, ymin:ymax, xmin:xmax].unsqueeze(0)
    
    # Resize back to 224x224
    C1 = F.interpolate(C1, size=(224, 224), mode='bilinear', align_corners=False)
    C2 = F.interpolate(C2, size=(224, 224), mode='bilinear', align_corners=False)
    
    return C1, C2

def apply_refinement(model, I1, I2, L1, L2, device):
    """
    Applies test-time refinement for a single pair of images.
    I1, I2: [1, 3, 224, 224] tensors
    L1, L2: [1024, 14, 14] tensors
    Returns: new similarity score, or None if fallback
    """
    kp1, kp2 = get_correspondences(L1, L2)
    
    # Find homography mapping I1 to I2
    M, mask = cv2.findHomography(kp1, kp2, cv2.RANSAC, 5.0)
    
    if not homography_ok(M):
        return None
        
    # Warp I1
    # We convert to numpy, warp, convert back. Alternatively we could use torch grid_sample
    I1_np = I1.squeeze(0).cpu().numpy().transpose(1, 2, 0) # [224, 224, 3]
    I1_prime_np = cv2.warpPerspective(I1_np, M, (224, 224), flags=cv2.INTER_LINEAR)
    
    I1_prime = torch.from_numpy(I1_prime_np.transpose(2, 0, 1)).to(device) # [3, 224, 224]
    I2_tensor = I2.squeeze(0) # [3, 224, 224]
    
    C1, C2 = crop_overlap(I1_prime, I2_tensor)
    if C1 is None or C2 is None:
        return None
        
    # Re-evaluate embeddings
    with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
        zc1, za1, _ = model(C1)
        zc2, za2, _ = model(C2)
        
    zc1 = F.normalize(zc1, dim=1).float()
    za1 = F.normalize(za1, dim=1).float()
    zc2 = F.normalize(zc2, dim=1).float()
    za2 = F.normalize(za2, dim=1).float()
    
    cos_c = (zc1 * zc2).sum().item()
    cos_a = (za1 * za2).sum().item()
    
    return 0.5 * cos_c + 0.5 * cos_a

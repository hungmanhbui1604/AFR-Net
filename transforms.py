import cv2
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image


class ExtractFingerprintForeground:
    def __init__(self, padding: int = 10):
        self.padding = padding

    def __call__(self, image: Image.Image) -> Image.Image:
        img_array = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        # Improve contrast
        gray = cv2.equalizeHist(gray)

        # Blur to reduce noise
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Combine Otsu + Adaptive threshold
        _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        adaptive = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )

        # Combine both masks (robust)
        thresh = cv2.bitwise_or(otsu, adaptive)

        # Add gradient-based mask (texture)
        grad = cv2.Laplacian(blur, cv2.CV_8U)
        _, grad_mask = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        thresh = cv2.bitwise_or(thresh, grad_mask)

        # Morphological cleanup
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return image

        # Filter contours by area (avoid noise)
        img_area = gray.shape[0] * gray.shape[1]
        valid_contours = [c for c in contours if cv2.contourArea(c) > 0.01 * img_area]

        if not valid_contours:
            return image

        largest_cnt = max(valid_contours, key=cv2.contourArea)

        # Convex hull
        hull = cv2.convexHull(largest_cnt)

        # Mask from hull
        hull_mask = np.zeros_like(gray)
        cv2.drawContours(hull_mask, [hull], -1, 255, -1)

        # White background instead of black
        white_bg = np.full_like(img_array, 255)
        masked = white_bg.copy()
        masked[hull_mask == 255] = img_array[hull_mask == 255]

        # Bounding box from hull
        x, y, w, h = cv2.boundingRect(hull)
        aspect_ratio = w / h if h != 0 else 0

        if aspect_ratio < 0.3 or aspect_ratio > 3:
            return image

        # Padding
        p = self.padding
        y1, y2 = max(0, y - p), min(img_array.shape[0], y + h + p)
        x1, x2 = max(0, x - p), min(img_array.shape[1], x + w + p)

        # Crop MASKED image (important)
        cropped_array = masked[y1:y2, x1:x2]

        return Image.fromarray(cropped_array)


class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        hp = (max_wh - w) // 2
        vp = (max_wh - h) // 2
        padding = (hp, vp, max_wh - w - hp, max_wh - h - vp)
        return F.pad(image, padding, fill=255, padding_mode="constant")


train_transform = transforms.Compose(
    [
        SquarePad(),
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomRotation(10, fill=255),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

test_transform = transforms.Compose(
    [
        SquarePad(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

infer_transform = transforms.Compose(
    [
        ExtractFingerprintForeground(padding=10),
        SquarePad(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def get_transforms(transform_name: str) -> tuple:
    if transform_name == "dual":
        return train_transform, test_transform, infer_transform

    raise ValueError("Unknown transform_name: " + transform_name)

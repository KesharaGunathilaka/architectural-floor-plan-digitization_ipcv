"""
Preprocessing Pipeline — Phase 2

PURPOSE:
    Transforms raw floor plan PNGs into model-ready tensors at 1024x1024.
    Applies Otsu binarization, morphological closing, letterboxing, and
    bounding box coordinate adjustment.

HOW TO RUN:
    python src/preprocessing/preprocess_pipeline.py
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.logger import get_logger

logger = get_logger(__name__)

# --- Configuration Parameters ---
TARGET_SIZE = 1024
PAD_COLOR = (114, 114, 114)  # YOLO standard gray
MORPH_KERNEL = np.ones((3, 3), np.uint8)


def clean_floorplan_image(img: np.ndarray) -> np.ndarray:
    """
    Cleans structural lines using binarization and morphology.
    """
    # 1. Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Otsu Binarization (Inverted so lines are white, paper is black)
    # This is required because OpenCV morphology works on white objects.
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 3. Morphological Closing (dilate then erode) to patch broken lines
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, MORPH_KERNEL, iterations=1)

    # 4. Invert back (lines are black, paper is white)
    cleaned_inv = cv2.bitwise_not(closed)

    # 5. Light Gaussian Blur to soften jagged binary edges
    blurred = cv2.GaussianBlur(cleaned_inv, (3, 3), 0)

    # 6. Stack back to 3 channels to maintain pre-trained weights compatibility
    final_img = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)

    return final_img


def letterbox_image_and_labels(
    img: np.ndarray, labels: list[str], target_size: int
) -> tuple[np.ndarray, list[str]]:
    """
    Scales image to target_size while preserving aspect ratio, pads with gray,
    and recalculates YOLO bounding box coordinates.
    """
    h, w = img.shape[:2]

    # Calculate scale factor
    scale = min(target_size / w, target_size / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize image
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Calculate padding
    pad_w = target_size - new_w
    pad_h = target_size - new_h
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    # Apply padding
    padded_img = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=PAD_COLOR,
    )

    # Adjust bounding boxes
    new_labels = []
    for line in labels:
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        cls_id = int(parts[0])
        cx_norm, cy_norm, bw_norm, bh_norm = map(float, parts[1:])

        # Denormalize to original pixel coordinates
        cx_orig = cx_norm * w
        cy_orig = cy_norm * h
        bw_orig = bw_norm * w
        bh_orig = bh_norm * h

        # Scale to new pixel coordinates and add padding offsets
        cx_new = (cx_orig * scale) + pad_left
        cy_new = (cy_orig * scale) + pad_top
        bw_new = bw_orig * scale
        bh_new = bh_orig * scale

        # Renormalize based on the new padded target size
        cx_final = cx_new / target_size
        cy_final = cy_new / target_size
        bw_final = bw_new / target_size
        bh_final = bh_new / target_size

        # Ensure boxes don't mathematically bleed over 1.0 due to float rounding
        cx_final = min(max(cx_final, 0.0), 1.0)
        cy_final = min(max(cy_final, 0.0), 1.0)

        new_labels.append(
            f"{cls_id} {cx_final:.6f} {cy_final:.6f} {bw_final:.6f} {bh_final:.6f}"
        )

    return padded_img, new_labels


def process_dataset():
    splits = ["train", "val", "test"]
    base_dir = Path("data/yolo_dataset")
    output_dir = Path("data/yolo_dataset_processed")

    for split in splits:
        img_dir = base_dir / "images" / split
        label_dir = base_dir / "labels" / split

        out_img_dir = output_dir / "images" / split
        out_label_dir = output_dir / "labels" / split

        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_label_dir.mkdir(parents=True, exist_ok=True)

        images = list(img_dir.glob("*.png"))
        if not images:
            continue

        logger.info(f"Preprocessing {split} set ({len(images)} images)...")

        for img_path in tqdm(images, desc=split):
            label_path = label_dir / (img_path.stem + ".txt")

            # Read image and labels
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            labels = []
            if label_path.exists():
                with open(label_path, "r") as f:
                    labels = f.readlines()

            # 1. Clean the drawing
            cleaned_img = clean_floorplan_image(img)

            # 2. Letterbox and adjust coordinates
            final_img, final_labels = letterbox_image_and_labels(
                cleaned_img, labels, TARGET_SIZE
            )

            # 3. Save outputs
            cv2.imwrite(str(out_img_dir / img_path.name), final_img)

            if final_labels:
                with open(out_label_dir / label_path.name, "w") as f:
                    f.write("\n".join(final_labels))


if __name__ == "__main__":
    logger.info(f"Starting pipeline. Target resolution: {TARGET_SIZE}x{TARGET_SIZE}")
    process_dataset()
    logger.info("Preprocessing complete. Dataset is ready for YOLO training.")

# Dataset_Creator.py
# Generate synthetic YOLO dataset for Dino game using real-label stats
# Requirements: numpy, opencv-python (cv2), pillow (PIL), pyyaml
# Put this script in a folder that contains subfolders: "Bird", "Cactus", "Dino"
# and templates: "day_template.jpg" and "night_template.jpg" (both 640x160).
# Also requires "summary.pkl" for label stats.

import pickle
import os
import random
import math
from glob import glob
import numpy as np
from PIL import Image, ImageOps
import sys
from pathlib import Path
import yaml

# -------------------------
# USER-CONFIGURABLE PARAMS
# -------------------------
IMG_WIDTH = 640  # template width
IMG_HEIGHT = 160  # template height
TEMPLATES = ["assets/day_template.jpg", "assets/night_template.jpg"]
BIRD_DIR = "assets/Bird"
CACTUS_DIR = "assets/Cactus"
DINO_DIR = "assets/Dino"
FINAL_DATASET_DIR = "final_dataset"  # direct output to final with splits

# Overlap / instance controls
ALLOW_OVERLAP_PROB = 0.25  # 0..1 probability to allow overlapping objects (0.25 = 25%)
MAX_PLACEMENT_TRIES = 30  # tries to place an object without overlap
RANDOM_SEED = 42  # set None for non-deterministic output
LABEL_DECIMALS = 12  # number of decimals in label txt

# Per-image instance count ranges (you can change these)
CACTUS_PER_IMAGE_RANGE = (3, 10)  # inclusive min,max
PTERO_PER_IMAGE_RANGE = (2, 5)  # inclusive min,max
DINO_PER_IMAGE = 1  # always 1 (game has single dino)


# -------------------------
# Helpers
# -------------------------
def ensure_dirs():
    for split in ["train", "valid", "test"]:
        os.makedirs(os.path.join(FINAL_DATASET_DIR, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(FINAL_DATASET_DIR, split, "labels"), exist_ok=True)


def list_images(folder, exts=("png", "jpg", "jpeg")):
    files = []
    for ext in exts:
        files.extend(glob(os.path.join(folder, f"*.{ext}")))
    return files


def load_image_pil(path, night=False, _cache={}):
    key = (str(path), bool(night))
    if key in _cache:
        # return a copy to avoid accidental mutation
        return _cache[key].copy()

    im = Image.open(path).convert("RGBA")
    if night:
        r, g, b, a = im.split()
        rgb = Image.merge("RGB", (r, g, b))
        inv = ImageOps.invert(rgb)
        im = Image.merge("RGBA", (*inv.split(), a))

    _cache[key] = im.copy()
    return im


def paste_with_alpha(base_pil, obj_pil, x, y):
    # base_pil: RGBA, obj_pil: RGBA, (x,y) top-left coordinates
    base_pil.paste(obj_pil, (int(x), int(y)), obj_pil)
    return base_pil


def rects_iou(a, b):
    # a, b = (x1,y1,x2,y2)
    xa1, ya1, xa2, ya2 = a
    xb1, yb1, xb2, yb2 = b
    inter_w = max(0, min(xa2, xb2) - max(xa1, xb1))
    inter_h = max(0, min(ya2, yb2) - max(ya1, yb1))
    inter_area = inter_w * inter_h
    area_a = max(0, xa2 - xa1) * max(0, ya2 - ya1)
    area_b = max(0, xb2 - xb1) * max(0, yb2 - yb1)
    union = area_a + area_b - inter_area
    if union == 0:
        return 0.0
    return inter_area / union


def clamp(v, a, b):
    return max(a, min(b, v))


# -------------------------
# dataset generation
# -------------------------
def sample_normal_clipped(mu, sigma, lo, hi):
    # sample from normal, clip to [lo,hi]
    if sigma == 0 or math.isnan(sigma):
        return float(clamp(mu, lo, hi))
    v = np.random.normal(mu, sigma)
    return float(clamp(v, lo, hi))


def sample_uniform_clipped(lo, hi):
    return float(np.random.uniform(lo, hi))


def compute_target_pixels(norm_w, norm_h, image_w=IMG_WIDTH, image_h=IMG_HEIGHT):
    return int(round(norm_w * image_w)), int(round(norm_h * image_h))


def place_object_no_overlap(existing_boxes, new_box, allow_overlap_prob):
    # if allow_overlap_prob allows overlap, accept immediately
    if random.random() < allow_overlap_prob:
        return True
    # otherwise require no overlap with any existing box
    for bx in existing_boxes:
        if rects_iou(bx, new_box) > 0.0:
            return False
    return True


def generate_synthetic(summary, num_samples=100, splits=None, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    ensure_dirs()

    # collect object image lists
    bird_imgs = list_images(BIRD_DIR)
    cactus_imgs = list_images(CACTUS_DIR)
    dino_imgs = list_images(DINO_DIR)
    if not bird_imgs or not cactus_imgs or not dino_imgs:
        raise RuntimeError("Make sure Bird/Cactus/Dino folders contain images.")

    for i in range(num_samples):
        # pick template
        template_path = random.choice(TEMPLATES)
        is_night = "night" in os.path.basename(template_path).lower()

        if not os.path.exists(template_path):
            raise RuntimeError(f"Template not found: {template_path}")
        base = load_image_pil(template_path, night=False)

        # record existing boxes for overlap checks in pixel coords (x1,y1,x2,y2)
        existing_boxes = []
        label_lines = []

        # Determine how many instances per class for this image
        n_dino = DINO_PER_IMAGE
        n_cactus = random.randint(CACTUS_PER_IMAGE_RANGE[0], CACTUS_PER_IMAGE_RANGE[1])
        n_ptero = random.randint(PTERO_PER_IMAGE_RANGE[0], PTERO_PER_IMAGE_RANGE[1])

        # Build a placement list (class name, class id, pool, count)
        placement_plan = [
            ('dino', 2, dino_imgs, n_dino),
            ('cactus', 0, cactus_imgs, n_cactus),
            ('ptero', 1, bird_imgs, n_ptero)
        ]

        # Place instances class-by-class
        for name, class_id, pool, count in placement_plan:
            for instance_idx in range(count):
                # choose random object image file (allow reuse)
                obj_path = random.choice(pool)
                obj_pil = load_image_pil(obj_path, night=is_night)  # RGBA
                raw_w, raw_h = obj_pil.size  # raw sizes in px

                # stats fallback: if class not in summary, use rough defaults
                stats = summary.get(class_id, None)
                if stats is None:
                    # fallback defaults (small)
                    stats = {
                        'w_mean': 0.05, 'w_std': 0.01, 'w_min': 0.01, 'w_max': 0.2,
                        'h_mean': 0.25, 'h_std': 0.05, 'h_min': 0.05, 'h_max': 0.6,
                        'x_mean': 0.5, 'x_std': 0.2, 'x_min': 0.0, 'x_max': 1.0,
                        'y_mean': 0.5, 'y_std': 0.2, 'y_min': 0.0, 'y_max': 1.0,
                    }

                # sample normalized width & height from normal (clip to observed min/max)
                w_norm = sample_normal_clipped(stats['w_mean'], stats['w_std'], stats['w_min'], stats['w_max'])
                h_norm = sample_normal_clipped(stats['h_mean'], stats['h_std'], stats['h_min'], stats['h_max'])

                # compute pixel target size
                target_w_px, target_h_px = compute_target_pixels(w_norm, h_norm, IMG_WIDTH, IMG_HEIGHT)

                # preserve aspect ratio of raw object: compute scale
                if raw_w == 0 or raw_h == 0:
                    continue
                scale_w = target_w_px / raw_w
                scale_h = target_h_px / raw_h
                scale = min(scale_w, scale_h)
                # if scale is zero or negative, skip
                if scale <= 0:
                    continue
                new_w = max(1, int(round(raw_w * scale)))
                new_h = max(1, int(round(raw_h * scale)))
                obj_resized = obj_pil.resize((new_w, new_h), resample=Image.LANCZOS)

                # attempt placements with sampling until success or max tries
                placed = False
                tries = 0
                last_box = None
                while not placed and tries < MAX_PLACEMENT_TRIES:
                    tries += 1

                    # sample centers:
                    if class_id == 2:  # dino: x ~ normal (clip), y uniform (top->bottom) but ensure inside
                        x_norm = sample_normal_clipped(stats.get('x_mean', 0.042493), stats.get('x_std', 0.006163),
                                                       stats.get('x_min', 0.0), stats.get('x_max', 1.0))
                        # y uniform but ensure within vertical bounds so object fully inside
                        half_h_norm = (new_h / IMG_HEIGHT) / 2.0
                        y_norm = sample_uniform_clipped(half_h_norm + 0.0, 1.0 - half_h_norm - 0.0)
                    else:
                        # for cactus and ptero use y_center normal from stats (clip to [min,max])
                        x_norm = sample_uniform_clipped(0.0, 1.0)
                        # but ensure x doesn't put object out of image when considering width
                        half_w_norm = (new_w / IMG_WIDTH) / 2.0
                        x_norm = clamp(x_norm, half_w_norm, 1.0 - half_w_norm)
                        y_norm = sample_normal_clipped(stats['y_mean'], stats['y_std'], stats['y_min'], stats['y_max'])

                    # convert to pixel center
                    x_px_center = x_norm * IMG_WIDTH
                    y_px_center = y_norm * IMG_HEIGHT
                    # compute top-left for pasting
                    top_left_x = int(round(x_px_center - new_w / 2.0))
                    top_left_y = int(round(y_px_center - new_h / 2.0))

                    # ensure inside bounds
                    top_left_x = int(clamp(top_left_x, 0, IMG_WIDTH - new_w))
                    top_left_y = int(clamp(top_left_y, 0, IMG_HEIGHT - new_h))
                    # recompute normalized centers after clamp (to keep labels consistent)
                    x_px_center = top_left_x + new_w / 2.0
                    y_px_center = top_left_y + new_h / 2.0
                    x_norm_final = x_px_center / IMG_WIDTH
                    y_norm_final = y_px_center / IMG_HEIGHT
                    w_norm_final = new_w / IMG_WIDTH
                    h_norm_final = new_h / IMG_HEIGHT

                    # prepare bounding box in pixel coords
                    new_box = (top_left_x, top_left_y, top_left_x + new_w, top_left_y + new_h)
                    last_box = new_box

                    # check overlap rules
                    if place_object_no_overlap(existing_boxes, new_box, ALLOW_OVERLAP_PROB):
                        placed = True
                        break
                    # else loop and try new sample

                # if not placed after tries, accept last attempted location anyway (to ensure object exists)
                if not placed and last_box is not None:
                    new_box = last_box
                    # recompute final normalized values from last_box
                    top_left_x, top_left_y, x2, y2 = new_box
                    new_w = x2 - top_left_x
                    new_h = y2 - top_left_y
                    x_px_center = top_left_x + new_w / 2.0
                    y_px_center = top_left_y + new_h / 2.0
                    x_norm_final = x_px_center / IMG_WIDTH
                    y_norm_final = y_px_center / IMG_HEIGHT
                    w_norm_final = new_w / IMG_WIDTH
                    h_norm_final = new_h / IMG_HEIGHT

                # paste object onto base using alpha
                base = paste_with_alpha(base, obj_resized, int(new_box[0]), int(new_box[1]))

                # record this box
                existing_boxes.append(new_box)

                # write label line with high precision
                label_lines.append(f"{class_id} " +
                                   f"{x_norm_final:.{LABEL_DECIMALS}f} " +
                                   f"{y_norm_final:.{LABEL_DECIMALS}f} " +
                                   f"{w_norm_final:.{LABEL_DECIMALS}f} " +
                                   f"{h_norm_final:.{LABEL_DECIMALS}f}\n")

        # after placing all objects, save image and label to the assigned split
        split = splits[i]
        out_name = f"synthetic_{i:06d}"
        out_image_path = os.path.join(FINAL_DATASET_DIR, split, "images", out_name + ".jpg")
        out_label_path = os.path.join(FINAL_DATASET_DIR, split, "labels", out_name + ".txt")
        # convert RGBA to RGB before saving as JPG (fill transparent with template)
        rgb = base.convert("RGB")
        rgb.save(out_image_path, quality=95)
        with open(out_label_path, "w", encoding="utf-8") as lf:
            lf.writelines(label_lines)

        if (i + 1) % 50 == 0 or i == num_samples - 1:
            print(f"[{i + 1}/{num_samples}] saved: {out_image_path}")


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python Dataset_Creator.py <num_samples> <train_ratio> <valid_ratio> <test_ratio>")
        print("Example: python Dataset_Creator.py 100 0.75 0.15 0.10")
        sys.exit(1)

    num_samples = int(sys.argv[1])
    train_ratio = float(sys.argv[2])
    valid_ratio = float(sys.argv[3])
    test_ratio = float(sys.argv[4])

    if abs(train_ratio + valid_ratio + test_ratio - 1.0) > 0.01:
        print("Error: Ratios must sum to approximately 1.0")
        sys.exit(1)

    # Compute exact counts
    train_count = int(num_samples * train_ratio)
    valid_count = int(num_samples * valid_ratio)
    test_count = num_samples - train_count - valid_count

    # Create and shuffle split assignments
    splits = ['train'] * train_count + ['valid'] * valid_count + ['test'] * test_count
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
    random.shuffle(splits)

    print(f"Dataset will be split: Train={train_count}, Valid={valid_count}, Test={test_count}")

    if RANDOM_SEED is not None:
        np.random.seed(RANDOM_SEED)
        random.seed(RANDOM_SEED)

    print("Loading existing label stats...")
    with open("assets/summary.pkl", "rb") as f:
        summary = pickle.load(f)
    for cid, s in summary.items():
        print(
            f"Class {cid}: count={s['count']}, w_mean={s['w_mean']:.6f}, w_std={s['w_std']:.6f}, h_mean={s['h_mean']:.6f}, h_std={s['h_std']:.6f}")
        if cid == 2:
            print(f"  x_mean={s['x_mean']:.6f}, x_std={s['x_std']:.6f}")

    print(f"Generating {num_samples} synthetic images and assigning to splits...")
    generate_synthetic(summary, num_samples=num_samples, splits=splits, seed=RANDOM_SEED)

    # Create data.yaml
    dataset_dir = Path(FINAL_DATASET_DIR)
    data = {
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 3,
        "names": ["cactus", "ptero", "trex"]
    }
    yaml_path = dataset_dir / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    print(f"Created {yaml_path}")

    print("Done. Outputs in:", FINAL_DATASET_DIR)

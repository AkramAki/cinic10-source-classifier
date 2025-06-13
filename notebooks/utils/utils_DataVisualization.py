import os
from tqdm import tqdm
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt


def check_images(df, data_dir):
    print("Checking if any images in the dataset are broken with img.verify()")
    broken = []
    for row in tqdm(df.itertuples(), total=len(df)):
        path = os.path.join(data_dir, row.full_path)
        try:
            with Image.open(path) as img:
                img.verify()
        except Exception:
            broken.append(path)

    if len(broken) == 0:
        print("No broken Images")
    else:
        print(f"Broken Images in the following file Paths")
        for filePath in broken:
            print(filePath)
    return broken


def show_images(df, data_dir, label, n=5, random_state=None, class_filter=None):
    """
    Show random images from CINIC-10 filtered by domain source and optionally by class.

    Parameters:
    - df: DataFrame with columns ['split', 'category', 'filename', 'source']
    - data_dir: Path to CINIC-10 root directory
    - label: 'CIFAR-10' or 'ImageNet'
    - n: Number of images to show
    - random_state: Optional int seed for reproducible sampling
    - class_filter: Optional string or list of class names (e.g., 'dog' or ['dog', 'cat'])
    """
    subset = df[df["source"] == label]

    # Apply class filter if provided
    if class_filter is not None:
        if isinstance(class_filter, str):
            class_filter = [class_filter]
        subset = subset[subset["category"].isin(class_filter)]

    # Sample and visualize
    subset = subset.sample(n=n, random_state=random_state)
    fig, axs = plt.subplots(1, n, figsize=(3 * n, 3))
    for ax, row in zip(axs, subset.itertuples()):
        img_path = os.path.join(data_dir, row.full_path)
        ax.imshow(Image.open(img_path))
        ax.set_title(row.category)
        ax.axis("off")
    plt.suptitle(
        f"{label} — {class_filter if class_filter else 'All Classes'}")
    plt.tight_layout()


def compute_rgb_stats(df, data_dir, n=1000, source=None, category=None, random_state=None):
    """
    Compute average RGB values for a random sample of images in the CINIC-10 dataset.

    Parameters:
    - df: DataFrame with image metadata
    - data_dir: Root path to CINIC-10 (e.g., 'data/CINIC-10')
    - n: Number of images to sample
    - source: Optional domain filter ('CIFAR-10' or 'ImageNet')
    - category: Optional class filter (string or list of strings)
    - random_state: Optional seed for reproducible sampling

    Returns:
    - mean RGB array (shape: [3,])
    """
    subset = df.copy()

    if source is not None:
        subset = subset[subset["source"] == source]

    if category is not None:
        if isinstance(category, str):
            category = [category]
        subset = subset[subset["category"].isin(category)]

    sample = subset.sample(n=min(n, len(subset)), random_state=random_state)
    means = []

    for row in sample.itertuples():
        path = os.path.join(data_dir, row.full_path)
        try:
            img = np.array(Image.open(path).convert("RGB")) / 255.0
            # Calculate mean pixel Value per color channel
            means.append(img.mean(axis=(0, 1)))
        except Exception as e:
            print(f"Skipping {path} \n Error with Image \n Exception: {e}")

    return np.array(means).mean(axis=0) if means else None


def compare_rgb_means_per_class(df, data_dir, n=1000):
    categories = sorted(df["category"].unique())
    print(f"{'Class':<12} {'CIFAR-10 (R,G,B)':<30} {'ImageNet (R,G,B)':<30} {'Difference (R,G,B)':<30}")
    print("-" * 100)

    for cat in categories:
        cifar_rgb = compute_rgb_stats(
            df, data_dir, n=n, source="CIFAR-10", category=cat)
        imagenet_rgb = compute_rgb_stats(
            df, data_dir, n=n, source="ImageNet", category=cat)

        if cifar_rgb is not None and imagenet_rgb is not None:
            diff = imagenet_rgb - cifar_rgb
            print(f"{cat:<12} "
                  f"{tuple(float(x) for x in np.round(cifar_rgb, 3))!s:<30} "
                  f"{tuple(float(x) for x in np.round(imagenet_rgb, 3))!s:<30} "
                  f"{tuple(float(x) for x in np.round(diff, 3))!s:<30}")

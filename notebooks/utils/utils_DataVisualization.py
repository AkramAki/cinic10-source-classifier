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
    - (mean RGB array (shape: [3,]), meanVariance RGB array (shape:[3,]))
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
    picture_variance = []

    for row in sample.itertuples():
        path = os.path.join(data_dir, row.full_path)
        try:
            img = np.array(Image.open(path).convert("RGB")) / 255.0
            # Calculate mean pixel Value per color channel
            means.append(img.mean(axis=(0, 1)))
            picture_variance.append(img.var(axis=(0, 1)))
        except Exception as e:
            print(f"Skipping {path} \n Error with Image \n Exception: {e}")
    # https://www.emathzone.com/tutorials/basic-statistics/combined-variance.html      note: n1 = n2 due to pictures always having the same size of 32*32
    N = 32*32
    means = np.array(means)
    combined_mean = means.mean(axis=0)
    combined_variance = (N * np.sum(np.array(picture_variance), axis=0) + N * np.sum((means - combined_mean)**2, axis=0)) / (len(picture_variance) * N)

    return (combined_mean, combined_variance)


def compare_rgb_means_per_class(df, data_dir, n=1000):
    categories = sorted(df["category"].unique())
    print(f"{'Class':<12} {'CIFAR-10 (R,G,B)':<30} {'ImageNet (R,G,B)':<30} {'Difference (R,G,B)':<30}")
    print("-" * 100)

    for cat in categories:
        cifar_rgb_mean, cifar_rgb_meanVariance = compute_rgb_stats(
            df, data_dir, n=n, source="CIFAR-10", category=cat)
        imagenet_rgb_mean, imagenet_rgb_meanVariance = compute_rgb_stats(
            df, data_dir, n=n, source="ImageNet", category=cat)

        if cifar_rgb_mean is not None and imagenet_rgb_mean is not None:
            diff = imagenet_rgb_mean - cifar_rgb_mean
            print(f"{cat:<12} "
                  f"{tuple(float(x) for x in np.round(cifar_rgb_mean, 3))!s:<30} "
                  f"{tuple(float(x) for x in np.round(imagenet_rgb_mean, 3))!s:<30} "
                  f"{tuple(float(x) for x in np.round(diff, 3))!s:<30}")
            

def compare_rgb_meanVariances_per_class(df, data_dir, n=1000):
    categories = sorted(df["category"].unique())
    print(f"{'Class':<12} {'CIFAR-10 (R,G,B)':<30} {'ImageNet (R,G,B)':<30} {'Difference (R,G,B)':<30}")
    print("-" * 100)

    for cat in categories:
        cifar_rgb_mean, cifar_rgb_meanVariance = compute_rgb_stats(
            df, data_dir, n=n, source="CIFAR-10", category=cat)
        imagenet_rgb_mean, imagenet_rgb_meanVariance = compute_rgb_stats(
            df, data_dir, n=n, source="ImageNet", category=cat)
            
        if cifar_rgb_meanVariance is not None and imagenet_rgb_meanVariance is not None:
            diff = imagenet_rgb_meanVariance - cifar_rgb_meanVariance
            print(f"{cat:<12} "
                  f"{tuple(float(x) for x in np.round(cifar_rgb_meanVariance, 3))!s:<30} "
                  f"{tuple(float(x) for x in np.round(imagenet_rgb_meanVariance, 3))!s:<30} "
                  f"{tuple(float(x) for x in np.round(diff, 3))!s:<30}")

def compute_pixelwise_stats(df, data_dir, n=1000, source=None, category=None, random_state=None, image_size=(32, 32)):
    """
    Compute per-pixel RGB variance across a sample of images for a given class/source.

    Returns:
    - (H, W, 3) numpy array of variances
    """
    subset = df.copy()

    if source is not None:
        subset = subset[subset["source"] == source]

    if category is not None:
        if isinstance(category, str):
            category = [category]
        subset = subset[subset["category"].isin(category)]

    sample = subset.sample(n=min(n, len(subset)), random_state=random_state)
    images = []

    for row in tqdm(sample.itertuples(), desc=f"Computing variance {category} | {source}"):
        path = os.path.join(data_dir, row.full_path)
        try:
            img = np.array(Image.open(path).convert("RGB").resize(image_size)) / 255.0
            images.append(img)
        except Exception as e:
            print(f"Skipping {path}\nException: {e}")

    if not images:
        return None

    images = np.stack(images, axis=0)  # shape: (N, H, W, C)
    return (np.mean(images, axis=0), np.var(images, axis=0))  # shape: (2, H, W, C)


def show_number_image(number_img, title="Pixelwise Variance", channel="R"):
    channel_idx = {"R": 0, "G": 1, "B": 2}[channel.upper()]
    data = number_img[:, :, channel_idx]

    fig, ax = plt.subplots(figsize=(12, 12))
    im = ax.imshow(data, cmap="viridis")

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = f"{data[i, j]:.2f}"
            ax.text(j, i, val, ha="center", va="center", color="white", fontsize=6)

    ax.set_title(f"{title} (Channel: {channel})")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()


## 📘 Project: *Domain Origin Detection in CINIC-10*

**Can a neural network distinguish whether an image in CINIC-10 comes from CIFAR-10 or ImageNet — even when class labels are the same?**

This project explores a meta-learning task using the CINIC-10 dataset. We train a binary classifier to detect the *origin domain* of an image (CIFAR vs. ImageNet), despite both domains sharing the same 10 class labels (e.g., airplane, dog, ship).

### 🔍 Goals

* Create a binary domain label (`CIFAR` or `ImageNet`) for each image.
* Train a simple CNN or MLP to predict the source domain.
* Compare with baseline methods like random guessing or pixel statistics.
* Visualize image embeddings using t-SNE to explore learned patterns.

### 📊 Dataset

* **CINIC-10**: A 270,000-image dataset mixing CIFAR-10 and downsampled ImageNet images.
* All images are 32×32 pixels and belong to one of 10 shared classes.

### 🧠 Methods

* Basic CNN or shallow MLP as classifiers.
* Use of pretrained features (e.g. ResNet18) for comparison.
* Optional: ROC AUC, accuracy, t-SNE visualization.

### 📁 Structure

```bash
data/         # CINIC-10 images and labels
notebooks/    # Jupyter notebooks for exploration and training
models/       # Trained model files
results/      # Evaluation metrics and plots
```

### ✅ Before Using This Repository

Before running the code, make sure to **download the CINIC-10 dataset**:

1. Navigate to the `data/` directory.
2. Follow the instructions provided inside the loading_the_dataset.md to download and extract the dataset.
3. Once the dataset is in place, you can:

   * Run the notebooks in `notebooks/`
   * Load trained models from `models/`
   * View visualizations and metrics in `results/`

Everything should work out of the box after the dataset is set up correctly.

### 👥 Team

* Akram Aki
* Muhammad Hassan
* ML Seminar, TU Dortmund, Germany, Semester 6

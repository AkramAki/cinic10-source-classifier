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
Note that we did not use results but used a build directory inside the notebooks directory to store some images.

### ✅ Before You Start

Before running the code, make sure to **download the CINIC-10 dataset**, set up the environment and run the python script to generate Labels for the Dataset.

1. Navigate to the `data/` directory.

2. Follow the instructions in `loading_the_dataset.md` to download and extract the dataset.

3. Install the required Python packages using the provided environment file.
   We recommend using **[Mamba](https://github.com/conda-forge/miniforge)** (a faster alternative to conda):

   ```bash
   mamba env create -f environment_2025.yml
   ```

4. Activate the environment before working on the project:

   ```bash
   mamba activate ml
   ```

5. Navigate to `generate_domain_labels.py` and follow the instructions provided in the comments in lines 6-8.

6. Once the environment is active and the dataset is ready, you can:

   * Run the notebooks in `notebooks/`
   * Load trained models from `models/`
   * View visualizations and metrics in `results/`

Everything should work out of the box if the steps above were followed correctly.


---

### GPU Support (Optional)
"You are welcome to add and correct instructions for GPU Support"

To use GPU acceleration with TensorFlow, a compatible **NVIDIA GPU** and driver must be installed.

#### How to Check for GPU Availability

**1. Run:**

```bash
nvidia-smi
```

If you see:

```
zsh: command not found: nvidia-smi
```

your system does **not** have the NVIDIA driver installed.

**2. Check for NVIDIA hardware:**

```bash
lspci | grep -i nvidia
```

If you see output like:

```
01:00.0 VGA compatible controller: NVIDIA Corporation ...
```

you **do** have an NVIDIA GPU and just need the driver.

---

#### 🐧 NVIDIA Driver Installation on Fedora 42 

I did not get it to work but maybe someone can add proper instructions

---

📎 Official source:
[NVIDIA CUDA Downloads for Fedora](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Fedora&target_version=41&target_type=rpm_network)

---


### 👥 Team

* Akram Aki, Bachelor Semester 6
* Muhammad Hassan, Master 
* ML Seminar, TU Dortmund, Germany

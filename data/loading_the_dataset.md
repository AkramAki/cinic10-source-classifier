# 📦 Loading the CINIC-10 Dataset

To use this project, you first need to download and set up the CINIC-10 dataset.

---

## 🔗 Step-by-Step Instructions

1. **Download the dataset**

   Go to the official CINIC-10 page:  
   👉 https://datashare.ed.ac.uk/handle/10283/3192

   Click on **“Download a file”** to download the full dataset archive.

2. **Unzip the downloaded file**

   You will get two files:
   - `LICENSE.txt`
   - `CINIC-10.tar.gz`

3. **Unpack the dataset archive**

   Extract the `CINIC-10.tar.gz` file. This will create a `CINIC-10/` directory.

4. **Copy the extracted files into your project**

   Your `data/` folder in this repository should now contain:
   ```
   data/
   ├── loading_the_dataset.md
   ├── LICENSE.txt
   └── CINIC-10/
       ├── train/
       ├── valid/
       ├── test/
       ├── imagenet-contributors.csv
       ├── README.md
       └── synsets-to-cifar-10-classes.txt
   ```

5. **Check the dataset structure**

   Each of `train/`, `valid/`, and `test/` should contain 10 folders:
   ```
   airplane/
   automobile/
   bird/
   cat/
   deer/
   dog/
   frog/
   horse/
   ship/
   truck/
   ```
   Each class folder contains `.png` image files.

---

Once everything is set up, you're ready to run the notebooks and experiments!
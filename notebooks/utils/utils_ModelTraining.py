import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from PIL import Image

import os

from tqdm import tqdm

root_path = ".."
data_path = os.path.join(root_path, "data")
CINIC10_path = os.path.join(data_path, "CINIC-10")

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['label'])  # CIFAR-10 = 0, ImageNet = 1
    return df

def load_data_all_splits():
    df_train = load_data(os.path.join(data_path, "prepared/train.csv"))
    df_valid = load_data(os.path.join(data_path, "prepared/valid.csv"))
    df_test  = load_data(os.path.join(data_path, "prepared/test.csv"))
    return (df_train, df_valid, df_test)

def load_image(path):
    path = os.path.join(CINIC10_path, path)
    img = Image.open(path).convert("RGB").resize((32, 32))
    return np.array(img) / 255.0  # normalize to [0, 1]

def prepare_dataset(df):
    images = []
    labels = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading images"):
        img = load_image(row["full_path"])
        images.append(img)
        labels.append(row["label"])
    
    return np.stack(images), np.array(labels)
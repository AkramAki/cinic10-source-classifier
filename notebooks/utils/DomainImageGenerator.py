import numpy as np
import pandas as pd
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder

import os

class DomainImageGenerator(Sequence):
    def __init__(self, csv_path_from_data, batch_size=64, img_size=(32, 32), shuffle=True, n_jobs=1, use_multiprocessing=False, max_queue_size=10, **kwargs):
        super().__init__(**kwargs)   
        self.root_path = ".."
        self.data_path = os.path.join(self.root_path, "data")
        self.CINIC10_path = os.path.join(self.data_path, "CINIC-10")

        self.df = pd.read_csv(os.path.join(self.data_path, csv_path_from_data))
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.encoder = LabelEncoder()
        self.df['label'] = self.encoder.fit_transform(self.df['label'])
        self.indexes = np.arange(len(self.df))
        self.on_epoch_end()
        # Information on  https://www.tensorflow.org/api_docs/python/tf/keras/utils/PyDataset
        self.workers = n_jobs 
        self.use_multiprocessing = use_multiprocessing
        self.max_queue_size = max_queue_size


    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, index):
        batch_idx = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_df = self.df.iloc[batch_idx]
        
        images = []
        labels = []

        for _, row in batch_df.iterrows():
            path = os.path.join(self.CINIC10_path, row["full_path"])
            img = load_img(path, target_size=self.img_size)
            img = img_to_array(img) / 255.0  # normalize
            images.append(img)
            labels.append(row["label"])
        return np.array(images), np.array(labels)

    def on_epoch_end(self):
        if self.shuffle:
            self.shuffle_indexes()

    def shuffle_indexes(self):
        np.random.shuffle(self.indexes)

    def getAllLabels(self):
        return self.return_Label_by_Index(self.indexes)

    def return_Label_by_Index(self, indexes):
        return self.df['label'].iloc[indexes].to_numpy()

    def return_Img_by_Index(self, indexes):
        index_df = self.df.iloc[indexes]
        images = []

        for _, row in index_df.iterrows():
            path = os.path.join(self.CINIC10_path, row["full_path"])
            img = load_img(path, target_size=self.img_size)
            img = img_to_array(img) / 255.0  # normalize
            images.append(img)
            
        return np.array(images)

    def getAllCategories(self):
        return self.return_Category_by_Index(self.indexes)

    def return_Category_by_Index(self, indexes):
        return self.df['category'].iloc[indexes].to_numpy()

    def return_Indexes(self):
        return self.indexes

    def return_Dataframe(self):
        return self.df
        
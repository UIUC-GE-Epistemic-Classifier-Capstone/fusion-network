import glob
import os

import cv2
import numpy as np
import torch
import csv
import pandas as pd
from torch.utils.data import Dataset


def get_dataset():
    return dataset_dict["Fusion_data"]()

class BaseDataset(Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def __len__(self):
        return self.n_img

    def __getitem__(self, index):
        label = self.label[index]
        feature = self.feature[index]
        return index, feature, label


class Fusion_data(BaseDataset):
    def __init__(self):
        super(Fusion_data, self).__init__()
        self.read_feature_csv()
        self.n_img = len(self.feature)
        self.read_label_csv()

    def read_feature_csv(self):
        features = []
        # file_path = './daytime_feature.csv' #'./feature_own.csv'
        file_path = './feature.csv' #'./feature_own.csv'

        with open(file_path, 'r') as csv_file:
            reader = csv.reader(csv_file)
            print("reader:", reader)
            for row in reader:
                feature = row  # Assuming the label is in the first column
                features.append([float(i) for i in feature])
        self.feature = features[1:]
        # print("self.feature:", self.feature)
        self.feature = torch.from_numpy(np.array(self.feature))

    def read_label_csv(self):
        labels = []
        # file_path = './daytime_label.csv'
        file_path = './label.csv'

        # file_path = './label.csv'
        with open(file_path, 'r') as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                label = row[0]  # Assuming the label is in the first column
                labels.append(int(label))
        self.label = labels
        self.label = torch.from_numpy(np.array(self.label))


dataset_dict = {
    "Fusion_data": Fusion_data,
}
"""
Helper functions for this MP, students can ignore this file
"""

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pickle
import random
import torch
from torch.utils.data import Dataset, DataLoader


def Load_dataset(filename):
    def unpickle(file):
        with open(file, "rb") as fo:
            dict = pickle.load(fo, encoding="bytes")
        return dict

    A = unpickle(filename)
    X = A[b"data"]
    Y = A[b"labels"].astype(np.int64)
    test_size = int(0.25 * len(X))  # set aside 25% for testing
    X_test = X[:test_size]
    Y_test = Y[:test_size]
    X = X[test_size:]
    Y = Y[test_size:]
    return X, Y, X_test, Y_test


def Preprocess(train_set, test_set):
    train_set = torch.tensor(train_set, dtype=torch.float32)
    test_set = torch.tensor(test_set, dtype=torch.float32)
    mu = train_set.mean(dim=0, keepdim=True)
    std = train_set.std(dim=0, keepdim=True)
    train_set = (train_set - mu) / std
    test_set = (test_set - mu) / std
    return train_set, test_set


def Get_DataLoaders(train_set, train_labels, test_set, test_labels, batch_size):
    train_dataset = MP_Dataset(train_set, train_labels)
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_dataset = MP_Dataset(test_set, test_labels)
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    return train_loader, test_loader


class MP_Dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, X, y):
        """
        Args:
            X [np.array]: features vector
            y [np.array]: labels vector
        """
        self.data = X
        self.labels = y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        features = self.data[idx, :]
        label = self.labels[idx]
        return features, label


def init_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def compute_accuracies(pred_labels, dev_labels):
    assert (
        pred_labels.dtype == int or pred_labels.dtype == np.int64
    ), "Your predicted labels have type {}, but they should have type np.int (consider using .astype(int) on your output)".format(
        pred_labels.dtype
    )
    if len(pred_labels) != len(dev_labels):
        print(
            "Lengths of predicted labels don't match length of actual labels",
            len(pred_labels),
            len(dev_labels),
        )
        return 0.0, None
    accuracy = np.mean(pred_labels == dev_labels)
    conf_m = np.zeros((len(np.unique(dev_labels)), len(np.unique(dev_labels))))
    for i, j in zip(dev_labels, pred_labels):
        conf_m[i, j] += 1
    return accuracy, conf_m


names = {0: "ship", 1: "automobile", 2: "dog", 3: "frog", 4: "horse"}


def show_train_image(train_set, train_labels, index):
    img_c1 = Image.fromarray(train_set[index][:961].reshape(31, 31))
    img_c2 = Image.fromarray(train_set[index][961:1922].reshape(31, 31))
    img_c3 = Image.fromarray(train_set[index][1922:].reshape(31, 31))
    img_rgb = np.zeros((31, 31, 3), "uint8")
    img_rgb[..., 0] = img_c1
    img_rgb[..., 1] = img_c2
    img_rgb[..., 2] = img_c3
    fig = plt.figure()
    plt.axis("off")
    plt.imshow(img_rgb)
    title = (
        "Train["
        + str(index)
        + "]  --  "
        + names[train_labels[index].item()]
        + " -- label "
        + str(train_labels[index].item())
    )
    plt.title(title)


def show_test_image(test_set, test_labels, index):
    img_c1 = Image.fromarray(test_set[index][:961].reshape(31, 31))
    img_c2 = Image.fromarray(test_set[index][961:1922].reshape(31, 31))
    img_c3 = Image.fromarray(test_set[index][1922:].reshape(31, 31))
    img_rgb = np.zeros((31, 31, 3), "uint8")
    img_rgb[..., 0] = img_c1
    img_rgb[..., 1] = img_c2
    img_rgb[..., 2] = img_c3
    fig = plt.figure()
    plt.axis("off")
    plt.imshow(img_rgb)
    title = (
        "Test["
        + str(index)
        + "]  --  "
        + names[test_labels[index].item()]
        + " -- label "
        + str(test_labels[index].item())
    )
    plt.imshow(img_rgb)
    plt.title(title)

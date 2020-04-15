import torch
import random
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import cv2
from torchvision import transforms
import torchvision.datasets as dset
from torch.utils.data import Dataset


def get_fine_tune_image_path(root, image_count):
    orig_path = root + 'orig/'
    df_path = root + 'df/'
    fs_path = root + 'fs/'
    orig_images = [os.path.join(orig_path, f) for f in listdir(orig_path) if isfile(join(orig_path, f))]
    df_images = [os.path.join(df_path, f) for f in listdir(df_path) if isfile(join(df_path, f))]
    fs_images = [os.path.join(fs_path, f) for f in listdir(fs_path) if isfile(join(fs_path, f))]
    pairs = list(zip(orig_images, df_images, fs_images))  # make pairs out of the two lists
    pairs = pairs[:image_count]  # random.sample(pairs, image_count)  # pick `image_count` random pairs
    orig_list, df_list, fs_list = zip(*pairs)  # separate the pairs
    image_list = list(orig_list) + list(df_list) + list(fs_list)
    return image_list, orig_list, df_list, fs_list


class SiameseDataset(Dataset):

    def __init__(self, dirpath, mode,  is_fine_tune=False, image_count=None, transform=None):
        self.dirpath = dirpath
        self.mode = mode
        if is_fine_tune:
            self.image_path = get_fine_tune_image_path(root=dirpath + '/', image_count=image_count)
            input_images = [(t, 0) if "orig" in t else (t, 1) for t in self.image_path]
        else:
            self.image_path = dset.ImageFolder(root=dirpath + '/').imgs
            input_images = [(t[0], 0) if "orig" in t[0] else (t[0], 1) for t in self.image_path]
        self.transform = transform
        self.to_tensor = transforms.ToTensor()
        # Training input images path

        self.labels = np.array([image_data[1] for image_data in input_images])
        self.data = np.array([image_data[0] for image_data in input_images])
        self.labels_set = set(self.labels)
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                 for label in self.labels_set}

        val_list = ['val', 'test']
        if mode in val_list:
            random_state = np.random.RandomState(0)
            # Positive Pair : index1, index2, 1
            positive_pairs = [[i, random_state.choice(self.label_to_indices[self.labels[i].item()]), 1]
                              for i in range(0, len(self.data), 2)]
            # Negative Pair : index1, index2, 0
            negative_pairs = [[i, random_state.choice(self.label_to_indices[np.random.choice(
                list(self.labels_set - set([self.labels[i].item()])))]), 0]
                              for i in range(1, len(self.data), 2)]
            self.test_pairs = positive_pairs + negative_pairs

    def __getitem__(self, index):

        if self.mode == 'train':
            target = np.random.randint(0, 2)
            img1, label1 = self.data[index], self.labels[index].item()
            if target == 1:  # Pick the same class
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(self.label_to_indices[label1])
            else:  # Pick the different class
                siamese_label = np.random.choice(list(self.labels_set - set([label1])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            img2, label2 = self.data[siamese_index], self.labels[siamese_index].item()

        else:
            img1, label1 = self.data[self.test_pairs[index][0]], self.labels[self.test_pairs[index][0]]
            img2, label2 = self.data[self.test_pairs[index][1]], self.labels[self.test_pairs[index][1]]
            target = self.test_pairs[index][2]

        # img0_tuple = random.choice(input_images)
        # # we need to make sure approx 50% of images are in the same class
        # should_get_same_class = random.randint(0, 1)
        # if should_get_same_class:
        #     while True:
        #         # keep looping till the same class image is found
        #         img1_tuple = random.choice(input_images)
        #         if img0_tuple[1] == img1_tuple[1]:
        #             break
        # else:
        #     while True:
        #         # keep looping till a different class image is found
        #         img1_tuple = random.choice(input_images)
        #         if img0_tuple[1] != img1_tuple[1]:
        #             break

        img1 = cv2.imread(img1)
        img2 = cv2.imread(img2)

        img1 = np.array(img1, dtype='uint8')
        img2 = np.array(img2, dtype='uint8')
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            img1_as_tensor = self.transform(img1)
            img2_as_tensor = self.transform(img2)
        else:
            img1_as_tensor = self.to_tensor(img1)
            img2_as_tensor = self.to_tensor(img2)

        return img1_as_tensor, label1, img2_as_tensor, label2, torch.from_numpy(
            np.array([int(label1 != label2)], dtype=np.float32))

    def __len__(self):
        return len(self.image_path)


class SiameseL2NormDataset(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

    def __init__(self, dirpath, mode, image_count=None, transform=None):
        self.dirpath = dirpath
        self.mode = mode
        _, _, self.df_list, self.fs_list = get_fine_tune_image_path(root=dirpath + '/', image_count=image_count)
        self.transform = transform
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, index):

        img_df = self.df_list[index]
        img_fs = self.fs_list[index]

        img_df = cv2.imread(img_df)
        img_fs = cv2.imread(img_fs)

        img_df = np.array(img_df, dtype='uint8')
        img_fs = np.array(img_fs, dtype='uint8')

        img_df = cv2.cvtColor(img_df, cv2.COLOR_BGR2RGB)
        img_fs = cv2.cvtColor(img_fs, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            img_df_as_tensor = self.transform(img_df)
            img_fs_as_tensor = self.transform(img_fs)
        else:
            img_df_as_tensor = self.to_tensor(img_df)
            img_fs_as_tensor = self.to_tensor(img_fs)

        return img_df_as_tensor, img_fs_as_tensor

    def __len__(self):
        return len(self.df_list)


class SiameseTripletDataset(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

    def __init__(self, dirpath, mode,  is_fine_tune=False, image_count=None, transform=None):
        self.dirpath = dirpath
        self.mode = mode
        if is_fine_tune:
            self.image_path = get_fine_tune_image_path(root=dirpath + '/', image_count=image_count)
            input_images = [(t, 0) if "orig" in t else (t, 1) for t in self.image_path]
        else:
            self.image_path = dset.ImageFolder(root=dirpath + '/').imgs
            input_images = [(t[0], 0) if "orig" in t[0] else (t[0], 1) for t in self.image_path]
        self.transform = transform
        self.to_tensor = transforms.ToTensor()
        # Training input images path

        self.labels = np.array([image_data[1] for image_data in input_images])
        self.data = np.array([image_data[0] for image_data in input_images])
        self.labels_set = set(self.labels)
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                 for label in self.labels_set}

        val_list = ['val', 'test']
        if mode in val_list:
            random_state = np.random.RandomState(0)

            triplets = [[i, random_state.choice(self.label_to_indices[self.labels[i].item()]),
                         random_state.choice(self.label_to_indices[np.random.choice(list(self.labels_set - set([self.labels[i].item()])))])
                         ] for i in range(len(self.data))]
            self.test_triplets = triplets

    def __getitem__(self, index):
        if self.mode == 'train':
            img1, label1 = self.data[index], self.labels[index].item()
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img2 = self.data[positive_index]
            label2 = self.labels[positive_index].item()
            img3 = self.data[negative_index]
            label3 = self.labels[negative_index].item()
        else:
            img1 = self.data[self.test_triplets[index][0]]
            label1 = self.labels[self.test_triplets[index][0]].item()
            img2 = self.data[self.test_triplets[index][1]]
            label2 = self.labels[self.test_triplets[index][1]].item()
            img3 = self.data[self.test_triplets[index][2]]
            label3 = self.labels[self.test_triplets[index][2]].item()

        img1 = cv2.imread(img1)
        img2 = cv2.imread(img2)
        img3 = cv2.imread(img3)

        img1 = np.array(img1, dtype='uint8')
        img2 = np.array(img2, dtype='uint8')
        img3 = np.array(img3, dtype='uint8')
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            img1_as_tensor = self.transform(img1)
            img2_as_tensor = self.transform(img2)
            img3_as_tensor = self.transform(img3)
        else:
            img1_as_tensor = self.to_tensor(img1)
            img2_as_tensor = self.to_tensor(img2)
            img3_as_tensor = self.to_tensor(img3)

        return img1_as_tensor, label1, img2_as_tensor, label2, img3_as_tensor, label3

    def __len__(self):
        return len(self.image_path)

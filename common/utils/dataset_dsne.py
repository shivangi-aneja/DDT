"""
    dataset class for all the datasets
"""
import PIL.ImageOps
from torch.utils.data import Dataset
import torchvision.datasets as dset
from torchvision import transforms
import numpy as np
import cv2
import os
from os import listdir
from os.path import isfile, join
import random

DATASETS = {"ff"}


def get_available_datasets():
    """
    gets all the datasets
    :return:
    """
    return sorted(DATASETS)


def make_dataset(name, base_path, img_cnt_src, img_cnt_target, mode, transform=None):
    """
    it returns dataset according to its name
    :param name: dataset name
    :param base_path: base path
    :return: dataset
    """
    name = name.strip().lower()
    if not name in DATASETS:
        raise ValueError("invalid dataset: '{0}'".format(name))

    return DsneDataset(name=name, img_cnt_src=img_cnt_src, img_cnt_target=img_cnt_target, base_path=base_path, mode=mode, transform=transform)


def get_source_image_path(root, image_count=1):
    real_path = root + '/orig/'
    fake_path = root + '/f2f/'
    real_images = [os.path.join(real_path, f) for f in listdir(real_path) if isfile(join(real_path, f))]
    fake_images = [os.path.join(fake_path, f) for f in listdir(fake_path) if isfile(join(fake_path, f))]
    pairs = list(zip(real_images, fake_images))  # make pairs out of the two lists
    pairs = random.sample(pairs, image_count) #pairs[:image_count] #  # pick `image_count` random pairs
    real_list, fake_list = zip(*pairs)  # separate the pairs
    source_image_list = list(real_list) + list(fake_list)
    # print(ft_image_list)
    return source_image_list


def get_target_image_path(root, image_count=1):
    real_path = root + '/orig/'
    fake_path = root + '/fs/'
    real_images = [os.path.join(real_path, f) for f in listdir(real_path) if isfile(join(real_path, f))]
    fake_images = [os.path.join(fake_path, f) for f in listdir(fake_path) if isfile(join(fake_path, f))]
    pairs = list(zip(real_images, fake_images))  # make pairs out of the two lists
    pairs = random.sample(pairs, image_count) # pairs[:image_count] #  # pick `image_count` random pairs
    real_list, fake_list = zip(*pairs)  # separate the pairs
    target_image_list = list(real_list) + list(fake_list)
    # print(ft_image_list)
    return target_image_list


class DsneDataset(Dataset):
    """
    DsneDataset dataset
    """

    def __init__(self, name, base_path, img_cnt_src, img_cnt_target, mode, transform=None, should_invert=False):
        super(Dataset, self).__init__()
        self.mode = mode
        self.img_cnt_src = img_cnt_src
        self.base_path = base_path
        self.source_path = get_source_image_path(root=self.base_path + '/source_' + mode, image_count=img_cnt_src)
        self.target_path = get_target_image_path(root=self.base_path + '/target_' + mode, image_count=img_cnt_target)
        self.transform = transform
        self.should_invert = should_invert
        self.to_tensor = transforms.ToTensor()
        #input_images = self.source_path + self.target_path
        # Assign label to class
        # 0 for original, 1 for fake
        self.source_path = [(t, 0) if "orig" in t else (t, 1) for t in self.source_path]
        self.target_path = [(t, 0) if "orig" in t else (t, 1) for t in self.target_path]
        self.image_pairs = self._create_pairs()

    def _create_pairs(self):
        """
        Create pairs for array
        :return:
        """
        pos_pairs, neg_pairs = [], []
        for ids, ys in self.source_path:
            for idt, yt in self.target_path:
                if ys == yt:
                    pos_pairs.append([ids, ys, idt, yt, 1])
                else:
                    neg_pairs.append([ids, ys, idt, yt, 0])

        # if self.ratio > 0:
        #     random.shuffle(neg_pairs)
        #     pairs = pos_pairs + neg_pairs[: self.ratio * len(pos_pairs)]
        # else:
        pairs = pos_pairs + neg_pairs
        if self.mode == 'train':
            # pairs = random.sample(pos_pairs, int(self.img_cnt_src)) + random.sample(neg_pairs, int(self.img_cnt_src))
            random.shuffle(pairs)
        return pairs

    def __getitem__(self, index):
        # Training input images path
        input_img_src = cv2.imread(self.image_pairs[index][0])
        input_img_target = cv2.imread(self.image_pairs[index][2])

        input_img_src = np.array(input_img_src, dtype='uint8')
        input_img_src = cv2.cvtColor(input_img_src, cv2.COLOR_BGR2RGB)
        input_img_target = np.array(input_img_target, dtype='uint8')
        input_img_target = cv2.cvtColor(input_img_target, cv2.COLOR_BGR2RGB)

        if self.should_invert:
            input_img_src = PIL.ImageOps.invert(input_img_src)
            input_img_target = PIL.ImageOps.invert(input_img_target)

        if self.transform is not None:
            input_img_as_tensor_src = self.transform(input_img_src)
            input_img_as_tensor_target = self.transform(input_img_target)
        else:
            input_img_as_tensor_src = self.to_tensor(input_img_src)
            input_img_as_tensor_target = self.to_tensor(input_img_target)
        return input_img_as_tensor_src, self.image_pairs[index][1], input_img_as_tensor_target, self.image_pairs[index][3], self.image_pairs[index][4]

    def __len__(self):
        return len(self.image_pairs)
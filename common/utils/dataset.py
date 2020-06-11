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
import imutils
from torchvision.utils import save_image
from imutils import face_utils
from imgaug import augmenters as iaa
import dlib
import torch
from os import listdir
from os.path import isfile, join
import random
import warnings

DATASETS = {"ff", "google", "dessa", "celeb", "aif"}
CLASS_MAPPING = {0: "orig", 1: "fake" }


def get_available_datasets():
    """
    gets all the datasets
    :return:
    """
    return sorted(DATASETS)


def make_dataset(name, num_classes, base_path=None, base_path_ff=None, transform=None, image_count=None, spatial_aug=False, mode='face'):
    """
    it returns dataset according to its name
    :param name: dataset name
    :param base_path: base path
    :return: dataset
    """
    name = name.strip().lower()
    if not name in DATASETS:
        raise ValueError("invalid dataset: '{0}'".format(name))
    elif mode == 'face':
        return FaceDataset(image_count=image_count, base_path=base_path, spatial_aug=spatial_aug, transform=transform,
                           num_classes=num_classes)
    elif mode == 'face_finetune':
        return FaceFinetuneDataset(image_count=image_count, base_path=base_path, spatial_aug=spatial_aug, base_path_ff=base_path_ff,
                                   transform=transform, num_classes=num_classes)


def get_train_image_path(root, num_classes, image_count='all'):
    real_path = root + 'orig/'
    fake_path = root + 'fake/'
    
    real_images = [os.path.join(real_path, f) for f in listdir(real_path) if isfile(join(real_path, f))]
    fake_images = [os.path.join(fake_path, f) for f in listdir(fake_path) if isfile(join(fake_path, f))]
    
    source_image_list = None

    if num_classes == 2:
        real_list = list(real_images)
        fake_list = list(fake_images)
        if image_count != 'all':
            real_list = random.sample(real_list, image_count)
            fake_list = random.sample(fake_list, image_count)
        source_image_list = list(real_list) + list(fake_list)

    else:
        print("Invalid Classes")
    return source_image_list


def get_finetune_image_path(root, image_count=1):
    real_path = root + 'orig/'
    fake_path = root + 'fake/'
    real_images = [os.path.join(real_path, f) for f in listdir(real_path) if isfile(join(real_path, f))]
    fake_images = [os.path.join(fake_path, f) for f in listdir(fake_path) if isfile(join(fake_path, f))]
    pairs = list(zip(real_images, fake_images))  # make pairs out of the two lists
    pairs = random.sample(pairs, image_count)  # pick `image_count` random pairs
    real_list, fake_list = zip(*pairs)  # separate the pairs
    ft_image_list = list(real_list) + list(fake_list)
    return ft_image_list



class FaceDataset(Dataset):
    """
    FaceDataset dataset
    """

    def __init__(self, base_path, spatial_aug, image_count, transform=None, should_invert=False, num_classes=2):
        super(Dataset, self).__init__()
        self.base_path = base_path
        self.spatial_aug = spatial_aug
        self.image_path = get_train_image_path(root=self.base_path + '/', image_count=image_count, num_classes=num_classes)
        self.transform = transform
        self.should_invert = should_invert
        self.to_tensor = transforms.ToTensor()
        self.labels = [0 if "orig" in t else 1 for t in self.image_path]

    def __getitem__(self, index):
        # Training input images path
        input_images = self.image_path
        # Assign label to class
        try:
            input_images = [(t, 0) if "orig" in t else (t, 1) for t in input_images]
            # random_file1, random_file2, random_file3 = None, None, None
            input_img = cv2.imread(input_images[index][0])
            input_img = np.array(input_img, dtype='uint8')
            input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

            # Proposed Spatial Aug
            rand_int1 = torch.rand(1)
            if self.spatial_aug and "train" in input_images[index][0] and rand_int1[0] > 0.5:
                dir_val = CLASS_MAPPING[input_images[index][1]]
                random_file = random.choice(os.listdir(self.base_path+'/' + dir_val + '/'))
                input_img = input_img[:, :128, :]
                rand_img = cv2.imread(self.base_path+'/' + dir_val + '/' + random_file)[:, 128:, :]
                rand_img = cv2.cvtColor(rand_img, cv2.COLOR_BGR2RGB)
                input_img = np.concatenate((input_img, rand_img), axis=1)
                assert input_img.shape == (256, 256, 3)

            if self.transform is not None:
                input_img_as_tensor = self.transform(input_img)
                # save_image(input_img_as_tensor, str(index) + '.png')
            else:
                input_img_as_tensor = self.to_tensor(input_img)
        except:
            print(input_images[index][0])
            exit()

        return input_img_as_tensor, input_images[index][1]

    def __len__(self):
        return len(self.image_path)



class FaceFinetuneDataset(Dataset):
    """
    FineTuneDataset dataset
    """

    def __init__(self, base_path, base_path_ff, spatial_aug,  image_count, transform=None, should_invert=False, num_classes=2):
        super(Dataset, self).__init__()
        self.base_path = base_path
        self.base_path_ff = base_path_ff
        self.spatial_aug =  spatial_aug
        self.image_path = get_train_image_path(root=self.base_path + '/', image_count=image_count, num_classes=num_classes)
        self.transform = transform
        self.should_invert = should_invert
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, index):
        # Training input images path
        input_images = self.image_path
        # Assign label to class
        try:
            input_images = [(t, 0) if "orig" in t else (t, 1) for t in input_images]
            random_file1, random_file2, random_file3 = None, None, None
            input_img = cv2.imread(input_images[index][0])
            input_img = np.array(input_img, dtype='uint8')
            input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

            # Proposed Spatial Aug
            rand_int1 = torch.rand(1)
            if self.spatial_aug and "train" in input_images[index][0] and rand_int1[0] > 0.5:
                dir_val = CLASS_MAPPING[input_images[index][1]]
                random_file = random.choice(os.listdir(self.base_path_ff + '/' + dir_val + '/'))
                input_img = input_img[:, :128, :]
                rand_img = cv2.imread(self.base_path_ff + '/' + dir_val + '/' + random_file)[:, 128:, :]
                rand_img = cv2.cvtColor(rand_img, cv2.COLOR_BGR2RGB)
                input_img = np.concatenate((input_img, rand_img), axis=1)
                assert input_img.shape == (256, 256, 3)

            if self.transform is not None:
                input_img_as_tensor = self.transform(input_img)
                # save_image(input_img_as_tensor, str(index) + '.png')
            else:
                input_img_as_tensor = self.to_tensor(input_img)
        except:
            print(input_images[index][0])
            exit()

        return input_img_as_tensor, input_images[index][1]

    def __len__(self):
        return len(self.image_path)


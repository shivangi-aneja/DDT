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

DATASETS = {"ff", "vgg", "ff_recon", "f2f", "orig", "df", "fs", "dfdc"}
CLASS_MAPPING = {0: "orig", 1: "df", 2: "fs", 3: "f2f", 4: "nt"}
dlib_model_path = '/home/shivangi/Desktop/Projects/pretrained_models/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(dlib_model_path)

def get_available_datasets():
    """
    gets all the datasets
    :return:
    """
    return sorted(DATASETS)


def make_dataset(name, num_classes, base_path = None, fake_classes=None, transform=None, image_count=None, face_path=None, lip_path=None, mode='face'):
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
        return FaceDataset(image_count=image_count, fake_classes=fake_classes, base_path=base_path, transform=transform,
                           num_classes=num_classes)
    elif mode == 'face_finetune':
        return FaceFinetuneDataset(image_count=image_count, fake_classes=fake_classes, base_path=base_path, transform=transform,
                           num_classes=num_classes)
    elif mode == 'face_residual':
        return FaceResidualDataset(image_count=image_count, fake_classes=fake_classes, base_path=base_path, transform=transform,
                           num_classes=num_classes)
    elif mode == 'lip':
        return LipDataset(image_count=image_count, fake_classes=fake_classes, base_path=base_path, transform=transform,
                           num_classes=num_classes)
    elif mode == 'face_lip':
        return FaceLipDataset(image_count=image_count, fake_classes=fake_classes, face_path=face_path, lip_path=lip_path,
                              transform=transform, num_classes=num_classes)


def get_source_image_path(root, num_classes, fake_classes, image_count=1):
    real_path = root + 'orig/'
    df_path = root + 'df/'
    fs_path = root + 'fs/'
    f2f_path = root + 'f2f/'
    nt_path = root + 'nt/'
    dfdc_path = root + 'dfdc/'
    
    real_images = [os.path.join(real_path, f) for f in listdir(real_path) if isfile(join(real_path, f))]
    df_images = [os.path.join(df_path, f) for f in listdir(df_path) if isfile(join(df_path, f))]
    fs_images = [os.path.join(fs_path, f) for f in listdir(fs_path) if isfile(join(fs_path, f))]
    f2f_images = [os.path.join(f2f_path, f) for f in listdir(f2f_path) if isfile(join(f2f_path, f))]
    nt_images = [os.path.join(nt_path, f) for f in listdir(nt_path) if isfile(join(nt_path, f))]
    dfdc_images = [os.path.join(dfdc_path, f) for f in listdir(dfdc_path) if isfile(join(dfdc_path, f))]

    fake_image_list_map = {'df': df_images, 'fs': fs_images, 'f2f': f2f_images, 'nt': nt_images, 'dfdc': dfdc_images}
    
    source_image_list = None

    if num_classes == 2:
        real_list = list(real_images)
        fake_list = list(fake_image_list_map[fake_classes[0]])
        if image_count != 'all':
            real_list = random.sample(real_list, image_count)
            fake_list = random.sample(fake_list, image_count)
        source_image_list = list(real_list) + list(fake_list)

    elif num_classes == 3:
        real_list = list(real_images)
        fake1_list = list(fake_image_list_map[fake_classes[0]])
        fake2_list = list(fake_image_list_map[fake_classes[1]])
        if image_count != 'all':
            real_list = random.sample(real_list, image_count)
            fake1_list = random.sample(fake1_list, image_count)
            fake2_list = random.sample(fake2_list, image_count)
        source_image_list = list(real_list) + list(fake1_list) + list(fake2_list)

    elif num_classes == 4:
        real_list = list(real_images)
        fake1_list = list(fake_image_list_map[fake_classes[0]])
        fake2_list = list(fake_image_list_map[fake_classes[1]])
        fake3_list = list(fake_image_list_map[fake_classes[2]])
        if image_count != 'all':
            real_list = random.sample(real_list, image_count)
            fake1_list = random.sample(fake1_list, image_count)
            fake2_list = random.sample(fake2_list, image_count)
            fake3_list = random.sample(fake3_list, image_count)
        source_image_list = list(real_list) + list(fake1_list) + list(fake2_list) + list(fake3_list)
        # pairs = list(zip(real_images, fake_image_list_map[fake_classes[0]], fake_image_list_map[fake_classes[1]], fake_image_list_map[fake_classes[2]]))  # make pairs out of the two lists
        # if image_count != 'all':
        #     pairs = random.sample(pairs, image_count)   # random.sample(pairs, image_count)  # pick `image_count` random pairs
        # real_list, fake1_list, fake2_list, fake3_list = zip(*pairs)  # separate the pairs
        # source_image_list = list(real_list) + list(fake1_list) + list(fake2_list) + list(fake3_list)

    elif num_classes == 5:
        real_list = list(real_images)
        fake1_list = list(fake_image_list_map[fake_classes[0]])
        fake2_list = list(fake_image_list_map[fake_classes[1]])
        fake3_list = list(fake_image_list_map[fake_classes[2]])
        fake4_list = list(fake_image_list_map[fake_classes[3]])
        if image_count != 'all':
            real_list = random.sample(real_list, image_count)
            fake1_list = random.sample(fake1_list, image_count)
            fake2_list = random.sample(fake2_list, image_count)
            fake3_list = random.sample(fake3_list, image_count)
            fake4_list = random.sample(fake4_list, image_count)
        source_image_list = list(real_list) + list(fake1_list) + list(fake2_list) + list(fake3_list) + list(fake4_list)
        # pairs = list(zip(real_images, fake_image_list_map[fake_classes[0]], fake_image_list_map[fake_classes[1]], fake_image_list_map[fake_classes[2]], fake_image_list_map[fake_classes[3]]))  # make pairs out of the two lists
        # if image_count != 'all':
        #     pairs = random.sample(pairs, image_count)   # random.sample(pairs, image_count)  # pick `image_count` random pairs
        # real_list, fake1_list, fake2_list, fake3_list, fake4_list = zip(*pairs)  # separate the pairs
        # source_image_list = list(real_list) + list(fake1_list) + list(fake2_list) + list(fake3_list) + list(fake4_list)
    # print(source_image_list)
    return source_image_list



def get_finetune_image_path(root, image_count=1):
    real_path = root + 'orig/'
    f2f_path = root + 'f2f/'
    df_path = root + 'df/'
    fs_path = root + 'fs/'
    ft_image_list = []
    real_images = [os.path.join(real_path, f) for f in listdir(real_path) if isfile(join(real_path, f))]
    f2f_images = [os.path.join(f2f_path, f) for f in listdir(f2f_path) if isfile(join(f2f_path, f))]
    df_images = [os.path.join(df_path, f) for f in listdir(df_path) if isfile(join(df_path, f))]
    fs_images = [os.path.join(fs_path, f) for f in listdir(fs_path) if isfile(join(fs_path, f))]
    pairs = list(zip(real_images, f2f_images, df_images, fs_images))  # make pairs out of the two lists
    pairs = random.sample(pairs, image_count)  # pick `image_count` random pairs
    real_list, f2f_list, df_list, fs_list = zip(*pairs)  # separate the pairs
    ft_image_list = list(real_list) + list(f2f_list) + list(df_list) + list(fs_list)
    # print(ft_image_list)
    return ft_image_list


class CustomData(Dataset):
    """
    CustomData dataset
    """

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.labels

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.labels

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.input_images

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.input_images

    def __init__(self, name, base_path, num_classes, transform=None, should_invert=False):
        super(Dataset, self).__init__()
        self.base_path = base_path
        self.num_classes = num_classes
        self.inputFolderDataset = dset.ImageFolder(root=self.base_path + '/')
        self.transform = transform
        self.should_invert = should_invert
        self.to_tensor = transforms.ToTensor()
        # Training input images path
        input_images = self.inputFolderDataset.imgs
        # Assign label to class
        if self.num_classes == 2:
            # 0 for original, 1 for fake
            self.input_images = [(t[0], 0) if "orig" in t[0] else (t[0], 1) for t in input_images]
            self.labels = [0 if "orig" in t[0] else 1 for t in input_images]
        elif self.num_classes == 3:
            # 0 for Orig, 1 for F2F, 2 for DF, 3 for FS
            self.input_images = [(t[0], 0) if "orig" in t[0] else (t[0], 1) if "df" in t[0] else (t[0], 2) if "fs" in t[0] else (
                t[0], 3) for t in input_images]
            self.labels = [0 if "orig" in t[0] else 1 if "df" in t[0] else 2 if "fs" in t[0] else 3 for t in input_images]
        elif self.num_classes == 4:
            self.input_images = [
                (t[0], 0) if "orig" in t[0] else (t[0], 1) if "df" in t[0] else (t[0], 2) if "fs" in t[0] else (
                    t[0], 3) for t in input_images]
            self.labels = [0 if "orig" in t[0] else 1 if "df" in t[0] else 2 if "fs" in t[0] else 3 for t in
                           input_images]


    def __getitem__(self, index):

        input_img = cv2.imread(self.input_images[index][0])
        input_img = np.array(input_img, dtype='uint8')
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

        if self.should_invert:
            input_img = PIL.ImageOps.invert(input_img)

        if self.transform is not None:
            input_img_as_tensor = self.transform(input_img)
        else:
            input_img_as_tensor = self.to_tensor(input_img)

        return input_img_as_tensor, self.input_images[index][1]

    def __len__(self):
        return len(self.inputFolderDataset.imgs)


class DsneDataset(Dataset):
    """
    DsneDataset dataset
    """

    def __init__(self, name, base_path, image_count, img_path, transform=None, should_invert=False):
        super(Dataset, self).__init__()
        self.base_path = base_path
        self.image_path = get_source_image_path(root=self.base_path + '/', image_count=image_count)
        # self.image_path = get_target_image_path(root=self.base_path + '/', image_count=image_count)
        self.transform = transform
        self.should_invert = should_invert
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, index):
        # Training input images path
        input_images = self.image_path
        # Assign label to class
        # 0 for original, 1 for fake
        input_images = [(t, 0) if "orig" in t else (t, 1) for t in input_images]
        input_img = cv2.imread(input_images[index][0])
        input_img = np.array(input_img, dtype='uint8')
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

        if self.should_invert:
            input_img = PIL.ImageOps.invert(input_img)

        if self.transform is not None:
            input_img_as_tensor = self.transform(input_img)
        else:
            input_img_as_tensor = self.to_tensor(input_img)

        return input_img_as_tensor, input_images[index][1]

    def __len__(self):
        return len(self.image_path)


class FaceDataset(Dataset):
    """
    FaceDataset dataset
    """

    def __init__(self, base_path, image_count, fake_classes, transform=None, should_invert=False, num_classes=2):
        super(Dataset, self).__init__()
        self.base_path = base_path
        self.image_path = get_source_image_path(root=self.base_path + '/', image_count=image_count, fake_classes=fake_classes, num_classes=num_classes)
        self.transform = transform
        self.should_invert = should_invert
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, index):
        # Training input images path
        input_images = self.image_path
        # Assign label to class
        try:
            input_images = [(t, 0) if "orig" in t else (t, 1) if "df" in t else (t, 2) if "fs" in t else (t, 3) if "f2f" in t else (t, 4) for t in input_images]
            random_file1, random_file2, random_file3 = None, None, None
            input_img = cv2.imread(input_images[index][0])
            input_img = np.array(input_img, dtype='uint8')
            input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

            rand_int1 = torch.rand(1)
            # if "train" in input_images[index][0] and rand_int1[0] > 0.5:
            #     dir_val = CLASS_MAPPING[input_images[index][1]]
            #     random_file = random.choice(os.listdir(self.base_path+'/' + dir_val + '/'))
            #     input_img = input_img[:, :128, :]
            #     rand_img = cv2.imread(self.base_path+'/' + dir_val + '/' + random_file)[:, 128:, :]
            #     rand_img = cv2.cvtColor(rand_img, cv2.COLOR_BGR2RGB)
            #     input_img = np.concatenate((input_img, rand_img), axis=1)
            #     assert input_img.shape == (256, 256, 3)

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

    def __init__(self, base_path, image_count, fake_classes, transform=None, should_invert=False, num_classes=2):
        super(Dataset, self).__init__()
        self.base_path = base_path
        self.image_path = get_source_image_path(root=self.base_path + '/', image_count=image_count, fake_classes=fake_classes, num_classes=num_classes)
        self.transform = transform
        self.should_invert = should_invert
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, index):
        # Training input images path
        input_images = self.image_path
        # Assign label to class
        try:
            input_images = [(t, 0) if "orig" in t else (t, 1) if "df" in t else (t, 2) if "fs" in t else (t, 3) if "f2f" in t else (t, 4) for t in input_images]
            random_file1, random_file2, random_file3 = None, None, None
            input_img = cv2.imread(input_images[index][0])
            input_img = np.array(input_img, dtype='uint8')
            input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

            rand_int1 = torch.rand(1)
            rand_int2 = torch.rand(1)
            if "train" in input_images[index][0] and rand_int1[0] > 0.5:
                dir_val = CLASS_MAPPING[input_images[index][1]]
                # if rand_int2[0] > 0.5:
                # dir_val = 'nt' if dir_val == 'df' else 'orig'
                if dir_val != 'orig':
                    dir_val = 'df' if rand_int2 < 0.5 else 'nt'
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

class LipDataset(Dataset):
    """
    Lip Dataset dataset
    """

    def __init__(self, base_path, image_count, fake_classes, transform=None, should_invert=False, num_classes=2):
        super(Dataset, self).__init__()
        self.base_path = base_path
        self.image_path = get_source_image_path(root=self.base_path + '/', image_count=image_count, fake_classes=fake_classes, num_classes=num_classes)
        self.transform = transform
        self.should_invert = should_invert
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, index):

        input_images = self.image_path
        # Assign label to class
        try:
            input_images = [(t, 0) if "orig" in t else (t, 1) if "df" in t else (t, 2) if "fs" in t else (t, 3) if "f2f" in t else (t, 4) for t in input_images]
            input_img = cv2.imread(input_images[index][0])
            input_img = np.array(input_img, dtype='uint8')
            input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

            # resize_img = imutils.resize(input_img, width=500)
            # rects = detector(resize_img, 1)
            #
            # if len(rects) == 0:
            #     x, y, w, h = 176, 305, 157, 54
            # else:
            #     shape = predictor(resize_img, rects[0])
            #     shape = face_utils.shape_to_np(shape)
            #     name = 'mouth'
            #     (i, j) = face_utils.FACIAL_LANDMARKS_IDXS[name]
            #     x, y, w, h = cv2.boundingRect(np.array([shape[i:j]]))
            # y_min = max(0, y - 30)
            # x_min = max(0, x - 20)
            # mouth_img = resize_img[y_min:y + h + 30, x_min:x + w + 20]
            # imutils.resize(mouth_img, width=250, inter=cv2.INTER_CUBIC)
            # mouth_img = cv2.resize(mouth_img, (250, 150))

            if self.should_invert:
                input_img = PIL.ImageOps.invert(input_img)

            if self.transform is not None:
                input_img_as_tensor = self.transform(input_img)
            else:
                input_img_as_tensor = self.to_tensor(input_img)
        except:
            print(input_images[index][0])
            exit()

        return input_img_as_tensor, input_images[index][1]

    def __len__(self):
        return len(self.image_path)


class FaceLipDataset(Dataset):
    """
    Face Lip Dataset dataset
    """

    def __init__(self, face_path, lip_path, image_count, fake_classes, transform=None, should_invert=False, num_classes=2):
        super(Dataset, self).__init__()
        self.face_path = face_path
        self.lip_path = lip_path
        self.image_path = get_source_image_path(root=self.face_path + '/', image_count=image_count, fake_classes=fake_classes, num_classes=num_classes)
        self.transform = transform
        self.should_invert = should_invert
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, index):

        face_images = self.image_path
        lip_images = [sub.replace('ff_face', 'ff_lip') for sub in self.image_path]
        # Assign label to class
        try:
            face_images = [(t, 0) if "orig" in t else (t, 1) if "df" in t else (t, 2) if "fs" in t else (t, 3) if "f2f" in t else (t, 4) for t in face_images]
            lip_images = [(t, 0) if "orig" in t else (t, 1) if "df" in t else (t, 2) if "fs" in t else (t, 3) if "f2f" in t else (t, 4) for t in lip_images]

            face_img = cv2.imread(face_images[index][0])
            face_img = np.array(face_img, dtype='uint8')
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

            lip_img = cv2.imread(lip_images[index][0])
            lip_img = np.array(lip_img, dtype='uint8')
            lip_img = cv2.cvtColor(lip_img, cv2.COLOR_BGR2RGB)
            lip_img = cv2.resize(lip_img, (250, 250))

            if self.should_invert:
                face_img = PIL.ImageOps.invert(face_img)
                lip_img = PIL.ImageOps.invert(lip_img)

            if self.transform is not None:
                face_img_as_tensor = self.transform(face_img)
                lip_img_as_tensor = self.transform(lip_img)
            else:
                face_img_as_tensor = self.to_tensor(face_img)
                lip_img_as_tensor = self.to_tensor(lip_img)
        except:
            print(face_img[index][0])
            print(lip_img[index][0])
            exit()

        return face_img_as_tensor, lip_img_as_tensor, face_images[index][1]

    def __len__(self):
        return len(self.image_path)


class FaceResidualDataset(Dataset):
    """
    Face6ChannelsDataset dataset
    """

    def __init__(self, base_path, image_count, fake_classes, transform=None, should_invert=False, num_classes=2):
        super(Dataset, self).__init__()
        self.base_path = base_path
        # self.gaussian_noise = iaa.AdditiveGaussianNoise(scale=(10, 60))     # add gaussian noise
        self.image_path = get_source_image_path(root=self.base_path + '/', image_count=image_count, fake_classes=fake_classes, num_classes=num_classes)
        self.transform = transform
        self.should_invert = should_invert
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, index):
        # Training input images path
        input_images = self.image_path
        # Assign label to class
        try:
            input_images = [(t, 0) if "orig" in t else (t, 1) if "df" in t else (t, 2) if "fs" in t else (t, 3) if "f2f" in t else (t, 4) for t in input_images]
            input_img = cv2.imread(input_images[index][0])
            input_img = np.array(input_img, dtype='uint8')

            image_row = (-1 * input_img[1:-3, 2:-2, :] + 3 * input_img[2:-2, 2:-2, :] - 3 * input_img[3:-1, 2:-2, :] + 1 * input_img[4:,2:-2, :])
            image_row = cv2.resize(image_row, (256, 256))
            image_col = (-1 * input_img[2:-2, 1:-3, :] + 3 * input_img[2:-2, 2:-2, :] - 3 * input_img[2:-2, 3:-1, :] + 1 * input_img[2:-2, 4:, :])
            image_col = cv2.resize(image_col, (256, 256))
            input_img = np.concatenate((image_row, image_col), 2) / 8

            if self.should_invert:
                input_img = PIL.ImageOps.invert(input_img)
            # if self.transform is not None:
            #     input_img_as_tensor = self.transform(input_img)
            # else:
            input_img_as_tensor = self.to_tensor(input_img)
        except:
            print(input_images[index][0])
            exit()

        return input_img_as_tensor, input_images[index][1]

    def __len__(self):
        return len(self.image_path)
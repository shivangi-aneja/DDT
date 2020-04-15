import numpy as np
import os
import torch
from os.path import isdir, isfile, join
from os import listdir
from torch.utils.data import Dataset
import cv2
import torchvision.transforms as transforms
import random

DATASETS = {"ff"}


def get_available_datasets():
    """
    Gets all the datasets
    :return: DATASETS
    """
    return sorted(DATASETS)


def make_dataset(name, num_classes, base_path, transform, image_count, fake_classes=None, loader_type='face', depth=None):
    """
    Returns dataset according to its name
    :param name:
    :param num_classes:
    :param base_path:
    :param split_file:
    :param transform:
    :param image_count:
    :param fake_classes:
    :param loader_type:
    :param mode:
    :return: Dataset
    """
    name = name.strip().lower()
    if not name in DATASETS:
        raise ValueError("invalid dataset: '{0}'".format(name))
    elif loader_type == 'face':
        return FaceDataset(num_classes, base_path, transform, image_count, fake_classes)
    elif loader_type == 'lip_seq':
        return LipSeqDataset(num_classes, base_path, image_count, fake_classes)
    elif loader_type == 'face_lip_seq':
        return FaceLipSeqDataset(num_classes, base_path, image_count, fake_classes, transform)


def get_frame_list(dir_path, frame_cnt=30):
    frame_list = list()
    for _dir in dir_path:
        frames = [os.path.join(_dir, f) for f in listdir(_dir) if isfile(join(_dir, f))]
        frames.sort()
        if frame_cnt == len(frames):
            frame_list.extend(frames)
        else:
            n_frames = len(frames) - 10 + 1
            interval = int(n_frames / frame_cnt)
            for i in range(frame_cnt):
                frame_list.append(frames[i * interval])
    return frame_list


# def get_all_frames(dir_path):
#     frame_list = list()
#     for _dir in dir_path:
#         frames = [os.path.join(_dir, f) for f in listdir(_dir) if isfile(join(_dir, f))]
#         frames.sort()
#         frame_list.extend(frames)
#     return frame_list


def get_dir_path(root, num_classes, fake_classes, image_count):
    real_path = root + 'orig/'
    df_path = root + 'df/'
    fs_path = root + 'fs/'
    f2f_path = root + 'f2f/'
    nt_path = root + 'nt/'

    real_dir = [os.path.join(real_path, f) for f in listdir(real_path) if isdir(join(real_path, f))]
    df_dir = [os.path.join(df_path, f) for f in listdir(df_path) if isdir(join(df_path, f))]
    fs_dir = [os.path.join(fs_path, f) for f in listdir(fs_path) if isdir(join(fs_path, f))]
    f2f_dir = [os.path.join(f2f_path, f) for f in listdir(f2f_path) if isdir(join(f2f_path, f))]
    nt_dir = [os.path.join(nt_path, f) for f in listdir(nt_path) if isdir(join(nt_path, f))]

    fake_dir_list_map = {'df': df_dir, 'fs': fs_dir, 'f2f': f2f_dir, 'nt': nt_dir}
    image_list = None
    frame_cnt = 30
    tsne_cnt = 10

    if num_classes == 2:
        real_list = list(real_dir)
        fake_list = list(fake_dir_list_map[fake_classes[0]])
        if image_count == 'all':
            image_list = get_frame_list(dir_path=real_list, frame_cnt=frame_cnt) + get_frame_list(dir_path=fake_list,
                                                                                                  frame_cnt=frame_cnt)
        elif image_count == 'tsne':
            # real_list = random.sample(real_list, image_count)
            # fake_list = random.sample(fake_list, image_count)
            image_list = get_frame_list(dir_path=real_list, frame_cnt=tsne_cnt) + get_frame_list(dir_path=fake_list, frame_cnt=tsne_cnt)
        else:
            real_list = random.sample(real_list, image_count)
            fake_list = random.sample(fake_list, image_count)
            image_list = get_frame_list(dir_path=real_list, frame_cnt=1) + get_frame_list(dir_path=fake_list,
                                                                                          frame_cnt=1)


    elif num_classes == 3:
        real_list = list(real_dir)
        fake1_list = list(fake_dir_list_map[fake_classes[0]])
        fake2_list = list(fake_dir_list_map[fake_classes[1]])
        if image_count != 'all':
            real_list = random.sample(real_list, image_count)
            fake1_list = random.sample(fake1_list, image_count)
            fake2_list = random.sample(fake2_list, image_count)
            image_list = get_frame_list(dir_path=real_list, frame_cnt=1) + get_frame_list(dir_path=fake1_list,
                        frame_cnt=1) + get_frame_list(dir_path=fake2_list, frame_cnt=1)
        else:
            image_list = get_frame_list(dir_path=real_list, frame_cnt=frame_cnt) + get_frame_list(dir_path=fake1_list,
                        frame_cnt=frame_cnt) + get_frame_list(dir_path=fake2_list, frame_cnt=frame_cnt)


    elif num_classes == 4:
        real_list = list(real_dir)
        fake1_list = list(fake_dir_list_map[fake_classes[0]])
        fake2_list = list(fake_dir_list_map[fake_classes[1]])
        fake3_list = list(fake_dir_list_map[fake_classes[2]])
        if image_count != 'all':
            real_list = random.sample(real_list, image_count)
            fake1_list = random.sample(fake1_list, image_count)
            fake2_list = random.sample(fake2_list, image_count)
            fake3_list = random.sample(fake3_list, image_count)
            image_list = get_frame_list(dir_path=real_list, frame_cnt=1) + get_frame_list(dir_path=fake1_list,
                        frame_cnt=1) + get_frame_list(dir_path=fake2_list, frame_cnt=1) + get_frame_list(dir_path=fake3_list, frame_cnt=1)
        else:
            image_list = get_frame_list(dir_path=real_list, frame_cnt=frame_cnt) + get_frame_list(dir_path=fake1_list,
                        frame_cnt=frame_cnt) + get_frame_list(dir_path=fake2_list, frame_cnt=frame_cnt) + get_frame_list(dir_path=fake3_list, frame_cnt=frame_cnt)


    elif num_classes == 5:
        real_list = list(real_dir)
        fake1_list = list(fake_dir_list_map[fake_classes[0]])
        fake2_list = list(fake_dir_list_map[fake_classes[1]])
        fake3_list = list(fake_dir_list_map[fake_classes[2]])
        fake4_list = list(fake_dir_list_map[fake_classes[3]])
        if image_count != 'all':
            real_list = random.sample(real_list, image_count)
            fake1_list = random.sample(fake1_list, image_count)
            fake2_list = random.sample(fake2_list, image_count)
            fake3_list = random.sample(fake3_list, image_count)
            fake4_list = random.sample(fake4_list, image_count)
            image_list = get_frame_list(dir_path=real_list, frame_cnt=1) + get_frame_list(dir_path=fake1_list,
                                                                                          frame_cnt=1) \
                         + get_frame_list(dir_path=fake2_list, frame_cnt=1) + get_frame_list(
                dir_path=fake3_list, frame_cnt=1) \
                         + get_frame_list(dir_path=fake4_list, frame_cnt=1)
        else:
            image_list = get_frame_list(dir_path=real_list, frame_cnt=frame_cnt) + get_frame_list(dir_path=fake1_list,
                                                                                                  frame_cnt=frame_cnt) \
                         + get_frame_list(dir_path=fake2_list, frame_cnt=frame_cnt) + get_frame_list(
                dir_path=fake3_list, frame_cnt=frame_cnt) \
                         + get_frame_list(dir_path=fake4_list, frame_cnt=frame_cnt)

    return image_list


class FaceDataset(Dataset):
    def __init__(self, num_classes, base_path, transform, image_count, fake_classes, frame_used=30):
        super(Dataset, self).__init__()
        self.num_classes = num_classes
        self.fake_classes = fake_classes
        self.root_dir = base_path
        self.image_count = image_count
        self.transform = transform
        self.to_tensor = transforms.ToTensor()
        self.frame_used = frame_used
        self.keys = get_dir_path(self.root_dir, self.num_classes, self.fake_classes, self.image_count)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):

        input_images = self.keys
        # Assign label to class
        try:
            input_images = [
                (t, 0) if "orig" in t else (t, 1) if "df" in t else (t, 2) if "fs" in t else (t, 3) if "f2f" in t else (
                t, 4) for t in input_images]
            input_img = cv2.imread(input_images[idx][0])
            input_img = np.array(input_img, dtype='uint8')
            input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

            if self.transform is not None:
                input_img_as_tensor = self.transform(input_img)
            else:
                input_img_as_tensor = self.to_tensor(input_img)
        except:
            print(input_images[idx][0])
            exit()

        return input_img_as_tensor, input_images[idx][1]


class LipSeqDataset(Dataset):
    def __init__(self, num_classes, base_path, image_count, fake_classes):
        super(Dataset, self).__init__()
        self.num_classes = num_classes
        self.fake_classes = fake_classes
        self.root_dir = base_path
        self.image_count = image_count
        self.keys = get_dir_path(self.root_dir, self.num_classes, self.fake_classes, self.image_count)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        input_images = self.keys
        try:
            input_images = [(t, 0) if "orig" in t else (t, 1) if "df" in t else (t, 2) if "fs" in t else (t, 3) if "f2f" in t else (t, 4) for t in input_images]
            input_img_as_tensor = torch.load(input_images[idx][0])
        except:
            print(input_images[idx][0])
            exit()

        return input_img_as_tensor, input_images[idx][1]


class FaceLipSeqDataset(Dataset):
    def __init__(self, num_classes, base_path, image_count, fake_classes, transform):
        super(Dataset, self).__init__()
        self.num_classes = num_classes
        self.fake_classes = fake_classes
        self.root_dir = base_path
        self.image_count = image_count
        self.transform = transform
        self.to_tensor = transforms.ToTensor()
        self.keys = get_dir_path(self.root_dir, self.num_classes, self.fake_classes, self.image_count)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        input_images = self.keys
        try:
            input_images = [(t, 0) if "orig" in t else (t, 1) if "df" in t else (t, 2) if "fs" in t else (t, 3) if "f2f" in t else (t, 4) for t in input_images]
            face_img, lip_seq = torch.load(input_images[idx][0])

            if self.transform is not None:
                face_img_as_tensor = self.transform(face_img)
            else:
                face_img_as_tensor = self.to_tensor(face_img)

        except:
            print(input_images[idx][0])
            exit()

        return face_img_as_tensor, lip_seq, input_images[idx][1]

class LipSeqDatasetOld(Dataset):
    def __init__(self, depth, num_classes, base_path, transform, image_count, fake_classes):
        super(Dataset, self).__init__()
        self.num_classes = num_classes
        self.fake_classes = fake_classes
        self.root_dir = base_path
        self.image_count = image_count
        self.transform = transform
        self.to_tensor = transforms.ToTensor()
        self.frame_used = 30
        self.in_channel = depth
        self.img_rows = 150
        self.img_cols = 250
        self.keys, self.all_frame_list = get_dir_path(self.root_dir, self.num_classes, self.fake_classes, self.image_count, self.frame_used)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):

        input_images = self.keys
        frame_idx = 0
        # Assign label to class
        try:
            input_images = [(t, 0) if "orig" in t else (t, 1) if "df" in t else (t, 2) if "fs" in t else (t, 3) if "f2f" in t else (t, 4) for t in input_images]
            lip_seq = torch.zeros([3 * self.in_channel,  self.img_rows, self.img_cols], dtype=torch.float64)
            for j in range(self.in_channel):
                img_path = input_images[idx][0]
                req_idx = self.all_frame_list.index(img_path)
                frame_idx = req_idx + j
                input_img = cv2.imread(self.all_frame_list[frame_idx])
                input_img = np.array(input_img, dtype='uint8')
                input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
                input_img = self.transform(input_img)
                lip_seq[3*j:3*j+3, :, :] = input_img

        except:
            print(self.all_frame_list[frame_idx])
            # exit()

        return lip_seq, input_images[idx][1]
import torch
import numpy as np
from os.path import isdir, isfile, join
from os import listdir
import os

category_to_label = {0: 'Orig', 1: 'DF', 2: 'FS', 3: 'F2F', 4: 'NT'}
label_to_category = {'orig': 0, 'df': 1, 'fs': 2, 'f2f': 3, 'nt': 4}
category_to_color = np.array(['#d92739', '#1987d1', '#09b526', '#8905b5', '#00ffff'])
width = 4000
height = 3000
max_dim = 100

def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def calc_activation_vector(latent_dim, z):
    latent_half = latent_dim // 2  # 64
    z_real = torch.mean(torch.abs(z[:, :latent_half]), dim=1)
    z_fake = torch.mean(torch.abs(z[:, latent_half:]), dim=1)
    act_vector = torch.stack((z_real, z_fake), dim=1)
    return act_vector


def rect_to_bb(rect):
    # https://gist.github.com/shravankumar147/056626de3fbdc7cf7b59de1d9f6279d1
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    # return a tuple of (x, y, w, h)
    return (x, y, w, h)


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


def get_frame_list(dir_path, frame_cnt_per_video):
    frame_list = list()
    frames = [os.path.join(dir_path, f) for f in listdir(dir_path) if isfile(join(dir_path, f))]
    frames.sort()
    n_frames = len(frames) - 1 + 1
    interval = int(n_frames / frame_cnt_per_video)
    for i in range(frame_cnt_per_video):
        frame_list.append(frames[i * interval])
    return frame_list


def get_all_frames(dir_path):
    frames = [os.path.join(dir_path, f) for f in listdir(dir_path) if isfile(join(dir_path, f))]
    frames.sort()
    return frames

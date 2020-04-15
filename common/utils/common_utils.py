import torch
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from tqdm import tqdm
from os.path import isdir, isfile, join
from os import listdir
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random
import os
# from mpl_toolkits.mplot3d import Axes3D
from sklearn import manifold
# from tsnecuda import TSNE

category_to_label = {0: 'Orig', 1: 'DF', 2: 'FS', 3: 'F2F', 4: 'NT'}
label_to_category = {'orig': 0, 'df': 1, 'fs': 2, 'f2f': 3, 'nt': 4}
category_to_color = np.array(['#d92739', '#1987d1', '#09b526', '#8905b5', '#00ffff'])
width = 4000
height = 3000
max_dim = 100


def calc_activation_vector(latent_dim, z):
    latent_half = latent_dim // 2  # 64
    z_real = torch.mean(torch.abs(z[:, :latent_half]), dim=1)
    z_fake = torch.mean(torch.abs(z[:, latent_half:]), dim=1)
    act_vector = torch.stack((z_real, z_fake), dim=1)
    return act_vector


def k_nearest_neighbor(source_embed, target_embed, k=1):

    # compute the L2 norm between every pair of source and target embedding
    # src = N1, tgt = N2, dist = N2*N1
    dist = torch.cdist(target_embed, source_embed, p=2)
    # values = N2*k, indices = N2*k
    values, indices = torch.topk(dist, k=k, largest=False)
    return values, indices


def visualize_latent_tsne_classifier(loader, file_name, best_path, model_name, model):
    """
    Function to evaluate the result on test data
    :param val_loader:
    :param model_path:target_all
    :param image_path_pred:
    :param image_path_gt:
    :return:
    """
    print("Loading Saved Model")
    print(best_path + model_name)
    checkpoint = torch.load(best_path + model_name)
    model.load_state_dict(checkpoint)
    print("Saved Model successfully loaded")
    try:
        print("Loading Saved Model")
        print(best_path+ model_name)
        checkpoint = torch.load(best_path + model_name)
        model.load_state_dict(checkpoint)
        print("Saved Model successfully loaded")
    except:
        print("Model not found.")
        exit()
    model.eval()
    images_all, latent_all, target_all = None, None, None
    with torch.no_grad():
        for epoch_iter, data in enumerate(tqdm(loader, desc='')):
            input_image, target = data
            input_image = input_image.cuda()
            target = target.cuda()
            latent, _ = model(input_image)
            if latent_all is None:
                latent_all = latent
                target_all = target
                images_all = input_image
            else:
                latent_all = torch.cat([latent_all, latent], dim=0)
                target_all = torch.cat([target_all, target], dim=0)
                images_all = torch.cat([images_all, input_image], dim=0)
    images_all = images_all.cpu().numpy()
    latent_all = latent_all.cpu().numpy()
    target_all = target_all.cpu().numpy()
    target_all = target_all.astype(int)
    x_tsne = latent_all

    tx, ty = x_tsne[:, 0], x_tsne[:, 1]
    tx = (tx - np.min(tx)) / (np.max(tx) - np.min(tx))
    ty = (ty - np.min(ty)) / (np.max(ty) - np.min(ty))

    full_image = Image.new('RGB', (width, height))
    for idx, x in enumerate(images_all):
        tile = Image.fromarray(np.uint8((0.5*np.transpose(x, [1, 2, 0])+0.5) * 255))
        # color = Image.new("RGB", (256, 256), category_to_color[target_all[idx]])
        tile = ImageOps.expand(tile, border=30, fill=category_to_color[target_all[idx]])
        # tile = Image.blend(tile, color, alpha=0.5)
        # Image._show(tile)
        rs = max(1, tile.width / max_dim, tile.height / max_dim)
        tile = tile.resize((int(tile.width / rs),
                            int(tile.height / rs)),
                           Image.ANTIALIAS)
        full_image.paste(tile, (int((width - max_dim) * tx[idx]),
                                int((height - max_dim) * ty[idx])))

    plt.figure(figsize=(16, 12))
    plt.imsave(fname=file_name + ".png", arr=full_image)


def visualize_latent_tsne(loader, file_name, best_path, model_name, model, mode='vae'):
    """
    Function to evaluate the result on test data
    :param val_loader:
    :param model_path:target_all
    :param image_path_pred:
    :param image_path_gt:
    :return:
    """
    print("Loading Saved Model")
    print(best_path + model_name)
    checkpoint = torch.load(best_path + model_name)
    model.load_state_dict(checkpoint)
    print("Saved Model successfully loaded")
    try:
        print("Loading Saved Model")
        print(best_path+ model_name)
        checkpoint = torch.load(best_path + model_name)
        model.load_state_dict(checkpoint)
        print("Saved Model successfully loaded")
    except:
        print("Model not found.")
        exit()
    model.eval()
    images_all, latent_all, target_all = None, None, None
    with torch.no_grad():
        for epoch_iter, data in enumerate(tqdm(loader, desc='')):
            input_image, target = data
            input_image = input_image.cuda().float()
            target = target.cuda().float()

            if mode == 'ft':
                latent = model(input_image)
            elif mode == 'ccsa' or mode == 'classifier':
                _, latent = model(input_image)
            else:
                latent, _, _ = model(input_image)
            if latent_all is None:
                latent_all = latent
                target_all = target
                images_all = input_image
            else:
                latent_all = torch.cat([latent_all, latent], dim=0)
                target_all = torch.cat([target_all, target], dim=0)
                images_all = torch.cat([images_all, input_image], dim=0)
    images_all = images_all.cpu().numpy()
    latent_all = latent_all.cpu().numpy()
    target_all = target_all.cpu().numpy()
    target_all = target_all.astype(int)
    print("T-SNE visualization started")
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    x_tsne = tsne.fit_transform(latent_all)

    if mode == 'residual':

        # Both
        fig = plt.figure()
        ax = fig.add_subplot(111)
        print(np.unique(target_all))
        for label_id in np.unique(target_all):
            ax.scatter(x_tsne[np.where(target_all == label_id), 0][:500],
                       x_tsne[np.where(target_all == label_id), 1][:500],
                       marker='o', linewidth=1, color=plt.cm.Set1(label_id),
                       alpha=0.8, label=category_to_label[label_id])
        plt.legend(loc='best')
        fig.savefig(file_name + "_both.png", dpi=fig.dpi)
        plt.close()

    else:
        tx, ty = x_tsne[:, 0], x_tsne[:, 1]
        tx = (tx - np.min(tx)) / (np.max(tx) - np.min(tx))
        ty = (ty - np.min(ty)) / (np.max(ty) - np.min(ty))

        full_image = Image.new('RGB', (width, height))
        for idx, x in enumerate(images_all):
            tile = Image.fromarray(np.uint8((0.5*np.transpose(x, [1, 2, 0])+0.5) * 255))
            # tile = Image.fromarray(np.uint8((np.transpose(x, [1, 2, 0])) * 255))
            # color = Image.new("RGB", (256, 256), category_to_color[target_all[idx]])
            tile = ImageOps.expand(tile, border=30, fill=category_to_color[target_all[idx]])
            # tile = Image.blend(tile, color, alpha=0.5)
            # Image._show(tile)
            rs = max(1, tile.width / max_dim, tile.height / max_dim)
            tile = tile.resize((int(tile.width / rs),
                                int(tile.height / rs)),
                               Image.ANTIALIAS)
            full_image.paste(tile, (int((width - max_dim) * tx[idx]),
                                    int((height - max_dim) * ty[idx])))

        plt.figure(figsize=(16, 12))
        plt.imsave(fname=file_name + "_images_both.png", arr=full_image)






    # Orig
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(x_tsne[np.where(target_all == 0), 0][:500], x_tsne[np.where(target_all == 0), 1][:500],
    #            marker='o', linewidth=1, color=plt.cm.Set1(0), alpha=0.8, label=category_to_label[0])
    # plt.legend(loc='best')
    # fig.savefig(file_name + "_orig.png", dpi=fig.dpi)
    # plt.close()

    # Fake
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(x_tsne[np.where(target_all == 1), 0][:500], x_tsne[np.where(target_all == 1), 1][:500],
    #            marker='o', linewidth=1, color=plt.cm.Set1(1), alpha=0.8, label=category_to_label[1])
    # plt.legend(loc='best')
    # fig.savefig(file_name + "_fake.png", dpi=fig.dpi)
    # plt.close()

def visualize_latent_tsne_combined(loader, file_name, best_path, model_name, model):
    """
    Function to evaluate the result on test data
    :param val_loader:
    :param model_path:target_all
    :param image_path_pred:
    :param image_path_gt:
    :return:
    """
    print("Loading Saved Model")
    print(best_path + model_name)
    checkpoint = torch.load(best_path + model_name)
    model.load_state_dict(checkpoint)
    print("Saved Model successfully loaded")
    try:
        print("Loading Saved Model")
        print(best_path+ model_name)
        checkpoint = torch.load(best_path + model_name)
        model.load_state_dict(checkpoint)
        print("Saved Model successfully loaded")
    except:
        print("Model not found.")
        exit()
    model.eval()
    images_all, latent_all, target_all = None, None, None
    with torch.no_grad():
        for epoch_iter, data in enumerate(tqdm(loader, desc='')):
            face_image, lip_image, target = data
            face_image = face_image.cuda()
            lip_image = lip_image.cuda().float()
            target = target.cuda()
            latent, _, _ = model(face_image, lip_image)
            if latent_all is None:
                latent_all = latent
                target_all = target
                images_all = face_image
            else:
                latent_all = torch.cat([latent_all, latent], dim=0)
                target_all = torch.cat([target_all, target], dim=0)
                images_all = torch.cat([images_all, face_image], dim=0)
    images_all = images_all.cpu().numpy()
    latent_all = latent_all.cpu().numpy()
    target_all = target_all.cpu().numpy()
    target_all = target_all.astype(int)
    print("T-SNE visualization started")
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    x_tsne = tsne.fit_transform(latent_all)

    tx, ty = x_tsne[:, 0], x_tsne[:, 1]
    tx = (tx - np.min(tx)) / (np.max(tx) - np.min(tx))
    ty = (ty - np.min(ty)) / (np.max(ty) - np.min(ty))

    full_image = Image.new('RGB', (width, height))
    for idx, x in enumerate(images_all):
        tile = Image.fromarray(np.uint8((0.5*np.transpose(x, [1, 2, 0])+0.5) * 255))
        # color = Image.new("RGB", (256, 256), category_to_color[target_all[idx]])
        tile = ImageOps.expand(tile, border=30, fill=category_to_color[target_all[idx]])
        # tile = Image.blend(tile, color, alpha=0.5)
        # Image._show(tile)
        rs = max(1, tile.width / max_dim, tile.height / max_dim)
        tile = tile.resize((int(tile.width / rs),
                            int(tile.height / rs)),
                           Image.ANTIALIAS)
        full_image.paste(tile, (int((width - max_dim) * tx[idx]),
                                int((height - max_dim) * ty[idx])))

    plt.figure(figsize=(16, 12))
    plt.imsave(fname=file_name + "_images_both.png", arr=full_image)

def visualize_latent_tsne_autoencoder(loader, file_name, best_path, model_name, model, mode='vae'):
    """
    Function to evaluate the result on test data
    :param val_loader:
    :param model_path:target_all
    :param image_path_pred:
    :param image_path_gt:
    :return:
    """
    print("Loading Saved Model")
    print(best_path + model_name)
    checkpoint = torch.load(best_path + model_name)
    model.load_state_dict(checkpoint)
    print("Saved Model successfully loaded")
    try:
        print("Loading Saved Model")
        print(best_path+ model_name)
        checkpoint = torch.load(best_path + model_name)
        model.load_state_dict(checkpoint)
        print("Saved Model successfully loaded")
    except:
        print("Model not found.")
        exit()
    model.eval()
    images_all, latent_all, target_all = None, None, None
    with torch.no_grad():
        for epoch_iter, data in enumerate(tqdm(loader, desc='')):
            input_image, target = data
            input_image = input_image.cuda().float()
            target = target.cuda()

            if mode == 'ft':
                latent, _ = model(input_image)
            elif mode == 'ccsa':
                _, latent, _ = model(input_image)
            else:
                latent, _, _, _ = model(input_image)
            if latent_all is None:
                latent_all = latent
                target_all = target
                images_all = input_image
            else:
                latent_all = torch.cat([latent_all, latent], dim=0)
                target_all = torch.cat([target_all, target], dim=0)
                images_all = torch.cat([images_all, input_image], dim=0)

    latent_all = latent_all.cpu().numpy()
    target_all = target_all.cpu().numpy()
    target_all = target_all.astype(int)
    print("T-SNE visualization started")
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    x_tsne = tsne.fit_transform(latent_all)

    # Both
    fig = plt.figure()
    ax = fig.add_subplot(111)
    print(np.unique(target_all))
    for label_id in np.unique(target_all):
        ax.scatter(x_tsne[np.where(target_all == label_id), 0][:500], x_tsne[np.where(target_all == label_id), 1][:500],
                   marker='o', linewidth=1, color=plt.cm.Set1(label_id),
                   alpha=0.8, label=category_to_label[label_id])
    plt.legend(loc='best')
    fig.savefig(file_name + "_both.png", dpi=fig.dpi)
    plt.close()



def calculate_clustering_metrics(data_loader, best_path, model_name, model, best_path_classifier,classifier):
    """
    Function to evaluate the result on test data
    :param val_loader:
    :param model_path:target_all
    :param image_path_pred:
    :param image_path_gt:
    :return:
    """

    try:
        print("Loading Saved Model")
        print(best_path)
        checkpoint = torch.load(best_path + model_name)
        model.load_state_dict(checkpoint)
        checkpoint_classifier = torch.load(best_path_classifier + model_name)
        classifier.load_state_dict(checkpoint_classifier)
        print("Saved Model successfully loaded")
    except:
        print("Model not found.")
        exit()
    model.eval()
    pred_all, target_all = None, None
    with torch.no_grad():
        for epoch_iter, data in enumerate(tqdm(data_loader, desc='')):
            input_image, target = data
            input_image = input_image.cuda()
            target = target.cuda()
            latent, _, _ = model(input_image)
            pred = classifier(latent)
            _, pred = torch.max(pred, 1)
            if pred_all is None:
                pred_all = pred
                target_all = target
            else:
                pred_all = torch.cat([pred_all, pred], dim=0)
                target_all = torch.cat([target_all, target], dim=0)


    pred_all = pred_all.cpu().numpy()
    pred_all = pred_all.astype(int)
    target_all = target_all.cpu().numpy()
    target_all = target_all.astype(int)

    rand_index = metrics.adjusted_rand_score(target_all, pred_all)
    nmi = metrics.normalized_mutual_info_score(target_all, pred_all)
    homo = metrics.homogeneity_score(target_all, pred_all)
    comp = metrics.completeness_score(target_all, pred_all)
    fm_score = metrics.fowlkes_mallows_score(target_all, pred_all)

    print('Rand Index: {:.4f} | NMI: {:.4f} | Homo | {:.4f} : Comp: {:.4f} | FM" {:.4f}'.format(rand_index, nmi, homo,
                                                                                           comp, fm_score))


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

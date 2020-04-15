"""
Utility class to save the images
"""
import os
import os.path as osp
import torchvision.utils as vutils


def save_image(tensor_minibatch, image_path, file_num, mode):
    """
    saves the image batch passed
    :param tensor_minibatch:  minibatch of images
    :param image_path:  image path
    :param file_name: file name
    :return: None
    """
    if not osp.exists(image_path):
        os.makedirs(image_path)

    num_imgs = tensor_minibatch.shape[0]
    for i in range(num_imgs):
        file = mode+"_" + str(file_num+i)+".png"
        vutils.save_image(tensor=tensor_minibatch[i], filename=image_path+file, normalize=True)

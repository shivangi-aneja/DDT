import os
import cv2
import imutils
import dlib
import numpy as np
from imutils import face_utils
from os import makedirs
import random


dlib_model_path = '/home/shivangi/Desktop/Projects/pretrained_models/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(dlib_model_path)


def create_lip_dataset(dir, input_path, output_path):
    # For images
    for image in sorted(os.listdir(input_path+dir+'/')):
        if image.endswith(".png"):
            img = cv2.imread(input_path + dir + '/' + image)
            resize_img = imutils.resize(img, width=500)
            rects = detector(resize_img, 1)

            if len(rects) == 0:
                x, y, w, h = 176, 305, 157, 54
            else:
                shape = predictor(resize_img, rects[0])
                shape = face_utils.shape_to_np(shape)
                name = 'mouth'
                (i, j) = face_utils.FACIAL_LANDMARKS_IDXS[name]
                x, y, w, h = cv2.boundingRect(np.array([shape[i:j]]))
            y_min = max(0, y - 30)
            x_min = max(0, x - 20)
            mouth_img = resize_img[y_min:y + h + 30, x_min:x + w + 20]
            imutils.resize(mouth_img, width=250, inter=cv2.INTER_CUBIC)
            mouth_img = cv2.resize(mouth_img, (250, 150))
            cv2.imwrite(output_dir + dir + '/'+ image.split(".")[0] + '.png', mouth_img)


def create_subset_dataset(input_path, output_path):
    # For images
    data_list = [os.path.join(input_path, f) for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f))]
    data_list = random.sample(data_list, 20000)
    for image in sorted(data_list):
        if image.endswith(".png"):
            print(image)
            img = cv2.imread(image)
            cv2.imwrite(output_path + image.split("/")[-1], img)


def create_fourier_transform_dataset(input_path, output_path):
    epsilon = 1e-8
    # For images
    for image in sorted(os.listdir(input_path)):
        if image.endswith(".png"):
            print(image)
            img = cv2.imread(input_path + image, cv2.IMREAD_GRAYSCALE)
            # Calculate FFT
            f = np.fft.fft2(img)
            fshift = np.fft.fftshift(f)
            fshift += epsilon
            magnitude_spectrum = 20 * np.log(np.abs(fshift))
            cv2.imwrite(output_path + image.split(".")[0] + '.png', magnitude_spectrum)


def create_gaussian_blur(input_path, output_path):
    epsilon = 1e-8
    # For images
    for image in sorted(os.listdir(input_path)):
        if image.endswith(".png"):
            print(image)
            img = cv2.imread(input_path + image)
            # Calculate Gaussian Blur
            blur = cv2.GaussianBlur(img, (5, 5), 0)
            cv2.imwrite(output_path + image.split(".")[0] + '.png', blur)


if __name__ == "__main__":
    manipulation = 'f2f'
    input_dir = os.path.join('/media/newhd/FaceForensics_video_crops/c23/train/' + manipulation + '/')
    output_dir = os.path.join('/media/newhd/lip_crops/c23/train/' + manipulation + '/')
    dir_list = [x[0] for x in os.walk(input_dir)][1:]
    print(manipulation)

    for dir in dir_list:
        dir = dir.split('/')[-1]
        if not os.path.isdir(output_dir + dir + '/'):
            makedirs(output_dir + dir + '/')
        create_lip_dataset(dir=dir, input_path=input_dir, output_path=output_dir)
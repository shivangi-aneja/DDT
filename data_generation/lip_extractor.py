from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import os

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
image_path = '/home/shivangi/Desktop/Projects/master_thesis/data/sample/face/nt/'
output_path = '/home/shivangi/Desktop/Projects/master_thesis/data/sample/lips/nt/'
model_path = '/home/shivangi/Desktop/Projects/pretrained_models/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(model_path)

for image in sorted(os.listdir(image_path)):
    if image.endswith(".png"):
        print(image)
        img = cv2.imread(image_path + image)
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
        cv2.imwrite(output_path + image.split(".")[0] + '.png', mouth_img)


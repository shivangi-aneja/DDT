from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2

image_path = '/home/shivangi/Desktop/Projects/master_thesis/data/ff_all/c23/sample/orig/135.png'

# load the input image, resize it, and convert it to grayscale
image = cv2.imread(image_path)
image = np.array(image, dtype='uint8')

print(np.min(image))
print(np.max(image))

image_row = (-1 * image[1:-3, 2:-2, :] + 3 * image[2:-2, 2:-2, :] - 3 * image[3:-1, 2:-2, :] + 1 * image[4:, 2:-2, :])
image_col = (-1 * image[2:-2, 1:-3, :] + 3 * image[2:-2, 2:-2, :] - 3 * image[2:-2, 3:-1, :] + 1 * image[2:-2, 4:, :])
cv2.imwrite("res_image_rows.png", image_row)
cv2.imwrite("res_image_col.png", image_col)

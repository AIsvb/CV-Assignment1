from Calibrate import calibrate
from Undistort import undistort
import glob
import cv2
import numpy as np

# 25 images for which the findChessboardCorners function is succesful
images_auto = glob.glob("C:/Users/svben/PycharmProjects/pythonProject/Schaakbord_fotos/WhatsApp*")

# 5 images for which the findChessboardCorners function is not succesful
images_manual = glob.glob("C:/Users/svben/PycharmProjects/pythonProject/Schaakbord_fotos/Manual*")

# test image
test_img = "C:/Users/svben/PycharmProjects/pythonProject/Schaakbord_fotos/test.jpeg"

# Run 1
#camera_matrix_1, distortion_coef_1, rot_vecs_1, trans_vecs_1 = calibrate(images_manual, (8,5), True)

#undistort(test_img, camera_matrix_1, distortion_coef_1, "C:/Users/svben/PycharmProjects/pythonProject/result_run1.png")

# Run 2
selection_1 = images_auto[0:10]
camera_matrix_2, distortion_coef_2, rot_vecs_2, trans_vecs_2 = calibrate(selection_1, (8,5))

undistort(test_img, camera_matrix_2, distortion_coef_2, "C:/Users/svben/PycharmProjects/pythonProject/result_run2.png")

print(camera_matrix_2)
# Run 3
#selection_2 = images_auto[0:5]
#camera_matrix_3, distortion_coef_3, rot_vecs_3, trans_vecs_3 = calibrate(selection_2, (8,5))
#undistort(test_img, camera_matrix_3, distortion_coef_3, "C:/Users/svben/PycharmProjects/pythonProject/result_run3.png")

'''
from SelectCornersInterface import SelectCornersInterface

# Collecting filenames
IF = SelectCornersInterface("C:/Users/svben/PycharmProjects/pythonProject/Schaakbord_fotos/test.jpeg", (8,5))
# Showing the selected corners on the image
IF.show_corners()
# Printing the coordinates of the corners
print(IF.corners)
'''
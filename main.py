from Calibrate import calibrate
from Undistort import undistort
import glob
import cv2
import numpy as np
from PoseEstimator import PoseEstimator

# 25 images for which the findChessboardCorners function is succesful
images_auto = glob.glob("C:/Users/svben/PycharmProjects/pythonProject/Schaakbord_fotos/WhatsApp*")

# 5 images for which the findChessboardCorners function is not succesful
images_manual = glob.glob("C:/Users/svben/PycharmProjects/pythonProject/Schaakbord_fotos/Manual*")

# test image
test_img = "C:/Users/svben/PycharmProjects/pythonProject/Schaakbord_fotos/test.jpeg"


# Run 2
selection = images_auto[0:10]
camera_matrix, distortion_coef, _, _ = calibrate(selection, (8,5))

#undistort(test_img, camera_matrix, distortion_coef, "C:/Users/svben/PycharmProjects/pythonProject/result_run2.png")

LPE = PoseEstimator(camera_matrix, distortion_coef)

LPE.estimate_pose(test_img, "C:/Users/svben/PycharmProjects/pythonProject/pose2.png")

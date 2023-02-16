from Calibrate import calibrate
from Undistort import undistort
import glob
import cv2
import numpy as np
from PoseEstimator import PoseEstimator
from CameraCalibrator import CameraCalibrator

# 25 images for which the findChessboardCorners function is succesful
images_auto = glob.glob("C:/Users/svben/PycharmProjects/pythonProject/Schaakbord_fotos/WhatsApp*")

# 5 images for which the findChessboardCorners function is not succesful
images_manual = glob.glob("C:/Users/svben/PycharmProjects/pythonProject/Schaakbord_fotos/Manual*")

# test image
test_img = "C:/Users/svben/PycharmProjects/pythonProject/Schaakbord_fotos/test.jpeg"

camera_matrix, distortion_coef, _, _ = calibrate(images_auto[0:10], (8,5))

undistort(test_img, camera_matrix, distortion_coef, "C:/Users/svben/PycharmProjects/pythonProject/result_run2.png")

LPE = PoseEstimator(camera_matrix, distortion_coef)

LPE.start_live_estimator()


# Computer Vision: Camera calibration assignment
# Creators: Gino Kuiper en Sander van Bennekom
# Date: 17-02-2023
# Recourses: docs.opencv.org

import cv2
import numpy as np
from PoseEstimator import PoseEstimator
from SelectCornersInterface import SelectCornersInterface


class Program:
    """ Program Class. When initialised with a set of chessboard images,
    the board shape (measured in number of squares) and the width of the chessboard squares (mm),
    the camera intrinsics and the distortion coefficients are computed. With this information
    the pose of the chessboard can then be estimated, either for a single image
    or in real-time using the webcam output."""
    def __init__(self, image_names, board_shape, square_size):
        self.fx = []
        self.fy = []
        self.cx = []
        self.cy = []

        self.board_shape = board_shape                          # The number of squares in width and height
        self.n_corners = (board_shape[0]+1, board_shape[1]+1)   # The number of corners to be detected
        self.square_size = square_size                          # THe size of the squares in mm

        # Computing and saving the camera matrix and distortion coefficients
        self.camera_matrix, self.distortion_coef, _, _ = self.calibrate(image_names)

        # Creating and object of the PoseEstimator class
        self.PE = PoseEstimator(self.camera_matrix, self.distortion_coef, board_shape, square_size)

    # Method that computes the camera intrinsics, extrinsics and distortion coefficients given a set of
    # chessboard images.
    def calibrate(self, image_names, show=False):
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        object_points = np.zeros(((self.n_corners[0]) * (self.n_corners[1]), 3), np.float32)
        object_points[:, :2] = np.mgrid[0:self.n_corners[0], 0:self.n_corners[1]].T.reshape(-1, 2)*self.square_size

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.

        for image in image_names:
            # Reading the image and converting it to gray scale
            img = cv2.imread(image)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Finding the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, self.n_corners, None)

            if ret:
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                # If the re-projection error of the image is larger than 1, do not include the image in the calibration
                if self.calculate_reprojection_error(image, corners2, object_points) >= 1:
                    continue

                objpoints.append(object_points)
                imgpoints.append(corners2)

                # Draw and display the corners
                if show:
                    cv2.drawChessboardCorners(img, self.n_corners, corners2, ret)
                    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
                    cv2.imshow('image', img)
                    cv2.waitKey(0)

            else:
                # If the OpenCV can not find the corners, create an instance of the interface for manual extraction
                IF = SelectCornersInterface(image, self.board_shape)

                # If the re-projection error of the image is larger than 1, do not include the image in the calibration
                if self.calculate_reprojection_error(image, IF.new_corners, object_points) >= 1:
                    continue

                objpoints.append(object_points)
                imgpoints.append(IF.new_corners)

                # Draw and display the corners
                if show:
                    cv2.drawChessboardCorners(img, self.n_corners, IF.new_corners, True)
                    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
                    cv2.imshow('image', img)
                    cv2.waitKey(0)

            cv2.destroyAllWindows()

        # Performing the calibration
        ret, camera_matrix, distortion_coef, rot_vecs, trans_vecs = cv2.calibrateCamera(objpoints, imgpoints,
                                                                                        gray.shape[::-1], None, None)
        return camera_matrix, distortion_coef, rot_vecs, trans_vecs

    # Method that starts the real-time estimation of the chessboard pose.
    def estimate_pose_live(self):
        self.PE.start_live_estimator()

    # Method that draws the world 3D axes and a cube on a given chessboard image to indicate its pose. The
    # resulting image is saved to the specified destination.
    def estimate_pose(self, path, destination):
        self.PE.estimate_pose(path, destination)

    # Method to calculate the re-projection error and to store intrinsic parameters in the lists.
    def calculate_reprojection_error(self, image, image_points, object_points):
        img = cv2.imread(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        imgpoints = []
        objpoints = []
        imgpoints.append(image_points)
        objpoints.append(object_points)

        # Performing the calibration
        ret, camera_matrix, distortion_coef, rot_vecs, trans_vecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rot_vecs[i], trans_vecs[i], camera_matrix,
                                              distortion_coef)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error

        reprojection_error = mean_error / len(objpoints)

        # Store the parameters if the image is included for calibration
        if reprojection_error < 1:
            self.fx.append(camera_matrix[0, 0])
            self.fy.append(camera_matrix[1, 1])
            self.cx.append(camera_matrix[0, 2])
            self.cy.append(camera_matrix[1, 2])

        return reprojection_error

    def print_results(self):
        print("Camera matrix:\n {}\n".format(self.camera_matrix))
        print("Standard deviation fx: {}".format(np.std(self.fx)))
        print("Standard deviation fy: {}".format(np.std(self.fy)))
        print("Standard deviation cx: {}".format(np.std(self.cx)))
        print("Standard deviation cy: {}".format(np.std(self.cy)))
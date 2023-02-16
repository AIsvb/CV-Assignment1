import numpy as np
import cv2
import glob


def calibrate(image_names, board_shape):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    object_points = np.zeros((board_shape[0]*board_shape[1], 3), np.float32)
    object_points[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    for image in image_names:
        img = cv2.imread(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (board_shape[0], board_shape[1]), None)

        if ret:
            objpoints.append(object_points)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (9, 6), corners2, ret)
            cv2.namedWindow("image", cv2.WINDOW_NORMAL)
            cv2.imshow('image', img)
            cv2.waitKey(0)
        cv2.destroyAllWindows()

    ret, camera_matrix, distortion_coef, rot_vecs, trans_vecs = cv2.calibrateCamera(objpoints, imgpoints,
                                                                                    gray.shape[::-1], None, None)

    return camera_matrix, distortion_coef




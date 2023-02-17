# Computer Vision: Camera calibration assignment
# Creators: Gino Kuiper en Sander van Bennekom
# Date: 17-02-2023
# Recourses: docs.opencv.org

import numpy as np
import cv2


class PoseEstimator:
    """ Instances of this class can be used to estimate the pose of a chessboard in an image
    given some specific camera matrix, distortion coefficients, chessboard shape (measured in squares)
    and square size (mm) """
    def __init__(self, camera_matrix, distortion_coef, board_shape, square_size):
        self.camera_matrix = camera_matrix
        self.distortion_coef = distortion_coef
        self.square_size = square_size                          # The number of squares in width and height
        self.n_corners = (board_shape[0]+1, board_shape[1]+1)   # The number of corners to be detected

    # Method for drawing the world 3D axes on a chessboard image.
    def draw_axes(self, img, corners, imgpts):
        corner = tuple(corners[0].ravel().astype(int))
        img = cv2.line(img, corner, tuple(imgpts[0].ravel().astype(int)), (255, 0, 0), 7)
        img = cv2.line(img, corner, tuple(imgpts[1].ravel().astype(int)), (0, 255, 0), 7)
        img = cv2.line(img, corner, tuple(imgpts[2].ravel().astype(int)), (0, 0, 255), 7)
        return img

    # Method for drawing a cube on a chessboard image.
    def draw_cube(self, img, imgpts):
        imgpts = np.int32(imgpts).reshape(-1, 2)
        # draw the bottom square
        img = cv2.drawContours(img, [imgpts[:4]], -1, (255, 255, 0), 3)
        # draw pillars
        for i, j in zip(range(4), range(4, 8)):
            img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255, 255, 0), 3)
        # draw top square
        img = cv2.drawContours(img, [imgpts[4:]], -1, (255, 255, 0), 3)
        return img

    # Method for drawing both a cube and the world 3D axes on a chessboard image.
    def draw_pose(self, img, camera_matrix, distortion_coef):
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((self.n_corners[0]*self.n_corners[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.n_corners[0], 0:self.n_corners[1]].T.reshape(-1, 2)*21

        # convert image to gray scale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, self.n_corners, None)

        if ret:
            # enhancing the location of the detected corners
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # find the rotation and translation vectors.
            ret, rvecs, tvecs = cv2.solvePnP(objp, corners2, camera_matrix, distortion_coef)

            # project 3D points to image plane
            axes = np.float32([[4, 0, 0], [0, 4, 0], [0, 0, -4]]).reshape(-1, 3)*self.square_size
            cube = np.float32([[0, 0, 0], [0, 2, 0], [2, 2, 0], [2, 0, 0],
                               [0, 0, -2], [0, 2, -2], [2, 2, -2], [2, 0, -2]])*self.square_size

            # project 3D points to image plane
            imgpts_a, _ = cv2.projectPoints(axes, rvecs, tvecs, camera_matrix, distortion_coef)
            imgpts_c, _ = cv2.projectPoints(cube, rvecs, tvecs, camera_matrix, distortion_coef)

            # drawing the axes and cube
            img = self.draw_axes(img, corners2, imgpts_a)
            img = self.draw_cube(img, imgpts_c)

        return img

    # Method that draws the world 3D axes and a cube on a given chessboard image to indicate its pose. The
    # resulting image is saved to the specified destination.
    def estimate_pose(self, path, destination):
        # reading the image
        img = cv2.imread(path)

        # drawing the axes and cube on the image
        img = self.draw_pose(img, self.camera_matrix, self.distortion_coef)

        # displaying the edited image
        cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        cv2.imshow("img", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # saving the edited image
        cv2.imwrite(destination, img)

    # Method that starts the webcam and displays the world 3D axes and cube whenever the chessboard is detected by
    # the OpenCV findChessboardCorners function.
    def start_live_estimator(self):
        # define a video capture object
        vid = cv2.VideoCapture(0)

        # create a window
        cv2.namedWindow("img", cv2.WINDOW_NORMAL)

        while True:
            # Capture the video frame by frame
            ret, frame = vid.read()

            # Display the resulting frame
            img = self.draw_pose(frame, self.camera_matrix, self.distortion_coef)
            cv2.imshow('img', img)

            # the 'q' button is set as the quitting button
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # After the loop release the cap object
        vid.release()

        # Destroy all the windows
        cv2.destroyAllWindows()

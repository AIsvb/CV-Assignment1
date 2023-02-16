import numpy as np
import cv2

class PoseEstimator:

    def __init__(self, camera_matrix, distortion_coef):
        self.camera_matrix = camera_matrix
        self.distortion_coef = distortion_coef

    def start_live_estimator(self):
        # define a video capture object
        vid = cv2.VideoCapture(0)

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
    def draw_axes(self, img, corners, imgpts):
        corner = tuple(corners[0].ravel().astype(int))
        img = cv2.line(img, corner, tuple(imgpts[0].ravel().astype(int)), (255, 0, 0), 7)
        img = cv2.line(img, corner, tuple(imgpts[1].ravel().astype(int)), (0, 255, 0), 7)
        img = cv2.line(img, corner, tuple(imgpts[2].ravel().astype(int)), (0, 0, 255), 7)
        return img

    def draw_cube(self, img, corners, imgpts):
        imgpts = np.int32(imgpts).reshape(-1, 2)
        # draw ground floor in green
        img = cv2.drawContours(img, [imgpts[:4]], -1, (255, 255, 0), 3)
        # draw pillars in blue color
        for i, j in zip(range(4), range(4, 8)):
            img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255, 255, 0), 3)
        # draw top layer in red color
        img = cv2.drawContours(img, [imgpts[4:]], -1, (255, 255, 0), 3)
        return img

    def draw_pose(self, img, camera_matrix, distortion_coef):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((6 * 9, 3), np.float32)
        objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        if ret == True:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # Find the rotation and translation vectors.
            ret, rvecs, tvecs = cv2.solvePnP(objp, corners2, camera_matrix, distortion_coef)

            # project 3D points to image plane
            axes = np.float32([[4, 0, 0], [0, 4, 0], [0, 0, -4]]).reshape(-1, 3)
            cube = np.float32([[0, 0, 0], [0, 2, 0], [2, 2, 0], [2, 0, 0],
                               [0, 0, -2], [0, 2, -2], [2, 2, -2], [2, 0, -2]])

            imgpts_a, _ = cv2.projectPoints(axes, rvecs, tvecs, camera_matrix, distortion_coef)
            imgpts_c, _ = cv2.projectPoints(cube, rvecs, tvecs, camera_matrix, distortion_coef)

            img = self.draw_axes(img, corners2, imgpts_a)
            img = self.draw_cube(img, corners2, imgpts_c)

        return img

    def estimate_pose(self, path, destination):
        img = cv2.imread(path)
        img = self.draw_pose(img, self.camera_matrix, self.distortion_coef)

        cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        cv2.imshow("img", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        cv2.imwrite(destination, img)
import cv2
import numpy as np


class SelectCornersInterface:
    def __init__(self, path):
        self.path = path
        self.img = cv2.imread(path)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.click = 0
        self.corners = np.empty((4,2), dtype=int)
        self.show_img()

    def show_img(self):
        # Printing the instructions for the user on the image
        text = "Select corners"
        t_size = cv2.getTextSize(text, self.font, 2, 4)
        text_x = int((self.img.shape[1] - t_size[0][0]) / 2)
        text_y= int((self.img.shape[0] + t_size[0][1]) / 2)

        cv2.putText(self.img, text, (text_x, text_y), self.font, 2,
                    (255, 0, 255), 4)
        # displaying the image
        cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        cv2.imshow('image', self.img)

        # setting mouse handler for the image
        # and calling the click_event() function
        cv2.setMouseCallback('image', self.click_event)

        # wait for a key to be pressed to exit
        cv2.waitKey(0)

        # close the window
        cv2.destroyAllWindows()

    def click_event(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Increasing the number of clicks by 1 after a corner point is selected
            self.click += 1

            # Saving the coordinates of the point
            corner = np.array([x, y])
            self.corners[self.click - 1] = corner

            # Displaying the points on the image window
            cv2.circle(self.img, (x, y), 10, (0, 0, 255), -1)
            cv2.imshow('image', self.img)

            # Closing the interface after four points have been selected
            if self.click == 4:
                cv2.waitKey(1000)
                cv2.destroyAllWindows()

    def show_corners(self):
        self.img = cv2.imread(self.path)
        # Writing the coordinates of each corner on the image
        for corner in self.corners:
            cv2.circle(self.img, (corner[0], corner[1]), 10, (0, 0, 255), -1)
            cv2.putText(self.img, str(corner[0]) + ',' +
                        str(corner[1]), (corner[0], corner[1]), self.font,
                        1, (255, 0, 0), 2)

        # Displaying the coordinates on the image window
        cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        cv2.imshow("image", self.img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

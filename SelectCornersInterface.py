import cv2
import numpy as np
from scipy.interpolate import interp1d

class SelectCornersInterface:
    def __init__(self, path):
        self.path = path
        self.img = cv2.imread(path)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.click = 0
        self.corners = np.empty((4, 2), dtype=float)
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
                self.interpolate_points(self.corners, 10, 7)
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

    # Tijdelijke methode om de interpolated punten te laten zien
    def show_corners2(self):
        self.img = cv2.imread(self.path)
        # Writing the coordinates of each corner on the image
        corners2 = self.interpolate_points(self.corners, 10, 7)
        cv2.(self.img, [10, 7], corners2)
        # Displaying the coordinates on the image window
        cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        cv2.imshow("image", self.img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    # Method to linearly interpolate the inner corners
    def interpolate_points(self, corners, height, width):
        # Array for calculated
        calculated_corners = []

        # Sort corners
        sorted_corners = self.sort_corners(corners)
        topleft = sorted_corners[0]
        topright = sorted_corners[1]
        bottomleft = sorted_corners[2]
        bottomright = sorted_corners[3]

        # Lists of top and bottom (x,y) tuples, starts off empty
        top_points = []
        bottom_points = []

        # Interpolate y-coordinates between top-left and top-right, fill top_points
        step_size = (topright[0] - topleft[0]) / width
        y_interpolate = interp1d([topleft[0], topright[0]], [topleft[1], topright[1]])
        for i in range(width + 1):
            top_points.append((topleft[0] + i * step_size, float(y_interpolate(topleft[0] + i * step_size))))

        # Interpolate y-coordinates between bottom-left and bottom-right, fill bottom_points
        step_size = (bottomright[0] - bottomleft[0]) / width
        y_interpolate = interp1d([bottomleft[0], bottomright[0]], [bottomleft[1], bottomright[1]])
        for i in range(width + 1):
            bottom_points.append((bottomleft[0] + i * step_size, float(y_interpolate(bottomleft[0] + i * step_size))))

        # Interpolate points between all top and bottom points
        for i in range(len(top_points)):
            x_interpolate = interp1d([top_points[i][1], bottom_points[i][1]], [top_points[i][0], bottom_points[i][0]])

            # Determine step size per row based on top and bottom y-coordinate
            step_size = (bottom_points[i][1] - top_points[i][1]) / height

            # Fill calculated_corners
            for j in range(height):
                calculated_corners.append([float(x_interpolate(top_points[i][1] + (j * step_size))),
                                                top_points[i][1] + (j * step_size)])
            calculated_corners.append([bottom_points[i][0], bottom_points[i][1]])

        self.new_corners = np.array(calculated_corners).reshape((88, 1, 2))
        return(self.new_corners)

    # Method to sort the corner top-left, top-right, bottom-left, bottom-right
    def sort_corners(self, corners):
        new_corners = sorted(((corners[0][0], corners[0][1]), (corners[1][0], corners[1][1]), (corners[2][0], corners[2][1]),
                   (corners[3][0], corners[3][1])), key=lambda x: x[1])

        if new_corners[1][0] < new_corners[0][0]:
            temp = np.copy(new_corners[0])
            new_corners[0] = new_corners[1]
            new_corners[1] = temp

        if new_corners[3][0] < new_corners[2][0]:
            temp = np.copy(new_corners[2])
            new_corners[2] = new_corners[3]
            new_corners[3] = temp

        return new_corners






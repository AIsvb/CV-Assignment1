import cv2 as cv
import numpy as np
from SelectCornersInterface import SelectCornersInterface

# Selecting the corners manually
IF = SelectCornersInterface("/Users/macbook/Desktop/testfoto.jpg")

# Showing the selected corners on the image
IF.show_corners2()

# Printing the coordinates of the corners
print(IF.corners)
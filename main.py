import cv2 as cv
import numpy as np
from SelectCornersInterface import SelectCornersInterface

# Selecting the corners manually
IF = SelectCornersInterface("C:/Users/svben/Downloads/Schaakbord_1.jpeg")

# Showing the selected corners on the image
IF.show_corners()

# Printing the coordinates of the corners
print(IF.corners)

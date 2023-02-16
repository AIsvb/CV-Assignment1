import cv2 as cv
import numpy as np
from SelectCornersInterface import SelectCornersInterface

# Selecting the corners manually
# IF = SelectCornersInterface("C:/Users/svben/PycharmProjects/pythonProject/Schaakbord_fotos/Manual_1.jpeg")
# "C:/Users/svben/PycharmProjects/pythonProject/Schaakbord_fotos/Schaakbord_1.jpeg"
IF = SelectCornersInterface("/Users/macbook/Desktop/testfoto.jpg")

# Showing the selected corners on the image
IF.show_corners2()

# Printing the coordinates of the corners
print(IF.corners)


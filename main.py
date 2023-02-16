from Calibrate import calibrate
import glob

# 25 images for which the findChessboardCorners function is succesful
images_auto = glob.glob("C:/Users/svben/PycharmProjects/pythonProject/Schaakbord_fotos/WhatsApp*")

# 5 images for which the findChessboardCorners function is not succesful
images_manual = glob.glob("C:/Users/svben/PycharmProjects/pythonProject/Schaakbord_fotos/Manual*")

# test image
test_img = glob.glob("C:/Users/svben/PycharmProjects/pythonProject/Schaakbord_fotos/test.jpeg")

# Run 1
camera_matrix_1, distortion_coef_1, rot_vecs_1, trans_vecs_1 = calibrate(images_manual+images_manual)

# Run 2
selection_1 = images_auto[0:10]
camera_matrix_2, distortion_coef_2, rot_vecs_2, trans_vecs_2 = calibrate(selection_1)

# Run 3
selection_2 = images_auto[0:5]
camera_matrix_3, distortion_coef_3, rot_vecs_3, trans_vecs_3 = calibrate(selection_2)

'''
from SelectCornersInterface import SelectCornersInterface

# Collecting filenames
IF = SelectCornersInterface("C:/Users/svben/PycharmProjects/pythonProject/Schaakbord_fotos/test.jpeg", (8,5))
# Showing the selected corners on the image
IF.show_corners()
# Printing the coordinates of the corners
print(IF.corners)
'''
# Computer Vision: Camera calibration assignment
# Creators: Gino Kuiper en Sander van Bennekom
# Date: 17-02-2023
# Recourses: docs.opencv.org

import glob
from Program import Program

# Main script. This file must be run to do the three different calibration runs. Depending on the location
# of the chessboard images, the file paths must be changed.

if __name__ == "__main__":
    # Collecting the filenames

    # 25 images for which the findChessboardCorners function is succesful
    images_auto = glob.glob("C:/Users/svben/PycharmProjects/pythonProject/Schaakbord_fotos/WhatsApp*")
    #images_auto = glob.glob("/Users/macbook/Desktop/Schaakbord_fotos/WhatsApp*")

    # 5 images for which the findChessboardCorners function is not succesful
    images_manual = glob.glob("C:/Users/svben/PycharmProjects/pythonProject/Schaakbord_fotos/Manual*")
    #images_manual = glob.glob("/Users/macbook/Desktop/Schaakbord_fotos/Manual*")

    # test image
    test_img = "C:/Users/svben/PycharmProjects/pythonProject/Schaakbord_fotos/test.jpeg"
    #test_img = "/Users/macbook/Desktop/Schaakbord_fotos/test.jpeg"


    # Performing the three calibration runs
    Run_1 = Program(images_auto + images_manual, (8,5), 21)
    Run_2 = Program(images_auto[0:10], (8,5), 21)
    Run_3 = Program(images_auto[0:5], (8,5), 21)

    # Using the obtained camera matrices and distortion coefficients to estimate the pose of the test image
    Run_1.estimate_pose(test_img, "C:/Users/svben/PycharmProjects/pythonProject/result_run1.png")
    #Run_1.estimate_pose(test_img, "/Users/macbook/Desktop/result_run1.png")
    Run_2.estimate_pose(test_img, "C:/Users/svben/PycharmProjects/pythonProject/result_run2.png")
    #Run_2.estimate_pose(test_img, "/Users/macbook/Desktop/result_run2.png")
    Run_3.estimate_pose(test_img, "C:/Users/svben/PycharmProjects/pythonProject/result_run3.png")
    #Run_3.estimate_pose(test_img, "/Users/macbook/Desktop/result_run3.png")

    # Printing the camera matrices and standard deviations of all runs
    print(Run_1.print_results())
    print(Run_2.print_results())
    print(Run_3.print_results())



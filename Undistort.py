import cv2


def undistort(image, camera_matrix, distortion_coef, path):
    # Reading the image that must be corrected
    img = cv2.imread(image)

    # Finding the new camera matrix
    height, width = img.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coef,
                                                           (width, height), 1, (width, height))

    # Finding a mapping function from the distorted image to the undistorted image
    mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, distortion_coef, None, new_camera_matrix,
                                             (width, height), 5)

    # Using the remap function to correct the image
    dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

    # Cropping the image and saving it
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv2.imwrite(path + "/calibration_result.png", dst)

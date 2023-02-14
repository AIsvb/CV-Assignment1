import numpy as np
from scipy.interpolate import interp1d

# Test values
xcorners = [10, 20, 10, 20]
ycorners = [10, 10, 20, 20]
def calculate_corners(xcorners, ycorners, height, width):
    # Array to store inner corner points
    top_points = []
    bottom_points = []
    calculated_points = []

    boardwidth = xcorners[1] - xcorners[0]
    boardheight = ycorners[2] - ycorners[0]

    # interpolate points between top-left and top-right
    X = [xcorners[0], xcorners[1]]
    Y = [ycorners[0], ycorners[1]]
    y_interpolate = interp1d(X, Y)
    for i in range(width + 1):
        top_points.append((xcorners[0] + (i * boardwidth / width), float(y_interpolate(xcorners[0]
                                                                                       + (i * boardwidth / width)))))

    # interpolate points between bottom-left and bottom-right
    X = [xcorners[2], xcorners[3]]
    Y = [ycorners[3], ycorners[3]]
    y_interpolate = interp1d(X, Y)
    for i in range(width + 1):
        bottom_points.append((xcorners[2] + (i * boardwidth / width), float(y_interpolate(xcorners[2]
                                                                                       + (i * boardwidth / width)))))

    # interpolate points between all top and bottom points
    for i in range(len(top_points)):
        X = [top_points[i][0], bottom_points[i][0]]
        Y = [top_points[i][1], bottom_points[i][1]]
        x_interpolate = interp1d(Y, X)
        for j in range(height + 1):
            calculated_points.append((float(x_interpolate(top_points[i][1] + (j * boardheight / height))), top_points[i][1] + (j * boardheight / height)))



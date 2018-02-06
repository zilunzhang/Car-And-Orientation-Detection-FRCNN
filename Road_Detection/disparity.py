import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

os.chdir('../../code/data')


def get_disparity(left_path, right_path):

    left = cv2.imread(left_path, 0)
    right = cv2.imread(right_path, 0)

    # Initial squared differences
    h, w = left.shape  # assume that both images are same size
    sd = np.empty((h, w), np.uint8)

    # SSD support window (kernel)
    win_ssd = np.empty((h, w), np.uint16)

    # Depth (or disparity) map
    disparity = np.empty((h, w), np.uint8)

    # Minimum ssd difference between both images
    min_ssd = np.empty((h, w), np.uint16)
    min_ssd.fill(65535)

    max_offset = 20
    offset_adjust = 255 / max_offset

    # Create ranges now instead of per loop
    y_range = range(h)
    x_range = range(w)
    x_range_ssd = range(w)

    # u and v support window
    window_range = range(-3, 4)  # 6x6
    for offset in range(max_offset):
        # Create initial image of squared differences between left and right image at the current offset
        for y in y_range:
            for x in x_range_ssd:
                if x - offset > 0:
                    diff = abs(int(left[y, x]) - int(right[y, x - offset]))
                    sd[y, x] = diff

        # Sum the squared differences over a support window at this offset
        for y in y_range:
            for x in x_range:
                sum_sd = 0
                for i in window_range:
                    for j in window_range:
                        if (-1 < y + i < h) and (-1 < x + j < w):
                            sum_sd += sd[y + i, x + j]

                # Store the sum in the window SSD image
                win_ssd[y, x] = sum_sd

        # Update the min ssd diff image with this new data
        for y in y_range:
            for x in x_range:
                # Is this new windowed SSD pixel a better match?
                if win_ssd[y, x] < min_ssd[y, x]:
                    # If so, store it and add to the disparity map
                    min_ssd[y, x] = win_ssd[y, x]
                    disparity[y, x] = offset * offset_adjust

    # cv2.imwrite('DPDisparityMatrix.jpg', disparityMat)
    # cv2.imshow('dispMat', disparity)
    plt.imshow(disparity), plt.show()
    return disparity

def cv2disparity(left_path, right_path):
    imgL = cv2.imread(left_path, 0)
    imgR = cv2.imread(right_path, 0)

    # stereo = cv2.StereoBM_create(numDisparities=48, blockSize=31)
    stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=64, blockSize=9, uniquenessRatio=2,
                                   speckleWindowSize=50, speckleRange=2)
    disparity = stereo.compute(imgL, imgR)
    # plt.imshow(disparity), plt.show()
    return disparity


def read_camlib(path):
    P0, P1, P2, P3 = [], [], [], []
    with open(path, "r") as file:
        lines = file.readlines()
        for line in lines:
            words = line.split()
            if words[0] == "P0:":
                P0 = words[1:]
            elif words[0] == "P1:":
                P1 = words[1:]
            elif words[0] == "P2:":
                P2 = words[1:]
            elif words[0] == "P3:":
                P3 = words[1:]
                # elif words[0] == "R0_rect:":
                #     R0 = words[1:]
                # elif words[0] == "Tr_velo_to_cam:":
                #     velo_to_cam = words[1:]
                # elif words[0] == "Tr_imu_to_velo:":
                #     imu_to_velo = words[1:]
                # elif words[0] == "Tr_cam_to_road:":
                #     cam_to_road = words[1:]
    return P0, P1, P2, P3


if __name__ == '__main__':
    # get_disparity("left.png", "right.png")
    disparity = cv2disparity("data_road/training/image_2/um_000010.png",
                             "data_road_right/training/image_3/um_000010.png")
    # plt.imshow(disparity), plt.colorbar(), plt.show()
    P0, P1, P2, P3 = read_camlib("data_road/training/calib/um_000010.txt")

    gt70 = len(np.argwhere(disparity < 70))
    total = disparity.shape[0] * disparity.shape[1]

    # print(gt70)
    # print(gt70 / total)
    # print(np.max(disparity))
    # print(np.min(disparity))
    disparity[disparity < 70] = 70

    f = float(P1[0]) / 10
    baseline = -float(P1[3])/f * 100
    depth = f * baseline * (1 / disparity * 10) / 100 - 10
    depth[depth < 10] = 0
    plt.imshow(depth), plt.colorbar(), plt.show()

    [M, N] = depth.shape

    x_2d = np.tile(np.asarray(list(range(0, N))), (M, 1))
    y_2d = np.tile(np.flip(np.asarray(list(range(0, M))), axis=0), (N, 1)).T

    x_3d = (x_2d - x_2d.shape[1] / 2) * depth / f
    y_3d = (y_2d - y_2d.shape[0] / 2) * depth / f + 10
    y_3d[y_3d < 0] = 0
    y_3d[y_3d > 50] = 50
    # plt.imshow(y_3d), plt.colorbar(), plt.show()
    plt.imshow(x_3d), plt.colorbar(), plt.show()

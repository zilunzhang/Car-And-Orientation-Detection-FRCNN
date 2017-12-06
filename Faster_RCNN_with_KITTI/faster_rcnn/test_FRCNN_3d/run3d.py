import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os, sys, cv2
import argparse
import os.path as osp
import glob
from tensorflow.core.protobuf import saver_pb2

from math import ceil
from math import floor
from run_kitti import *
from PIL import Image
from IoU import bb_intersection_over_union

this_dir = osp.dirname(__file__)
print(this_dir)

from faster_rcnn.lib.networks.factory import get_network
from faster_rcnn.lib.fast_rcnn.config import cfg
from faster_rcnn.lib.fast_rcnn.test import im_detect
from faster_rcnn.lib.fast_rcnn.nms_wrapper import nms
from faster_rcnn.lib.utils.timer import Timer
import re
CLASSES = ('background', 'Pedestrian', 'Car', 'Cyclist')
font = cv2.FONT_HERSHEY_DUPLEX



left_path = "../data/KITTI/3D/sample/left/006081.png"
right_path = "../data/KITTI/3D/sample/right/006081.png"
calib_path = "../data/KITTI/3D/sample/calib/006081.txt"


image_dir = "image/"
label_dir = "label/"


def run_single_image(filename):
    img_ext = ".png"
    label_ext = ".txt"

    image_path = image_dir + filename + img_ext
    label_path = label_dir + filename + label_ext

    img = cv2.imread(image_path)
    label_file = open(label_path, "r")
    lines = label_file.readlines()
    run_model(image_path)
    return image_path




def get3d(left, right, calib):
    imgL = cv2.imread(left, 0)
    imgR = cv2.imread(right, 0)

    stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=64, blockSize=9, uniquenessRatio=2,
                                   speckleWindowSize=50, speckleRange=2)
    disparity = stereo.compute(imgL, imgR)

    # gt70 = len(np.argwhere(disparity < 70))
    # total = disparity.shape[0] * disparity.shape[1]

    # disparity[disparity < 70] = 70

    P1 = readCalib(calib)

    f = float(P1[0]) / 10
    baseline = -float(P1[3]) / f * 100
    depth = f * baseline * (1 / disparity * 10) / 300

    # depth[depth < -3] = -3
    # plt.imshow(depth), plt.colorbar(), plt.show()

    [M, N] = depth.shape

    x_2d = np.tile(np.asarray(list(range(0, N))), (M, 1))
    y_2d = np.tile(np.flip(np.asarray(list(range(0, M))), axis=0), (N, 1)).T

    x_3d = (x_2d - x_2d.shape[1] / 2) * depth / f
    y_3d = (y_2d - y_2d.shape[0] / 2) * depth / f + 10
    # y_3d[y_3d < 0] = 0.001
    # y_3d[y_3d > 50] = 50

    return concate3d(x_3d, y_3d, depth)


def readCalib(calib):
    P1 = []
    with open(calib, "r") as file:
        lines = file.readlines()
        for line in lines:
            if words[0] == "P1:":
                P1 = words[1:]
                return P1
    return P1


def concate3d(x, y, z):
    M, N = z.shape
    image_3d = np.zeros((M, N, 3))

    image_3d[:, :, 0] = x
    image_3d[:, :, 1] = y
    image_3d[:, :, 2] = z

    return image_3d



def get_seg(box_matrix, matrix_3d, img):

    threshold = 3
    seg_matrix = np.zeros((box_matrix.shape[0], 12))

    x_list = np.arange(0, matrix_3d.shape[1], 1)
    y_list = np.arange(0, matrix_3d.shape[0], 1)

    [M, N] = img.shape[:2]

    x_matrix = np.tile(np.asarray(list(range(M))), (N, 1)).T
    y_matrix = np.tile(np.asarray(list(range(N))), (M, 1))
    X_matrix = matrix_3d[:, :, 0]
    Y_matrix = matrix_3d[:, :, 1]
    depth_matrix = matrix_3d[:, :, 2]

    # container for mass center
    mass_centers = np.zeros((box_matrix.shape[0], 3))
    for i in range(box_matrix.shape[0]):
        x_left = np.int(box_matrix[i][0])
        y_top = np.int(box_matrix[i][1])
        x_right = np.int(box_matrix[i][2])
        y_bottom = np.int(box_matrix[i][3])

        # pts = np.array([[x_left, y_top],
        #                 [x_right, y_top],
        #                 [x_right, y_bottom],
        #                 [x_left, y_bottom]], np.int32)
        # pts = pts.reshape((-1, 1, 2))
        #
        # if type == "car":
        #     img = cv2.polylines(img, [pts], True, (0, 0, 255))
        #     cv2.putText(img, 'Car', (x_left, y_top - 5), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        # if type == "person":
        #     img = cv2.polylines(img, [pts], True, (255, 0, 0))
        #     cv2.putText(img, 'Person', (x_left, y_top - 5), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        # if type == "bicycle":
        #     img = cv2.polylines(img, [pts], True, (0, 255, 0))
        #     cv2.putText(img, 'Bicycle', (x_left, y_top - 5), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        Z = depth_matrix[y_top:y_bottom, x_left:x_right]
        X = X_matrix[y_top:y_bottom, x_left:x_right]
        Y = Y_matrix[y_top:y_bottom, x_left:x_right]
        x = x_matrix[y_top:y_bottom, x_left:x_right]
        y = y_matrix[y_top:y_bottom, x_left:x_right]

        x_median = np.median(X)
        y_median = np.median(Y)
        z_median = np.median(Z)

        # how_far = np.power(np.power(x_median,2)+np.power(y_median,2)+np.power(z_median,2), 1/2)

        # dis_list.append(how_far)

        mass_centers[i,0] = x_median
        mass_centers[i,1] = y_median
        mass_centers[i,2] = z_median

        max_x, max_y, max_z = -9999, -9999, -9999
        min_x, min_y, min_z = 9999, 9999, 9999
        for m in range(X.shape[0]):
            for n in range(X.shape[1]):
                distance = np.power((X[m, n] - x_median), 2) + \
                           np.power((Y[m, n] - y_median), 2) + \
                           np.power((Z[m, n] - z_median), 2)
                if distance < threshold:
                    img[x[m, n], y[m, n]] = (0, 255, 0)
                    if X[m, n] > max_x:
                        max_x = X[m, n]
                        seg_matrix[i, 0] = x[m, n]
                        seg_matrix[i, 1] = y[m, n]
                    if X[m, n] < min_x:
                        min_x = X[m, n]
                        seg_matrix[i, 2] = x[m, n]
                        seg_matrix[i, 3] = y[m, n]
                    if Y[m, n] > max_y:
                        max_y = Y[m, n]
                        seg_matrix[i, 4] = x[m, n]
                        seg_matrix[i, 5] = y[m, n]
                    if Y[m, n] < min_y:
                        min_y = Y[m, n]
                        seg_matrix[i, 6] = x[m, n]
                        seg_matrix[i, 7] = y[m, n]
                    if Z[m, n] > max_z:
                        max_z = Z[m, n]
                        seg_matrix[i, 8] = x[m, n]
                        seg_matrix[i, 9] = y[m, n]
                    if Z[m, n] < min_z:
                        min_z = Z[m, n]
                        seg_matrix[i, 10] = x[m, n]
                        seg_matrix[i, 11] = y[m, n]
        print('!!!!!!!', seg_matrix)
        # seg_matrix[i, 0], seg_matrix[i, 1], seg_matrix[i, 2] = min_x, min_y, min_z
        # seg_matrix[i, 3], seg_matrix[i, 4], seg_matrix[i, 5] = max_x, max_y, max_z

    return seg_matrix


def draw_box(img, seg_matrix):
    for box_num in range(seg_matrix.shape[0]):
        seg_val = seg_matrix[box_num, :]
        print(seg_val)
        for i in range(6):
            print(seg_val[i])
            img[int(seg_val[i]), int(seg_val[i+1])] = (255, 0, 0)
        # plt.imshow(img), plt.show()

        # top layer plane:
        t_center = [seg_val[4], seg_val[5]]
        x_ver_len = seg_val[2] - seg_val[0]
        x_hor_len = seg_val[3] - seg_val[1]
        z_ver_len = seg_val[10] - seg_val[8]
        z_hor_len = seg_val[11] - seg_val[9]
        # four top layer edge centers
        tl_center = [t_center[0] - x_ver_len / 2, t_center[1] - x_hor_len / 2]
        tr_center = [t_center[0] + x_ver_len / 2, t_center[1] + x_hor_len / 2]
        # tb_center = [t_center[0] - z_ver_len / 2, t_center[1] - z_hor_len / 2]
        tt_center = [t_center[0] + z_ver_len / 2, t_center[1] + z_hor_len / 2]
        # four top layer corners
        ttl = [tl_center[0] + tt_center[0] - t_center[0], tl_center[1] + tt_center[1] - t_center[1]]
        ttr = [tr_center[0] + tt_center[0] - t_center[0], tr_center[1] + tt_center[1] - t_center[1]]
        tbl = [tl_center[0] - tt_center[0] + t_center[0], tl_center[1] - tt_center[1] + t_center[1]]
        tbr = [tr_center[0] - tt_center[0] + t_center[0], tr_center[1] - tt_center[1] + t_center[1]]

        # bottom layer plane:
        b_center = [seg_val[6], seg_val[7]]
        # four bottom layer edge centers
        bl_center = [b_center[0] - x_ver_len / 2, b_center[1] - x_hor_len / 2]
        br_center = [b_center[0] + x_ver_len / 2, b_center[1] + x_hor_len / 2]
        # bb_center = [b_center[0] - z_ver_len / 2, b_center[1] - z_hor_len / 2]
        bt_center = [b_center[0] + z_ver_len / 2, b_center[1] + z_hor_len / 2]
        # four bottom layer corners
        btl = [bl_center[0] + bt_center[0] - b_center[0], bl_center[1] + bt_center[1] - b_center[1]]
        btr = [br_center[0] + bt_center[0] - b_center[0], br_center[1] + bt_center[1] - b_center[1]]
        bbl = [bl_center[0] - bt_center[0] + b_center[0], bl_center[1] - bt_center[1] + b_center[1]]
        bbr = [br_center[0] - bt_center[0] + b_center[0], br_center[1] - bt_center[1] + b_center[1]]

        # draw 12 lines

        # pts = np.array([[204, 250], [204, 78], [390, 407], [390, 579]], np.int32)
        # pts = pts.reshape((-1, 1, 2))
        # cv2.polylines(img, [pts], True, (0, 255, 255))

        # b1 = np.array([[ttl[1], ttl[0]], [ttr[1], ttr[0]], [tbr[1], tbr[0]], [tbl[1], tbl[0]]], np.int32)
        b1 = np.array([ttl, ttr, tbr, tbl], np.int32)
        # b1 = b1.reshape((-1, 1, 2))
        cv2.polylines(img, [b1], True, (0, 255, 0))
        # b2 = np.array([[btl[1], btl[0]], [btr[1], btr[0]], [bbr[1], bbr[0]], [bbl[1], bbl[0]]], np.int32)
        b2 = np.array([btl, btr, bbr, bbl], np.int32)
        b2 = b2.reshape((-1, 1, 2))
        cv2.polylines(img, [b2], True, (0, 255, 255))
        # b3 = np.array([[ttl[1], ttl[0]], [ttr[1], ttr[0]], [btr[1], btr[0]], [btl[1], btl[0]]], np.int32)
        b3 = np.array([ttl, ttr, btr, btl], np.int32)
        b3 = b3.reshape((-1, 1, 2))
        cv2.polylines(img, [b3], True, (255, 0, 0))
        # b4 = np.array([[tbl[1], tbl[0]], [tbr[1], tbr[0]], [bbr[1], bbr[0]], [bbl[1], bbl[0]]], np.int32)
        b4 = np.array([tbl, tbr, bbr, bbl], np.int32)
        b4 = b4.reshape((-1, 1, 2))
        cv2.polylines(img, [b4], True, (255, 0, 0))
        plt.imshow(img), plt.show()


        cv2.imshow("img", img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()






def main():

    matrix_3d = get3d(left_path, right_path, calib_path)
    image_path = run_single_image("006081")
    # image_path = "image/006327.png"
    # box_matrix = np.load("006327_Car.npy")
    # print(image_path)
    # old_im = Image.open(image_path)
    # # old_im = cv2.imread(image_path)
    # # old_im = cv2.cvtColor(old_im, cv2.COLOR_BGR2GRAY)
    # new_size = (2500, 2000)
    # new_im = Image.new("RGB", new_size)
    # new_im.paste(old_im, (0, 0))
    # new_im.save('test_image_with_border.jpg')
    # img = cv2.imread('test_image_with_border.jpg')
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # # print("image shape is: {}".format(img.shape))
    # # img = cv2.imread(image_path)
    # seg_matrix = get_seg(box_matrix, matrix_3d, img)
    # draw_box(img, seg_matrix)


if __name__ == '__main__':
    main()
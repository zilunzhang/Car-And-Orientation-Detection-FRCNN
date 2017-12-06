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
from IoU import bb_intersection_over_union
from sklearn.metrics import confusion_matrix

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

image_dir = "image/"
label_dir = "label/"

T = 0.5

# for infile in sorted(glob.glob('*.txt')):
    #     print "Current File Being Processed is: " + infile


def get_all_images_acc_cf(input_path):
    filenames = get_file_names(input_path)
    z_size = len(filenames)
    confusion_m = np.zeros((4, 4, z_size))
    for i in range(len(filenames)):
        # print(i)
        # print(filenames[i])
        # if filenames[i] == "007406":
        #     print("here")
        stripped_name = filenames[i]
        cf_matrix = get_single_image_acc_cm(stripped_name)
        confusion_m [:, :, i] = cf_matrix
    confusion_m = np.sum(confusion_m, axis=2)
    np.savetxt("FRCNN_confusion_matrix.csv", confusion_m, fmt='%10.5f', delimiter=",")
    return confusion_m


def get_single_image_acc_cm(filename):
    img_ext = ".png"
    label_ext = ".txt"

    image_path = image_dir + filename + img_ext
    label_path = label_dir + filename + label_ext



    detected_box_car, detected_box_pedestrian, detected_box_cyclist = process_predictions(filename)

    car_matrix, pedestrian_matrix, cyclist_matrix, line_counter = process_labels(label_path)

    label_matrix = get_label_matrix(label_path, line_counter)

    # t_p, f_p, t_n = get_accuracy(detected_box_car, car_matrix, T)

    cf_matrix = confusion_matrix_for_img((detected_box_car, detected_box_pedestrian, detected_box_cyclist), label_matrix, T)

    # print(t_p)
    return cf_matrix


def get_accuracy(predicted_matrix, true_matrix, T):
    predicted_matrix = predicted_matrix.tolist()
    true_matrix = true_matrix.tolist()
    true_pos, false_pos = 0, 0
    ground_true = len(true_matrix)
    for predicted_box in predicted_matrix:
        tp_flag = 0
        for i in range(len(true_matrix)):
            if bb_intersection_over_union(predicted_box, true_matrix[i]) > T:
                tp_flag = i + 1
                break
        if tp_flag == 0:
            false_pos += 1
        else:
            true_matrix.pop(tp_flag - 1)
            true_pos += 1
    return true_pos, false_pos, ground_true - true_pos


def confusion_matrix_for_img(pred, true, T):
    """
    pred: prediction tuple of 3 elements (car_pred, per_pred, cyc_pred)
            car_pred, per_pred and cyc_pred can either be matrix or 0
    true: n x 5 npy matrix. First four columns are coordinates, the last one is
    an integer from 0 to 2. 0 -> car, 1 -> person, 2 -> cyclist.

    return: [[int]] list of list of two integers, one for pred, the other for
    true. 0 -> car, 1 -> person, 2 -> cyclist, 3 -> don't care
    """
    result = []
    # pred = [car_pred, per_pred, cyc_pred]
    true_lst = true.tolist()
    for i in range(3):
        if type(pred[i]) is int:
            continue
        pred_lst = pred[i].tolist()
        for pred_box in pred_lst:
            for j in range(len(true_lst)):
                if bb_intersection_over_union(pred_box, true_lst[j][:4]) > T:
                    result.append([i, true_lst[j][4]])
                    true_lst.pop(j)
                    break
                if j == len(true_lst) - 1:
                    result.append([i, 3])
    result += [[3, true_box[4]] for true_box in true_lst]

    result_matrix = np.reshape(np.array(result), (-1, 2))

    predict = result_matrix[:, 0]
    label = result_matrix[:, 1]

    cf_matrix = confusion_matrix(label, predict, labels=[0, 1, 2, 3])

    return cf_matrix


def process_predictions(filename):
    car_data_name = "{}_Car.npy".format(filename)
    pedestrian_data_name = "{}_Pedestrian.npy".format(filename)
    cyclist_data_name = "{}_Cyclist.npy".format(filename)
    detected_box_car = 0
    detected_box_pedestrian = 0
    detected_box_cyclist = 0

    if os.path.isfile(car_data_name):
        detected_box_car = np.load("{}_Car.npy".format(filename))
    if os.path.isfile(pedestrian_data_name):
        detected_box_pedestrian = np.load("{}_Pedestrian.npy".format(filename))
    if os.path.isfile(cyclist_data_name):
        detected_box_cyclist = np.load("{}_Cyclist.npy".format(filename))


    return detected_box_car, detected_box_pedestrian, detected_box_cyclist


def get_label_matrix(label_path, line_counter):
    label_file = open(label_path, "r")
    lines = label_file.readlines()
    return_matrix = np.zeros((len(lines), 5))
    for i in range(len(lines)):
        line = lines[i].split(" ")
        left, top, right, bottom = floor(float(line[4])), floor(float(line[5])), ceil(float(line[6])), ceil(
            float(line[7]))
        if line[0] == "Car":
            return_matrix[i, :] = [left, top, right, bottom, 0]

        elif line[0] == "Pedestrian":
            return_matrix[i, :] = [left, top, right, bottom, 1]

        elif line[0] == "Cyclist":
            return_matrix[i, :] = [left, top, right, bottom, 2]

        else:
            pass

    return_matrix = return_matrix[~(return_matrix==0).all(1)]

    return return_matrix


def process_labels(label_path) :
    car_matrix = []
    cyclist_matrix = []
    pedestrian_matrix = []

    label_file = open(label_path, "r")
    lines = label_file.readlines()
    line_counter = 0
    for line in lines:
        line = line.split(" ")
        left, top, right, bottom = floor(float(line[4])), floor(float(line[5])), ceil(float(line[6])), ceil(
            float(line[7]))
        if line[0] == "Car":
            car_matrix.append([left, top, right, bottom])
            line_counter += 1
        elif line[0] == "Cyclist":
            cyclist_matrix.append([left, top, right, bottom])
            line_counter += 1
        elif line[0] == "Pedestrian":
            pedestrian_matrix.append([left, top, right, bottom])
            line_counter += 1
    car_matrix = np.reshape(np.asarray(car_matrix), (-1, 4))
    pedestrian_matrix = np.reshape(np.asarray(pedestrian_matrix), (-1, 4))
    cyclist_matrix = np.reshape(np.asarray(cyclist_matrix), (-1, 4))

    return car_matrix, pedestrian_matrix, cyclist_matrix, line_counter


def get_file_names(path):
    names = os.listdir(path)
    return_list = []
    for name in names:
        without_ext = name.split(".")[0]
        return_list.append(without_ext)
    return return_list


if __name__ == '__main__':
    # run_model()
    get_all_images_acc_cf(image_dir)

























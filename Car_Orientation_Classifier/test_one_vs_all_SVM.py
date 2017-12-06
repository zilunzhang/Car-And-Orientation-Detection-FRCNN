import numpy as np
import cv2 as cv2
import glob as glob
import os
import matplotlib.pyplot as plt
from math import ceil
from math import floor
from sklearn.svm import SVC
from datetime import datetime
from skimage.feature import hog
from sklearn.model_selection import KFold
from scipy import stats
import pandas as pd
from sklearn.externals import joblib
from one_vs_all_SVM import get_most_common_shape, get_data
from sklearn.metrics import confusion_matrix

test_path = "output/test/"
recursive_glob_path = "output/test/*/*.png"
y = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]





def reload_pkl():
    pkl_list = []
    for i in range(12):
        name = "svm_median_{}.pkl".format(i+1)
        pkl = joblib.load(name)
        pkl_list.append(pkl)
    return pkl_list


def data_testing(datas, labels, svm_list):
    total_count = 0
    conf_3d_matrix = np.zeros((2, 2, datas.shape[0]))
    for i in range(datas.shape[0]):
        print("i : {}".format(i))
        data = datas[i].reshape(1, -1)
        label = labels[i]
        predict_list = []
        ground_true_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ground_true_list[int(label) - 1] = 1
        for j in range(len(svm_list)):
            predict_result = svm_list[j].predict(data)
            predict_list.append(predict_result)

        ground_true_list = np.asarray(ground_true_list)
        predict_list = np.asarray(predict_list).T.ravel()

        conf_matrix = confusion_matrix(ground_true_list, predict_list)
        conf_3d_matrix[:, :, i] = conf_matrix
        result_list = ground_true_list - predict_list
        result = 12 - np.count_nonzero(result_list)
        print(result)
        total_count += result
    accuracy = total_count / (12 * datas.shape[0])
    conf_2d_matrix = np.sum(conf_3d_matrix, 2)
    return accuracy, conf_2d_matrix


def data_testing_2(datas, labels, svm_list):
    result_list = []
    for i in range(datas.shape[0]):
        print("i : {}".format(i))
        data = datas[i].reshape(1, -1)
        predict_list = []
        score_list = []

        for j in range(len(svm_list)):
            predict_result = svm_list[j].predict(data)
            predict_list.append(predict_result)
            score = np.max(svm_list[j].decision_function(data))
            score_list.append(score)

        max_index = 0
        max_prob = 0
        for i in range(12):
            if predict_list[i] == 1:
                score = score_list[i]
                if score > max_prob:
                    max_index = i
                    max_prob = score

        result_list.append(np.int(y[max_index]))

    ground_true_list = labels.astype(int)
    np.save("result_list.npy", result_list)

    test_acc = (result_list == ground_true_list).mean()

    conf_2d_matrix = confusion_matrix(ground_true_list, result_list)

    return  test_acc, conf_2d_matrix



def data_pre_processing(X, y):
    image_count_list = []
    for class_data in X:
        image_count_list.append(class_data.shape[0])
    print("data as follow distribution: {}".format(image_count_list))
    data_in_one = np.concatenate(X, axis=0)
    print("X's shape is: {}".format(data_in_one.shape))

    labels = []
    for i in range(12):
        labels += [y[i]] * image_count_list[i]
    print("y's shape is: {}".format(len(labels)))

    return data_in_one, labels


def main():

    start = datetime.now()
    print("start time is: {}".format(start))
    mode, mean, median = get_most_common_shape(recursive_glob_path)
    print("mode, mean, median are: {}, {}, {}".format(mode, mean, median))
    datas = get_data(test_path, median)
    X, labels = data_pre_processing(datas, y)
    print("saving......")
    np.save("test_data.npy", X)
    np.save("test_label.npy", labels)
    print("done!")
    data = np.load("test_data.npy")
    label = np.load("test_label.npy")
    print("test data'shape is: {}".format(data.shape))
    print("test label' shape is {}".format(label.shape))
    print("loading......")
    svm_list = reload_pkl()
    print("done!")
    print("testing......")
    accuracy, conf_matrix = data_testing(data, label, svm_list)
    # accuracy = data_testing_2(data, label, svm_list)
    np.savetxt("svm confusion matrix.csv", conf_matrix, fmt='%10.5f', delimiter="," )
    print("accuracy is: {}".format(accuracy))

    stop = datetime.now()
    print("run time is: {}".format(stop-start))



if __name__ == "__main__":
    main()

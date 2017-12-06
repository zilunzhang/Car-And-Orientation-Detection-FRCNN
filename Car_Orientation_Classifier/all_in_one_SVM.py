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
from sklearn.svm import LinearSVC


input_path = "output/train/"
recursive_glob_path = "output/train/*/*.png"
y = ["-30", "-60", "-90", "-120", "-150", "-180", "30", "60", "90", "120", "150", "180"]
rescale_num = 3000

def get_most_common_shape(input_path):
    name_list = glob.glob(input_path, recursive=True)
    shape_matrix = np.zeros((len(name_list), 2))
    for i in range(len(name_list)):
        temp_img = cv2.imread(name_list[i])
        temp_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2GRAY)
        shape_matrix[i, :] = temp_img.shape
    most_common_shape_mode = stats.mode(shape_matrix, axis=0)
    most_common_shape_mean = np.ceil(np.mean(shape_matrix, axis=0)).astype(int)
    most_common_shape_median = np.ceil(np.median(shape_matrix, axis=0)).astype(int)

    most_common_shape_mode = (most_common_shape_mode[0][0][0], most_common_shape_mode[0][0][1])
    most_common_shape_mean = (most_common_shape_mean[0], most_common_shape_mean[1])
    most_common_shape_median = (most_common_shape_median[0], most_common_shape_median[1])
    print("shape info for training data is: ")
    print(pd.DataFrame(shape_matrix).describe())
    return  most_common_shape_mode, most_common_shape_mean, most_common_shape_median


def get_data(input_path, shape):
    directories = []
    for name in y:
        new_name = input_path + name + "/"
        directories.append(new_name)
    classes_features = []
    # 30, 60, ...180
    for directory in directories:
        print(directory)
        files = os.listdir(directory)
        features_matrix = np.zeros((len(files), 2592))
        for i in range(len(files)):
            abs_path = directory + files[i]
            image = cv2.imread(abs_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, shape)
            features = hog(image, feature_vector=True, visualise= False, block_norm="L2-Hys")
            # cv2.imshow("img", img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            features_matrix[i, :] = features
        classes_features.append(features_matrix)
    # list of matrices
    return classes_features


def pre_processing_data (X):
    image_count_list = []
    for class_data in X:
        image_count_list.append(class_data.shape[0])
    print("data as follow distribution: {}".format(image_count_list))

    data = np.concatenate(X, axis=0)
    label = np.array(make_labels(image_count_list, y)).ravel()


    print("data's shape is: {}".format(data.shape))
    print("label's shape is: {}".format(label.shape))


    return data, label


def train_svm(X, y):
    svm = LinearSVC(random_state=0, dual= False, multi_class="ovr", class_weight = "balanced", max_iter= 10000, C= 100.0)
    svm.fit(X, y)
    joblib.dump(svm, 'svm_median.pkl')


def make_labels(image_count_list, y):
    return_list = []
    for i in range(len(image_count_list)):
        return_list += [y[i]] * image_count_list[i]
    return return_list


def reload_pkl ():
    name = "svm_median.pkl"
    pkl = joblib.load(name)
    return pkl


def draw(img_path, bbox):
    shape = (49, 83)
    svm_list = reload_pkl()
    img =  cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    left, top, right, bottom = floor(float(bbox[0])), floor(float(bbox[1])), ceil(float(bbox[2])), ceil(float(bbox[3]))
    cropped_test_data = img[top:bottom+1, left:right+1]
    cropped_test_data = cv2.resize(cropped_test_data, shape)
    features = hog(cropped_test_data, feature_vector=True, visualise=False, block_norm="L2-Hys").reshape(1, -1)

    predict_list = []
    score_list = []
    for j in range(len(svm_list)):
        predict_result = svm_list[j].predict(features)
        score = np.max(svm_list[j].predict_proba(features))
        predict_list.append(predict_result)
        score_list.append(score)

    max_index = 0
    max_prob = 0
    for i in range(12):
        if predict_list[i] == 1:
            score = score_list[i]
            if score > max_prob:
                max_index = i
                max_prob = score

    desired_class_angle = y[max_index]
    print(desired_class_angle)


def data_testing(svm, new_data):

    predict_result = svm.predict(new_data)
    # scores = svm.predict_proba(new_data)
    print("new data might in class {}".format(predict_result))
    # print("scores are: {}".format(scores))


def get_test_data(path, shape):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, shape)
    features = hog(image, feature_vector=True, visualise= False, block_norm="L2-Hys")
    return features


def main():

    start = datetime.now()
    print("start time is: {}".format(start))

    mode, mean, median = get_most_common_shape(recursive_glob_path)
    print("mode, mean, median are: {}, {}, {}".format(mode, mean, median))
    datas = get_data(input_path, median)
    np.save("datas.npy", datas)
    datas = np.load("datas.npy")

    print("pre-processing")
    data, labels = pre_processing_data(datas)
    print("done")

    print("training......")
    train_svm(data, labels)
    print("done!")

    print("reloading......")
    svm = reload_pkl()
    print("done!")


    test_data_1 = get_test_data("/home/alaaaaan/Desktop/006005_1.png", median).reshape(1, -1)
    data_testing(svm, test_data_1)

    test_data_2 = get_test_data("/home/alaaaaan/Desktop/006007_1.png", median).reshape(1, -1)
    data_testing(svm, test_data_2)

    test_data_3 = get_test_data("/home/alaaaaan/Desktop/006000_1.png", median).reshape(1, -1)
    data_testing(svm, test_data_3)

    # path = "/home/alaaaaan/Desktop/006120.png"
    # bbox = [255, 178,  441, 304]
    # draw(path, bbox)
    stop = datetime.now()
    print("run time is: {}".format(stop-start))

if __name__ == "__main__":
    main()
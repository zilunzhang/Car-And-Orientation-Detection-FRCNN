import numpy as np
import cv2 as cv2
import glob as glob
import os

from datetime import datetime
from skimage.feature import hog

from scipy import stats
import pandas as pd
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix

test_path = "output/test/"
recursive_glob_path = "output/test/*/*.png"
y = ["-30", "-60", "-90", "-120", "-150", "-180", "30", "60", "90", "120", "150", "180"]


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


def data_testing_2(datas, labels, svm):

    result_list = []
    for i in range(datas.shape[0]):
        print("i : {}".format(i))
        data = datas[i].reshape(1, -1)
        predict_result = svm.predict(data)
        result_list.append(predict_result)
    ground_true_list = labels
    np.save("result_list.npy", result_list)

    result_list = np.array(result_list).ravel()
    ground_true_list = np.array(ground_true_list).ravel()

    test_acc =  (result_list == ground_true_list).mean()

    conf_2d_matrix = confusion_matrix(ground_true_list, result_list)

    return  test_acc, conf_2d_matrix



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


def make_labels(image_count_list, y):
    return_list = []
    for i in range(len(image_count_list)):
        return_list += [y[i]] * image_count_list[i]
    return return_list


def reload_pkl ():
    name = "svm_median.pkl"
    pkl = joblib.load(name)
    return pkl


def main():

    start = datetime.now()
    print("start time is: {}".format(start))
    mode, mean, median = get_most_common_shape(recursive_glob_path)
    print("mode, mean, median are: {}, {}, {}".format(mode, mean, median))
    datas = get_data(test_path, median)
    X, labels = pre_processing_data(datas)
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
    accuracy, conf_matrix = data_testing_2(data, label, svm_list)
    # accuracy = data_testing_2(data, label, svm_list)
    np.savetxt("svm confusion matrix.csv", conf_matrix, fmt='%10.5f', delimiter="," )
    print("accuracy is: {}".format(accuracy))

    stop = datetime.now()
    print("run time is: {}".format(stop-start))



if __name__ == "__main__":
    main()
import numpy as np
import cv2 as cv2
import glob as glob
import os
from math import ceil
from math import floor
from sklearn.svm import SVC
from datetime import datetime
from skimage.feature import hog
from scipy import stats
import pandas as pd
from sklearn.externals import joblib
from math import cos, sin
font = cv2.FONT_HERSHEY_DUPLEX


input_path = "./output/train/"
recursive_glob_path = "./output/train/*/*.png"
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


def get_test_data(path, shape):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, shape)
    features = hog(image, feature_vector=True, visualise= False, block_norm="L2-Hys")
    return features



def pre_processing_data (X, index):
    image_count_list = []
    for class_data in X:
        image_count_list.append(class_data.shape[0])
    print("data as follow distribution: {}".format(image_count_list))

    data_1 = X[index]
    data_except_1 = np.delete(X, index)

    else_in_one = np.concatenate(data_except_1, axis=0)
    np.random.shuffle(else_in_one)
    else_in_one = else_in_one[:np.int(data_1.shape[0]*1.3), :]
    print("1 class's shape is: {}".format(data_1.shape))
    print("0 class's shape is: {}".format(else_in_one.shape))
    all_in_one_data = np.concatenate((data_1, else_in_one), axis=0)
    all_in_one_label = np.zeros((all_in_one_data.shape[0], 1))
    all_in_one_label[:data_1.shape[0], :] = 1
    all_in_one_label = all_in_one_label.ravel()

    return all_in_one_data, all_in_one_label


def train_svm(X, y):
    svm_list = []
    for i in range(len(y)):
        data, label = pre_processing_data(X, i)
        print("data shape after pre-processing is: {}, and label shape after processing is:{}".format(data.shape, label.shape))
        clf = SVC(kernel="linear", probability= True)
        clf.fit(data, label)
        joblib.dump(clf, 'svm_median_{}.pkl'.format(i+1))
        svm_list.append(clf)
    return svm_list


def reload_pkl ():
    pkl_list = []
    for i in range(12):
        name = "svm_median_{}.pkl".format(i+1)
        pkl = joblib.load(name)
        pkl_list.append(pkl)
    return pkl_list


def draw(img_path, bbox):
    shape = (49, 83)
    svm_list = reload_pkl()
    rgb_img =  cv2.imread(img_path)
    img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
    left, top, right, bottom = int(floor(float(bbox[0]))), int(floor(float(bbox[1]))), int(ceil(float(bbox[2]))), int(ceil(float(bbox[3])))
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
    angle = int(y[max_index])
    desired_class_angle = int(y[max_index])/ 180 * np.pi

    print(desired_class_angle)

    midpoint_horizentol = int((left + right)/2)
    midpoint_vertical = int((top + bottom) / 2)
    start_point = (midpoint_horizentol, midpoint_vertical)
    cosine = cos(desired_class_angle)
    sine = sin(desired_class_angle)
    end_point = (int(midpoint_horizentol -100 * cosine), int(midpoint_vertical + 100 * sine))

    cv2.arrowedLine(rgb_img, start_point, end_point,(210, 142, 40), 3)

    pts = np.array([[left, top],
                    [right, top],
                    [right, bottom],
                    [left, bottom]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    rgb_img = cv2.polylines(rgb_img, [pts], True, (0, 0, 255), thickness = 4)
    cv2.putText(rgb_img, '60~{} degrees'.format(angle), (int((start_point[0] + end_point[0])/2)+10, int((start_point[1] + end_point[1])/2)), font, 0.5, (210, 142, 40), 1, cv2.LINE_AA)
    cv2.putText(rgb_img, 'Car', (left, top - 5), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imshow("img", rgb_img)
    cv2.waitKey(0)

def main():

    start = datetime.now()
    print("start time is: {}".format(start))

    mode, mean, median = get_most_common_shape(recursive_glob_path)
    print("mode, mean, median are: {}, {}, {}".format(mode, mean, median))
    datas = get_data(input_path, median)
    np.save("datas.npy", datas)
    datas = np.load("datas.npy")

    print("training......")
    train_svm(datas, y)
    print("done!")


    path = "/home/alaaaaan/Desktop/006120.png"
    bbox = [255, 178,  441, 304]
    draw(path, bbox)
    stop = datetime.now()
    print("run time is: {}".format(stop-start))

if __name__ == "__main__":
    main()

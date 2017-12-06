import numpy as np
import cv2 as cv2
import glob as glob
import os
import matplotlib.pyplot as plt
from math import ceil
from math import floor
from datetime import datetime

# input_img_path = "./data/training/image_2"
# training_or_testing = "training"
# train_or_test = "train"

input_img_path = "./data/testing/image_2"
training_or_testing = "testing"
train_or_test = "test"

def crop_car():

    filenames = get_file_names(input_img_path)

    count = 0

    count_n_30, count_n_60, count_n_90, count_n_120, count_n_150, count_n_180, count30, count60, count90, count120, count150, count180 = 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1

    for filename in filenames:

        print("file count is: {}".format(count))
        img_name = "{}.png".format(filename)
        label_name = "{}.txt".format(filename)
        file_path = "./data/{}/image_2/{}".format(training_or_testing, img_name)
        label_path = "./data/{}/label_2/{}".format(training_or_testing, label_name)
        img = cv2.imread(file_path)
        label_file = open(label_path, "r")
        lines = label_file.readlines()

        acc_n_30, acc_n_60, acc_n_90, acc_n_120, acc_n_150, acc_n_180, acc30, acc60, acc90, acc120, acc150, acc180 = 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1

        for line in lines:
            line = line.split(" ")
            if line[0] == "Car":
                left, top, right, bottom = floor(float(line[4])), floor(float(line[5])), ceil(float(line[6])), ceil(float(line[7]))
                angle = float(line[-1])
                angle = (angle/np.pi)*180
                crop = img[top:bottom+1, left:right+1, :]


                if -180 <= angle < -150:
                    cv2.imwrite("./output/{}/-180/{}_{}.png".format(train_or_test, filename, acc_n_180), crop)
                    acc_n_180 += 1
                    count_n_180 += 1
                elif -150 <= angle < -120:
                    cv2.imwrite("./output/{}/-150/{}_{}.png".format(train_or_test, filename, acc_n_150), crop)
                    acc_n_150 += 1
                    count_n_150 += 1
                elif -120 <= angle < -90:
                    cv2.imwrite("./output/{}/-120/{}_{}.png".format(train_or_test, filename, acc_n_120), crop)
                    acc_n_120 += 1
                    count_n_120 += 1
                elif -90 <= angle < -60:
                    cv2.imwrite("./output/{}/-90/{}_{}.png".format(train_or_test, filename, acc_n_90), crop)
                    acc_n_90 += 1
                    count_n_90 += 1
                elif -60 <= angle < -30:
                    cv2.imwrite("./output/{}/-60/{}_{}.png".format(train_or_test, filename, acc_n_60), crop)
                    acc_n_60 += 1
                    count_n_60 += 1
                elif -30 <= angle < 0:
                    cv2.imwrite("./output/{}/-30/{}_{}.png".format(train_or_test, filename, acc_n_30), crop)
                    acc_n_30 += 1
                    count_n_30 += 1
                elif 0 <= angle < 30:
                    cv2.imwrite("./output/{}/30/{}_{}.png".format(train_or_test, filename, acc30), crop)
                    acc30 += 1
                    count30 += 1
                elif 30 <= angle < 60:
                    cv2.imwrite("./output/{}/60/{}_{}.png".format(train_or_test, filename, acc60), crop)
                    acc60 += 1
                    count60 += 1
                elif 60 <= angle < 90:
                    cv2.imwrite("./output/{}/90/{}_{}.png".format(train_or_test, filename, acc90), crop)
                    acc90 += 1
                    count90 += 1
                elif 90 <= angle < 120:
                    cv2.imwrite("./output/{}/120/{}_{}.png".format(train_or_test, filename, acc120), crop)
                    acc120 += 1
                    count120 += 1
                elif 120 <= angle < 150:
                    cv2.imwrite("./output/{}/150/{}_{}.png".format(train_or_test, filename, acc150), crop)
                    acc150 += 1
                    count150 += 1
                elif 150 <= angle < 180:
                    cv2.imwrite("./output/{}/180/{}_{}.png".format(train_or_test, filename, acc180), crop)
                    acc180 += 1
                    count180 += 1

        count+=1

    print("Count n30: {}, Count n60: {}, Count n90: {}, Count n120: {}, Count n150: {}, Count n180: {}, "
          "Count 30: {}, Count 60: {}, Count 90: {}, Count 120: {}, Count 150: {}, Count 180: {}"
          .format(count_n_30, count_n_60, count_n_90, count_n_120, count_n_150, count_n_180,
                  count30, count60, count90, count120, count150, count180))


def get_file_names(path):
    names = os.listdir(path)
    return_list = []
    for name in names:
        without_ext = name.split(".")[0]
        return_list.append(without_ext)
    return return_list

def main():
    start = datetime.now()
    print("start time is: {}".format(start))
    crop_car()
    stop = datetime.now()
    print("run time is: {}".format(stop-start))

if __name__ == "__main__":
    main()
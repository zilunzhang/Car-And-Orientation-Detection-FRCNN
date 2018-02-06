# Create first network with Keras
import os

import cv2
import matplotlib.pyplot as plt
import numpy
import numpy as np
from keras.models import Sequential
from keras.models import model_from_json
from matplotlib import pylab
from mpl_toolkits import mplot3d
from copy import deepcopy

# from . import plane

os.chdir('../left')


def readCalib(calib):
    P1 = []
    with open(calib, "r") as file:
        lines = file.readlines()
        for line in lines:
            words = line.split()
            if words[0] == "P1:":
                P1 = words[1:]
    return P1

def concate3d(x, y, z):
    M, N = z.shape
    image_3d = np.zeros((M, N, 3))

    image_3d[:, :, 0] = x
    image_3d[:, :, 1] = y
    image_3d[:, :, 2] = z

    return image_3d


def add_layer(m_3d, new):
    (M, N, R) = m_3d.shape
    result = np.zeros((M, N, 4))

    result[:, :, :3] = m_3d
    result[:, :, 3] = new

    return result


def flatten(seg_with_3d):
    result = []

    for i in range(seg_with_3d.shape[2]):
        result.append(seg_with_3d[:, :, i].flatten())

    return np.array(result).T


def filter_by_y(seg_with_3d, T):
    y = deepcopy(seg_with_3d[:, :, 1])

    y[y > 70] = 70

    y[y < 0] = 0

    result = y + seg_with_3d[:, :, 3]
    result[result > 100 + T] = 0
    result[result < 95] = 0
    result[result != 0] = 100
    return result


def get3d(left, right, calib):
    imgL = cv2.imread(left, 0)
    imgR = cv2.imread(right, 0)

    stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=64, blockSize=9, uniquenessRatio=2,
                                   speckleWindowSize=50, speckleRange=2)
    disparity = stereo.compute(imgL, imgR)

    # gt70 = len(np.argwhere(disparity < 70))
    # total = disparity.shape[0] * disparity.shape[1]

    disparity[disparity < 70] = 70

    P1 = readCalib(calib)

    f = float(P1[0]) / 10
    baseline = -float(P1[3]) / f * 100
    depth = f * baseline * (1 / disparity * 10) / 300 - 10
    depth[depth < -3] = -3
    # plt.imshow(depth), plt.colorbar(), plt.show()

    [M, N] = depth.shape

    x_2d = np.tile(np.asarray(list(range(0, N))), (M, 1))
    y_2d = np.tile(np.flip(np.asarray(list(range(0, M))), axis=0), (N, 1)).T

    x_3d = (x_2d - x_2d.shape[1] / 2) * depth / f
    y_3d = (y_2d - y_2d.shape[0] / 2) * depth / f + 10
    y_3d[y_3d < 0] = 0.001
    y_3d[y_3d > 50] = 50

    return concate3d(x_3d, y_3d, depth)


def get_img_path(fold_ins, left_or_right, img_num, training_or_testing):
    if left_or_right == 'left':
        # e.g. training/image/um_
        prefix = training_or_testing + '/image/' + fold_ins
    else:
        # e.g. ../right/testing/image/um_
        prefix = '../right/' + training_or_testing + '/image/' + fold_ins

    if img_num > 9:
        path = prefix + '0000' + str(img_num) + '.png'
    else:
        path = prefix + '00000' + str(img_num) + '.png'

    return path


def get_calib_path(fold_ins, img_num, training_or_testing):
    if img_num > 9:
        path = training_or_testing + '/calib/' + fold_ins + '0000' + str(img_num) + '.txt'
    else:
        path = training_or_testing + '/calib/' + fold_ins + '00000' + str(img_num) + '.txt'

    return path


def loadSeg(file_name):
    # Load Data
    seg = np.load(file_name)

    # Display Message
    print('Segments for all images are Successfully Loaded!')
    return seg


def as_model(j_data):
    model = Sequential()
    model.__dict__.update(j_data)
    return model


def test_data(testing_num, plot_img_num):
    model_name = 'training_data0'

    # test on training data 2
    dataset = numpy.loadtxt('training_data' + str(testing_num) + '.out', delimiter=',')

    X = dataset[:, 0:64].astype(float)
    Y = dataset[:, 64].astype(int)
    image_numbers = np.unique(dataset[:, 64:].astype(int)[:, 1])

    # load json and create model
    json_file = open(model_name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_name + '.h5')
    print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    score = loaded_model.evaluate(X, Y, verbose=0)
    # estimated_val = loaded_model.predict(X)
    print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))

    segments_for_all_images = loadSeg('training_seg' + str(testing_num) + '.npy')

    counter = 0
    for img_num in image_numbers:

        ins = ['um_', 'uu_', 'umm_']
        training_or_testing = ['training', 'testing']
        left_or_right = ['left', 'right']

        # get calib path and img path
        left_img_path = get_img_path(ins[testing_num], left_or_right[0], img_num, training_or_testing[0])
        right_img_path = get_img_path(ins[testing_num], left_or_right[1], img_num, training_or_testing[0])
        calib_path = get_calib_path(ins[testing_num], img_num, training_or_testing[0])

        # get 3d coordinates for the image
        coor_3d = get3d(left_img_path, right_img_path, calib_path)

        test_img_X = dataset[dataset[:, 65].astype(int) == img_num][:, 0:64].astype(float)
        # test_img0 = info[info[:, 1] == img_num]
        segments_for_img = segments_for_all_images[img_num - 1]
        is_road = loaded_model.predict(test_img_X)

        for seg_num in range(test_img_X.shape[0]):
            segments_for_img[segments_for_img == seg_num] = is_road[seg_num] - 2

        # -2 corresponds to the pixels that are not in road
        # so 100 means road, 0 means not road
        segments_for_img[segments_for_img != -2] = 100
        segments_for_img[segments_for_img == -2] = 0

        # merge segments and 3d coordinates for the image
        seg_with_coor3d = add_layer(coor_3d, segments_for_img)

        # flatten data
        flatten_seg_with_coor = flatten(seg_with_coor3d)

        # look for flatten road 3d coordinates
        road_3d_coor = flatten_seg_with_coor[flatten_seg_with_coor[:, 3] == 100]
        road_3d_coor = road_3d_coor[road_3d_coor[:, 1] > -3][:, :3]
        road_3d_coor = road_3d_coor[road_3d_coor[:, 1] < 7]
        road_3d_coor = road_3d_coor[road_3d_coor[:, 2] > 0]
        road_3d_coor[:, 1:] = np.flip(road_3d_coor[:, 1:], 0)

        if counter == plot_img_num:
            # filter by 3d y
            # segments_for_img = filter_by_y(seg_with_coor3d, T=7)
            print('plot training dataset ', ins[testing_num], ', image #', counter)
            np.savetxt('3d_coor' + str(counter) + '.out', road_3d_coor, delimiter=',', fmt='%.4f')

            plt.imshow(segments_for_img), plt.show()
            break

        counter += 1


def test_data_testing(testing_num, plot_img_num):
    model_name = 'training_data0'

    # test on training data 2
    dataset = numpy.loadtxt('testing_data' + str(testing_num) + '.out', delimiter=',')

    X = dataset[:, 0:64].astype(float)
    Y = dataset[:, 64].astype(int)
    image_numbers = np.unique(dataset[:, 64:].astype(int)[:, 1])

    # load json and create model
    json_file = open(model_name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_name + '.h5')
    print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # score = loaded_model.evaluate(X, Y, verbose=0)
    # estimated_val = loaded_model.predict(X)
    # print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))

    segments_for_all_images = loadSeg('testing_seg' + str(testing_num) + '.npy')

    counter = 0
    for img_num in image_numbers:

        ins = ['um_', 'uu_', 'umm_']
        training_or_testing = ['training', 'testing']
        left_or_right = ['left', 'right']

        # get calib path and img path
        left_img_path = get_img_path(ins[testing_num], left_or_right[0], img_num, training_or_testing[1])
        right_img_path = get_img_path(ins[testing_num], left_or_right[1], img_num, training_or_testing[1])
        calib_path = get_calib_path(ins[testing_num], img_num, training_or_testing[1])

        # get 3d coordinates for the image
        coor_3d = get3d(left_img_path, right_img_path, calib_path)

        test_img_X = dataset[dataset[:, 65].astype(int) == img_num][:, 0:64].astype(float)
        # test_img0 = info[info[:, 1] == img_num]
        segments_for_img = segments_for_all_images[img_num - 1]
        is_road = loaded_model.predict(test_img_X)

        for seg_num in range(test_img_X.shape[0]):
            segments_for_img[segments_for_img == seg_num] = is_road[seg_num] - 2

        # -2 corresponds to the pixels that are not in road
        # so 100 means road, 0 means not road
        segments_for_img[segments_for_img != -2] = 100
        segments_for_img[segments_for_img == -2] = 0

        # merge segments and 3d coordinates for the image
        seg_with_coor3d = add_layer(coor_3d, segments_for_img)

        # # flatten data
        # flatten_seg_with_coor = flatten(seg_with_coor3d)
        #
        # # look for flatten road 3d coordinates
        # road_3d_coor = flatten_seg_with_coor[flatten_seg_with_coor[:, 3] == 100]
        # road_3d_coor = road_3d_coor[road_3d_coor[:, 1] > -3][:, :3]
        # road_3d_coor = road_3d_coor[road_3d_coor[:, 1] < 7]
        # road_3d_coor = road_3d_coor[road_3d_coor[:, 2] > 0]
        # road_3d_coor[:, 1:] = np.flip(road_3d_coor[:, 1:], 0)

        if counter == plot_img_num:
            # filter by 3d y
            segments_for_img = filter_by_y(seg_with_coor3d, T=7)
            print('plot testing dataset ', ins[testing_num], ', image #', counter)
            # np.savetxt('3d_coor' + str(counter) + '.out', road_3d_coor, delimiter=',', fmt='%.4f')

            plt.imshow(segments_for_img), plt.show()
            break

        counter += 1


test_data(testing_num=2, plot_img_num=30)

# test_data_testing(testing_num=0, plot_img_num=1)

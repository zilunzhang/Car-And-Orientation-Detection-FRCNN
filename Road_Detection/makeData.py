# Data Loading and Feature Extraction

from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage.measure import perimeter
from skimage import io
import numpy as np
from copy import deepcopy
import cv2
import matplotlib.pyplot as plt
import os

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
    depth = f * baseline * (1 / disparity * 10) / 100 - 10
    depth[depth < 5] = 5
    # plt.imshow(depth), plt.colorbar(), plt.show()

    [M, N] = depth.shape

    x_2d = np.tile(np.asarray(list(range(0, N))), (M, 1))
    y_2d = np.tile(np.flip(np.asarray(list(range(0, M))), axis=0), (N, 1)).T

    x_3d = (x_2d - x_2d.shape[1] / 2) * depth / f
    y_3d = (y_2d - y_2d.shape[0] / 2) * depth / f + 10
    y_3d[y_3d < 0] = 0.001
    # y_3d[y_3d > 50] = 50

    return concate3d(x_3d, y_3d, depth)


# Read Images, Perform Slic, Save the Data
def saveTrainingData(i):
    instruction = ['um_', 'uu_', 'umm_']
    imgNum = len([name for name in os.listdir('training/image') if name[0:len(instruction[i])] == instruction[i]])

    # Bins used for Histogram
    bins = np.linspace(0,1,num=21)
    
    # Data Container: [centerx, centery, area, perimeter, rhist:20bin, ghist, bhist; label]
    data = []

    # segments for all the images
    segments_for_all = []
    
    # Start Extracting Data
    print('Start Reading Training Images...')
    
    # Start Main Loop
    for idx in range(imgNum):
        
        # Display Message
        print('Loading Image #',idx)
    
        # Load Image and Ground Truth Image as Float
        if idx > 9:
            img_path = 'training/image/' + instruction[i] + '0000'+str(idx)+'.png'
            img_gt = img_as_float(io.imread('training/gt_image/' + instruction[i] + 'road_0000' +str(idx)+'.png'))
        else:
            img_path = 'training/image/' + instruction[i] + '00000' + str(idx) + '.png'
            img_gt = img_as_float(io.imread('training/gt_image/' + instruction[i] + 'road_00000' + str(idx) + '.png'))
        img = img_as_float(io.imread(img_path))

        # Perform SLIC
        img_seg = slic(img, n_segments = 1000, compactness=8, convert2lab=True, min_size_factor=0.3)

        # add segments for a single image
        segments_for_all.append(img_seg)

        # get 3d coordinates
        coor_3d = np.zeros((img_seg.shape[0], img_seg.shape[1], 4))
        if idx > 9:
            right_path = '../right/training/image/' + instruction[i] + '0000' + str(idx) + '.png'
            calib_path = 'training/calib/' + instruction[i] + '0000' + str(idx) + '.txt'
        else:
            right_path = '../right/training/image/' + instruction[i] + '00000' + str(idx) + '.png'
            calib_path = 'training/calib/' + instruction[i] + '00000' + str(idx) + '.txt'
        coor_3d[:, :, :3] = get3d(img_path, right_path, calib_path)

        # Loop through all Segments
        for label in range(np.amax(img_seg)+1):
            
            # Display Message
            if label%100 == 0:
                print('Processing Image #:',idx+1,' Seg #:',label)
            
            # Create Entry for Segment
            data.append([])
            
            # Boolean Mask of Region of Interests
            roi = (img_seg == label)
            
            # Extract Index Information
            roi_idx = np.nonzero(roi)
            # Attach center x(row), center y(col)
            data[-1].append(np.mean(roi_idx[0]))
            data[-1].append(np.mean(roi_idx[1]))
    
            # Append Area
            data[-1].append(np.shape(roi_idx)[1])
            # Append Perimeter
            data[-1].append(perimeter(roi))
            # Extract Color Information
            roi_rgb = img[roi]


            image_seg = deepcopy(roi)

            plt.imshow(image_seg), plt.show()


            # Adding R,G,B Channels
            for channel in range(3):
                hist, b = np.histogram(roi_rgb[:,channel],bins,density=True)
                data[-1].extend(hist.tolist())

            # adding 3d feature
            coor_3d[:, :, 3] = img_seg
            coor_3d[:, :, 3][coor_3d[:, :, 3] != label] = -1
            coor_3d[:, :, 3][coor_3d[:, :, 3] == label] = 1
            y = coor_3d[:, :, 1] * coor_3d[:, :, 3]
            y_median = np.median(y[y > 0]) / 200
            data[-1].append(y_median.item())
    
            # Adding Label
            if img_gt[int(data[-1][0]),int(data[-1][1]),2] > 0:
                data[-1].append(1)
            else:
                data[-1].append(0)
    
            # Append Data Flag
            data[-1].append(idx+1)

            # add segment flag
            data[-1].append(label)
    # Output Data to file
    np.save('training_data' + str(i), data)

    # output segments to file
    np.save('training_seg' + str(i), segments_for_all)

    # Display Message
    print('Data Successfully Saved!')
    return


def saveTestingData(i):
    instruction = ['um_', 'uu_', 'umm_']
    imgNum = len([name for name in os.listdir('testing/image') if name[0:len(instruction[i])] == instruction[i]])

    # Bins used for Histogram
    bins = np.linspace(0, 1, num=21)

    # Data Container: [centerx, centery, area, perimeter, rhist:20bin, ghist, bhist; label]
    data = []

    # segments for all the images
    segments_for_all = []

    # Start Extracting Data
    print('Start Reading Testing Images...')

    # Start Main Loop
    for idx in range(imgNum):

        # Display Message
        print('Loading Image #', idx, sep='')

        # Load Image and Ground Truth Image as Float
        if idx > 9:
            img_path = 'testing/image/' + instruction[i] + '0000' + str(idx) + '.png'
        else:
            img_path = 'testing/image/' + instruction[i] + '00000' + str(idx) + '.png'
        img = img_as_float(io.imread(img_path))

        # Perform SLIC
        img_seg = slic(img, n_segments=1000, compactness=8, convert2lab=True, min_size_factor=0.3)

        # add segments for a single image
        segments_for_all.append(img_seg)

        # # get 3d coordinates
        # coor_3d = np.zeros((img_seg.shape[0], img_seg.shape[1], 4))
        # if idx > 9:
        #     right_path = '../right/testing/image/' + instruction[i] + '0000' + str(idx) + '.png'
        #     calib_path = 'testing/calib/' + instruction[i] + '0000' + str(idx) + '.txt'
        # else:
        #     right_path = '../right/testing/image/' + instruction[i] + '00000' + str(idx) + '.png'
        #     calib_path = 'testing/calib/' + instruction[i] + '00000' + str(idx) + '.txt'
        # coor_3d[:, :, :3] = get3d(img_path, right_path, calib_path)

        # Loop through all Segments
        for label in range(np.amax(img_seg) + 1):

            # Display Message
            if label % 100 == 0:
                print('Processing Image #:', idx + 1, ' Seg #:', label, sep='')

            # Create Entry for Segment
            data.append([])

            # Boolean Mask of Region of Interests
            roi = (img_seg == label)

            # Extract Index Information
            roi_idx = np.nonzero(roi)
            # Attach center x(row), center y(col)
            data[-1].append(np.mean(roi_idx[0]))
            data[-1].append(np.mean(roi_idx[1]))

            # Append Area
            data[-1].append(np.shape(roi_idx)[1])
            # Append Perimeter
            data[-1].append(perimeter(roi))
            # Extract Color Information
            roi_rgb = img[roi]
            # Adding R,G,B Channels
            for channel in range(3):
                hist, b = np.histogram(roi_rgb[:, channel], bins, density=True)
                data[-1].extend(hist.tolist())

            # # adding 3d feature
            # coor_3d[:, :, 3] = img_seg
            # coor_3d[:, :, 3][coor_3d[:, :, 3] != label] = -1
            # coor_3d[:, :, 3][coor_3d[:, :, 3] == label] = 1
            # y = coor_3d[:, :, 1] * coor_3d[:, :, 3]
            # y_median = np.median(y[y > 0]) / 200
            # data[-1].append(y_median.item())

            # Append Data Flag
            data[-1].append(idx + 1)

            # add segment flag
            data[-1].append(label)

    # Output Data to file
    np.save('testing_data' + str(i), data)

    # output segments to file
    np.save('testing_seg' + str(i), segments_for_all)

    # Display Message
    print('Data Successfully Saved!')
    return


def loadTrainingData(i):
    # Load Data
    data = np.load('training_data' + str(i) + '.npy')
    
    # Display Message
    print('Data Successfully Loaded!')
    return data


def loadTestingData(i):
    # Load Data
    data = np.load('testing_data' + str(i) + '.npy')

    # Display Message
    print('Data Successfully Loaded!')
    return data


op = 0

# Save Data to File
if op == 0:
    for i in range(3):
        saveTrainingData(i)
    for j in range(3):
        saveTestingData(j)
# Load Data to Workspace
elif op == 1:
    for i in range(3):
        training_data = loadTrainingData(i)
        testing_data = loadTestingData(i)

# left = './training/image/um_000003.png'
# right = '../right/training/image_3/um_000003.png'
# calib = './training/calib/um_000003.txt'
#
# concate = get3d(left, right, calib)
#
# plt.imshow(concate[:,:,0]), plt.show()
# plt.imshow(concate[:,:,1]), plt.show()
# plt.imshow(concate[:,:,2]), plt.show()



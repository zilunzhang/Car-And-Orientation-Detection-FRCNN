# zhan1381
This is our group's final project's code and data. 

Suggestion: download those data files: https://drive.google.com/open?id=1R9h9arNOHKpY12K4ap1fGtcxPDiysrAg and put them into correspoding path on your local folder.

Thanks for CharlesShang's open source!

______________________________________________________


Car_Orientation_Classifier
  - This is Car Orientation Classifier (Using SVM)

Env Requirement: 
- Using python 3.6, conda installed version.
- Install numpy, opencv3 menpo version, sklearn, skimage by anaconda3.
- Download KITTI dataset: http://www.cvlibs.net/download.php?file=data_object_image_2.zip. 
- Using only training set (testing set does not have labels), divided it to training and testing again (training 6000, testing 1481)
- put new training set image to Car_Orientation_Classifier/data/training/image_2
- put new training set label to Car_Orientation_Classifier/data/training/label_2
- put new testing set image to Car_Orientation_Classifier/data/testing/image_2
- put new testing set label to Car_Orientation_Classifier/data/testing/label_2

How to run:
- run MakeData.py (make test and train set respectively. After run first time, comment out 14-16 and uncomment 10-12 and rerun)
- run all_in_one_SVM.py if you want to use method 1 Notice: change path in main
- run one_vs_all_SVM.py if you want to use method 2 Notice: change path in main
- run their respective test file to get the accuracy and confusion matrix Notice: change path in main

_______________________________________________________


Faster_RCNN_with_KITTI
  - This is Faster-RCNN Object detector and Classifier

Env Requirement: 
- Using python 2.7 with tensorflow 1.3 (1.4 will cause error), conda installed version.
- Install packages: lots of packages, show on "Dependency" file. Make sure you have all of them. If this is too much, install when error appears is OK.
- Download KITTI dataset: http://www.cvlibs.net/download.php?file=data_object_image_2.zip. 
- Using only training set (testing set does not have labels), divided it to training and testing again (training 6000, testing 1481)
- put new training set image to Faster_RCNN_with_KITTI/data/training/image_2
- put new training set label to Faster_RCNN_with_KITTI/data/training/label_2
- put new testing set image to Faster_RCNN_with_KITTI/data/testing/image_2
- put new testing set label to Faster_RCNN_with_KITTI/data/testing/label_2
- Install Faster-RCNN, follow instruction on https://github.com/CharlesShang/TFFRCNN

How to run:
- run train_net.py to train the network or you can use pre-trained model on google drive.
- run test_kitti.py on Faster_RCNN_with_KITTI/faster_rcnn/test_KITTI to look result for Faster-RCNN
- run FRCNN_getaccuracy.py on Faster_RCNN_with_KITTI/faster_rcnn/test_FRCNN_3d to get the accuracy and confusion matrix.
- run run3d.py on Faster_RCNN_with_KITTI/faster_rcnn/test_FRCNN_3d to get the result demo for 3d bounding box. Note for this one you need to download the right Object dataset from KITTI and put them to specific location following the instruction on code file. 



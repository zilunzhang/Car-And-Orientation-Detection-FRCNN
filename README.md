# zhan1381
This is our group's final project's code and file. 
______________________________________________________
Car_Orientation_Classifier
  - This is Car Orientation Classifier (Using SVM)

Env Requirement: 
- Using python 3.6, conda installed version.
- Install numpy, opencv3 menpo version by anaconda3.
- Download KITTI dataset: http://www.cvlibs.net/download.php?file=data_object_image_2.zip. 
- Using only training set (testing set does not have labels), divided it to training and testing again (training 6000, testing 1481)
- put new training set image to Car_Orientation_Classifier/data/training/image_2
- put new training set label to Car_Orientation_Classifier/data/training/label_2
- put new testing set image to Car_Orientation_Classifier/data/testing/image_2
- put new testing set label to Car_Orientation_Classifier/data/testing/label_2

How to run:
- run MakeData.py
- run all_in_one_SVM.py if you want to use method 1
- run one_vs_all_SVM.py if you want to use method 2
- run their respective test file to get the accuracy and confusion matrix

_______________________________________________________

import cv2
import numpy as np
import pandas as pd
import os
import glob
import re
import pickle
from matplotlib import pyplot as plt
from sklearn import datasets, neighbors, linear_model, model_selection, svm
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split,KFold,learning_curve, LeavePOut

import utils


def hist_features(img_path):
    img = cv2.imread(img_path, 0)
    # Based on my research 256 is the value to use for full range
    hist = cv2.calcHist([img],[0],None,[256],[0,256])
    return hist.flatten()

def baseline_hist_diff(img_before, img_after):
    return hist_features(img_before) - hist_features(img_after)

if __name__ == '__main__':
    input_csv = pd.read_csv("data_binary.csv")
    FILE_EXTENSION = "tif"
    dir_name = "legs_folder/"
    img_hist = []
    patient_ids = []
    Y = []

    for index, row in input_csv.iterrows():
        patient_id = str(row["patient_id"])
        file_1 = os.path.join(dir_name, patient_id, '.'.join((row["scan_1"], FILE_EXTENSION)))
        file_2 = os.path.join(dir_name, patient_id, '.'.join((row["scan_2"], FILE_EXTENSION)))

        if not (os.path.isfile(file_1) and os.path.isfile(file_2)):
            continue
        Y.append(row["y"])

        # Skip if the file does not exist (due to poor quality)

        patient_ids.append(patient_id)
        diff = baseline_hist_diff(file_1, file_2)
        img_hist.append(diff)

    column_names = ["hist" + str(i) for i in range(256)]
    df = pd.DataFrame(img_hist, columns=column_names, index=patient_ids)
    df["y"] = Y

    data = df.loc[:, (df != 0).any(axis=0)]

    # Split the data into training set and test set.
    # DONT look at what is in the test set
    test_patient_ids = ['2','32','24','24b','6','7', '41']
    test_data = data.loc[test_patient_ids]
    train_data = data.loc[data.index.difference(test_patient_ids)]

    pickle.dump(train_data, open("train_data_binary.pkl", "wb") )
    pickle.dump(test_data, open("test_data_binary.pkl", "wb") )

    y = train_data["y"]
    X = train_data.drop('y', axis=1)

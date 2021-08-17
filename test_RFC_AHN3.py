#
#
#      0=================================0
#      |       Random Forest Tester      |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Callable script to test a RFC model trained with the AHN3 dataset
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Manos Papageorgiou - 20/08/2021
#      TU Delft - MSc Geomatics - GEO2020 MSc Thesis
#      scikit-learn RFC version


# ----------------------------------------------------------------------------------------------------------------------
#
#              Imports - common libs        
#       \**********************************/
#       

import numpy as np
import time
from laspy.file import File
import os
import pandas as pd
from pandas.io.parsers import read_csv
from sklearn.metrics import accuracy_score, jaccard_score, classification_report, confusion_matrix
from tabulate import tabulate
import pickle
from scipy.spatial import cKDTree
from point_cloud_analysis import data_preparation, uniform_sampling, Z_values, spherical_neighborhoods, density, input_dataframe

# ----------------------------------------------------------------------------------------------------------------------
#
#    calculate and export metrics
#       \******************/
#

def calculate_metrics(true_labels, predicted_labels, file):
    IOU_classes = jaccard_score(true_labels, predicted_labels, labels=[0, 1, 2],
                                average=None)  # other, building, ground
    
    OA = np.round(accuracy_score(true_labels, predicted_labels)*100, 3)
    CCI = np.round((1 - (np.var(IOU_classes) / abs(np.average(IOU_classes))))*100, 3)
    IOU_mean = np.average(IOU_classes)*100
    IOU_classes = np.round(IOU_classes*100, 3)
    IOU_mean = np.round(IOU_mean, 3)
    ####################
    class_report = classification_report(true_labels, predicted_labels)
    con_matrix = confusion_matrix(true_labels, predicted_labels, [0,1,2])
    con_table = [ ["", "other", "building", "ground"], 
        ["other", con_matrix[0][0], con_matrix[0][1], con_matrix[0][2]], 
        ["building", con_matrix[1][0], con_matrix[1][1], con_matrix[1][2]], 
        ["ground", con_matrix[2][0], con_matrix[2][1], con_matrix[2][2]] ]
    ####################

    f = "evaluation_metrics/Evaluation_metrics_" + file + ".txt"
    obj = open(f, 'w')
    table = [["Method", "OA", "mean IoU", "other IoU", "building IoU", "ground IoU", "CCI"],
             ["RF", OA, IOU_mean, IOU_classes[0], IOU_classes[1], IOU_classes[2], CCI]]
    obj.write(tabulate(table))

    ####################
    obj.write("\n\n-------------------------------------\n\n")
    obj.write(class_report)
    obj.write("\n\n-------------------------------------")
    obj.write(tabulate(con_table))
    ####################

    obj.close()
    return


# ----------------------------------------------------------------------------------------------------------------------
#
#    export ground truth and predicted labels and error map
#       \******************/
#


def export(path, Y_true, Y_pred, method):
    
    if path[-3:] == 'las':
        inFile = File(path, mode="r")
        outFile1 = File(path[:-4] + "_ground_truth_" + method + ".las", mode="w", header=inFile.header)
        outFile2 = File(path[:-4] + "_predictions_" + method + ".las", mode="w", header=inFile.header)
        outFile3 = File(path[:-4] + "_error_map_" + method + ".las", mode="w", header=inFile.header)
        # copy fields
        for dimension in inFile.point_format:
            dat = inFile.reader.get_dimension(dimension.name)
            outFile1.writer.set_dimension(dimension.name, dat)
            outFile2.writer.set_dimension(dimension.name, dat)
            outFile3.writer.set_dimension(dimension.name, dat)

        outFile1.classification = Y_true.astype(np.uint8)
        outFile2.classification = Y_pred.astype(np.uint8)
        errors = np.where(Y_pred == Y_true, 1, 0) # this is to make it an error map
        outFile3.classification = errors.astype(np.uint8)

        outFile1.close()
        outFile2.close()
        outFile3.close()
    
    elif path[-3:] == 'csv':
        df = read_csv(path)
        df['ground_truth'] = Y_true
        df['predictions'] = Y_pred
        errors = np.where(Y_pred == Y_true, 1, 0)
        df['error_map'] = errors
        df.to_csv(path[:-4] + "_ground_truth_predictions_error_map_" + method +'.csv', index=False)

    return


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#

if __name__ == '__main__':
    start = time.time()
    
    # training data folder
    path = "data/testing"
    # reading file names from the training data folder
    testing_data = os.listdir(path)
    # name of the model you want to test
    model_name = "pre_trained_models/model_RFC_AHN3.sav"
    # loading the model
    loaded_model = pickle.load(open(model_name, 'rb'))
    voxel_size = 1
    all_ground_truth = []
    all_predicted = []

    for file in testing_data:
        print('Processing ' + file[:-4] + '...')
        print('##################################')
        # path to the training data file
        file_path = path + "/" + file
        # reading coordinates and labels of the points
        points, labels = data_preparation(file_path)
        grid_candidate_center, grid_candidate_center_lbs = uniform_sampling(points, labels)
        del grid_candidate_center_lbs
        # creating the file dataframe
        df = pd.DataFrame(data=grid_candidate_center, columns=['X', 'Y', 'Z'])
        del grid_candidate_center
        # calculating the normalized Z and height below
        df = Z_values(df)
        # calculate neighborhoods
        n_test = spherical_neighborhoods(df[["X", "Y", "Z"]].values, [2, 3, 4])
        # calculating density
        df = density(df, n_test[2], '4m')
        # preparing input dataframe
        df = input_dataframe(df[["X", "Y", "Z"]].values, n_test[0], "2m", df)
        df = input_dataframe(df[["X", "Y", "Z"]].values, n_test[1], "3m", df)
        df = input_dataframe(df[["X", "Y", "Z"]].values, n_test[2], "4m", df)
        # exporting the features in csv format
        features_path = "data/testing_features/" + file[:-4] + '_features.csv'
        df.to_csv(features_path, index=False)
        del n_test
        # predict labels
        Y_pred = loaded_model.predict(df.drop(df.columns[[0, 1, 2]], axis=1).values)
        # Assign class labels to all point using the nearest neighbor approach
        tree = cKDTree(df[['X','Y','Z']].values)
        pred_ids = tree.query(points, k=1)[1]
        predictions = Y_pred[pred_ids]
        all_ground_truth += list(labels)
        all_predicted += list(predictions)
        del df
        # Evaluation metrics
        calculate_metrics(labels, predictions, file[:-4] + "_" + model_name[:-4])
        # Exorting the predicted point cloud
        export_path = "data/predicted_tiles/" + file
        export(export_path, labels, predictions, model_name[:-4])
        del labels

        print('##################################\n')

    calculate_metrics(all_ground_truth, all_predicted, 'all_tiles_' + model_name[:-4])

    end = time.time()
    print(f"Script ended after {round((end - start) / 60, 2)} minutes.")
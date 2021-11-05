#
#
#      0=================================0
#      |     Random Forest Trainer       |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Callable script to start training a RFC with AHN3 dataset
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Manos Papageorgiou - 16/08/2021
#      TU Delft - MSc Geomatics - GEO2020 MSc Thesis
#      scikit-learn RFC version


# ----------------------------------------------------------------------------------------------------------------------
#
#                    Imports        
#       \**********************************/
#       

from point_cloud_analysis import data_preparation, uniform_sampling, Z_values, spherical_neighborhoods, density, input_dataframe
import numpy as np
import time
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle



# ----------------------------------------------------------------------------------------------------------------------
#
#          train RFC model
#       \******************/
#

def RFC(X_train, Y_train, name):
    start = time.time()
    print(f"Started training RFC...")

    model = RandomForestClassifier( n_estimators = 100, 
                                    criterion = 'gini', 
                                    max_depth = 15, 
                                    min_samples_split = 2, 
                                    min_samples_leaf = 1, 
                                    max_features = 'sqrt', 
                                    bootstrap = True, 
                                    oob_score = True)

    
    model.fit(X_train, Y_train)
    filename = 'pre_trained_models/model_' + name + '.sav'
    pickle.dump(model, open(filename, 'wb'))

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    ff = open("pre_trained_models/features_ranking_model_" + name + ".txt", "w")
    ff.write("Feature ranking: \n")
    for f in range(X_train.shape[1]):
        ff.write("%d. feature %d (%f) \n" % (f + 1, indices[f], importances[indices[f]]))
    
    end = time.time()
    print(f"Finished training RFC {round((end - start) / 60, 2)} minutes.")
    return


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#

if __name__ == '__main__':
    start = time.time()
    
    # training data folder
    path = "data/training/"
    # reading file names from the training data folder
    training_data = os.listdir(path)
    # declaring the training dataframe
    df_train = pd.DataFrame()
    train_lbs = []

    for file in training_data:
        print('Processing ' + file[:-4] + '...')
        print('##################################')
        # path to the training data file
        file_path = path + file
        # reading coordinates and labels of the points
        points, labels = data_preparation(file_path)
        grid_candidate_center, grid_candidate_center_lbs = uniform_sampling(points, labels)
        train_lbs += list(grid_candidate_center_lbs)
        del grid_candidate_center_lbs
        # creating the file dataframe
        df = pd.DataFrame(data=grid_candidate_center, columns=['X', 'Y', 'Z'])
        del grid_candidate_center
        # calculating the normalized Z and height below
        df = Z_values(df)
        # calculate neighborhoods
        n_train = spherical_neighborhoods(df[["X", "Y", "Z"]].values, [2, 3, 4])
        # calculating density
        df = density(df, n_train[2], '4m')
        # preparing input dataframe
        df = input_dataframe(df[["X", "Y", "Z"]].values, n_train[0], "2m", df)
        df = input_dataframe(df[["X", "Y", "Z"]].values, n_train[1], "3m", df)
        df = input_dataframe(df[["X", "Y", "Z"]].values, n_train[2], "4m", df)
        # exporting the features in csv format
        features_path = "data/training_features/" + file[:-4] + '_features.csv'
        df.to_csv(features_path, index=False)
        # removing the point coordinates
        df = df.drop(df.columns[[0, 1, 2]], axis=1)
        del n_train
        df_train = pd.concat([df_train, df], axis=0)
        del df
        print('##################################\n')
    
    model_name = "RFC_AHN3"
    RFC(df_train.values, train_lbs, model_name)


    end = time.time()
    print(f"Script ended after {round((end - start) / 60, 2)} minutes.")

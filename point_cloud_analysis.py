#
#
#      0=================================0
#      |      point cloud analysis       |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Script with point cloud analysis functions
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Manos Papageorgiou - 16/08/2021
#      TU Delft - MSc Geomatics - GEO2020 MSc Thesis
#      3D point cloud analysis


# ----------------------------------------------------------------------------------------------------------------------
#
#              Imports - common libs        
#       \**********************************/
#       

import numpy as np
import time
from scipy.spatial import cKDTree
import os
import pandas as pd
from scipy.special import softmax
from sklearn.ensemble import RandomForestClassifier
import pickle
#import pylas
#from laspy.file import File

# ----------------------------------------------------------------------------------------------------------------------
#
#   data preparation function
#       \******************/
#

def data_preparation(dir):

    # Reading LAS or LAZ or csv file
    '''
    if dir[-3:] == "LAZ":
        laz = pylas.open(dir)
        inFile = laz.read()
        points = np.vstack((inFile.x, inFile.y, inFile.z)).transpose()
        labels = inFile.classification
    elif dir[-3:] == "las":
        inFile = File(dir)
        points = np.vstack((inFile.x, inFile.y, inFile.z)).transpose()
        labels = inFile.classification
    else:
    '''
    df = pd.read_csv(dir)
    points = df[['X','Y','Z']].values
    labels = df['labels'].values

    ###### For the AHN3 dataset ######
    # Reclassification into 3 classes, 0:other - 1:building - 2:ground ==> water and bridges added to the ground class
    labels = np.where(labels==9, 2, labels)
    labels = np.where(labels==26, 2, labels)
    labels = np.where(np.logical_and(labels!=2, labels!=6), 0, labels)
    labels = np.where(labels==6, 1, labels)

    return points, labels

# ----------------------------------------------------------------------------------------------------------------------
#
#         uniform sampling
#       \******************/
#

def uniform_sampling(points, labels):
        voxel_size = 1
        non_empty_voxel_keys, inverse, nb_pts_per_voxel = np.unique(((points - np.min(points, axis=0)) // voxel_size).astype(int), axis=0, return_inverse=True, return_counts=True)
        idx_pts_vox_sorted = np.argsort(inverse)
        # uniform sampling
        voxel_grid = {}
        grid_candidate_center, grid_candidate_center_lbs = [], []
        last_seen = 0

        for idx, vox in enumerate(non_empty_voxel_keys):
            pts_ids = idx_pts_vox_sorted[last_seen:last_seen+nb_pts_per_voxel[idx]]
            voxel_grid[tuple(vox)] = points[pts_ids]
            lbs = labels[pts_ids]
            id = np.linalg.norm(voxel_grid[tuple(vox)] - np.mean(voxel_grid[tuple(vox)],axis=0),axis=1).argmin()
            grid_candidate_center.append( voxel_grid[tuple(vox)][id] ) 
            grid_candidate_center_lbs.append( lbs[id] )
            last_seen+=nb_pts_per_voxel[idx]

        return grid_candidate_center, grid_candidate_center_lbs



# ----------------------------------------------------------------------------------------------------------------------
#
#   calculate spherical neighborhoods function
#       \******************/
#

def spherical_neighborhoods(pts, radiuses):

    tree = cKDTree(pts)
    neighborhoods = []
    for r in radiuses:
        n = tree.query_ball_point(pts, r)
        neighborhoods.append(n)

    return neighborhoods

# ----------------------------------------------------------------------------------------------------------------------
#
#   calculate normalized and neighborhood Z values
#       \******************/
#

def Z_values(df):
    start = time.time(); print(f"Started Z values calculation...")

    candidate_center_xy = df[["X","Y"]].values
    candidate_center_z = df["Z"].values
    kdtree = cKDTree( candidate_center_xy )
    zb = np.zeros( (df.shape[0], ) )
    hb = np.zeros( (df.shape[0], ) )

    for i in range( df.shape[0] ):   
        neigh =  kdtree.query_ball_point( candidate_center_xy[i], 50 )
        z_values = candidate_center_z[neigh]
        zi = candidate_center_z[i]
        zmax = np.amax(z_values)
        zmin = np.amin(z_values)
        zb[i] = np.sqrt((zi-zmin)/(zmax-zmin))
        hb[i] = zi-zmin
    
    df['Z_normalized'] = pd.Series(zb)
    df['Height_below'] = pd.Series( hb )
    
    end = time.time(); print(f"Finished Z values calculation in {round(end-start, 2)} seconds.")
    return df

# ----------------------------------------------------------------------------------------------------------------------
#
#   calculate neighborhood densities
#       \******************/
#

def density(df, neighborhoods, r):
    n_of_points = np.zeros( (df.shape[0],), dtype=int)
    for i in range(len(neighborhoods)):
        n_of_points[i] = len(neighborhoods[i])
    
    df['density_'+ r] = pd.Series(n_of_points)

    return df


# ----------------------------------------------------------------------------------------------------------------------
#
#   calculate eigen-features function
#       \******************/
#

def covariance_features(l_e, method):
    start = time.time()
    print(f"Started covariance_features for {method}...")
    l_e = np.asarray(l_e)
    l1 = 'Eigenvalue_1_' + method
    l2 = 'Eigenvalue_2_' + method
    l3 = 'Eigenvalue_3_' + method
    nx = 'Normal_X_' + method
    ny = 'Normal_Y_' + method
    nz = 'Normal_Z_' + method
    df = pd.DataFrame(data=l_e.astype(np.half), columns=[l1, l2, l3, nx, ny, nz])
    # 1. Omnivariance
    #df['Omnivariance_' + method] = np.power(df[l1] * df[l2] * df[l3], 1 / 3)
    # 2. Eigenentropy
    #df['Eigenentropy_' + method] = (-1) * ( df[l1] * np.log(df[l1]) + df[l2] * np.log(df[l2]) + df[l3] * np.log(df[l3]) )
    # 3. Anisotropy
    #df['Anisotropy_' + method] = (df[l1] - df[l3]) / df[l1]
    # 4. Planarity
    df['Planarity_' + method] = (df[l2] - df[l3]) / df[l1]
    # 5. Linearity
    if method == '4m':
        df['Linearity_' + method] = (df[l1] - df[l2]) / df[l1]
    # 6. Surface Variation / Change of Curvature
    df['Surface_Variation_' + method] = df[l3] / (df[l1] + df[l2] + df[l3])
    # 7. Sphericity
    #df['Sphericity_' + method] = df[l3] / df[l1]
    # 8. Verticality
    df['Verticality_' + method] = 1 - abs(df[nz])
    end = time.time()
    df = df.drop(df.columns[[0, 1, 2, 3, 4, 5]], axis=1)
    print(f"Finished covariance_features for {method} in {round((end - start) / 60, 2)} minutes.")
    return df

# ----------------------------------------------------------------------------------------------------------------------
#
#   calculate neighborhood features function
#       \******************/
#

def input_dataframe(pts, neighborhoods, method, df_xyz):
    start = time.time()
    print(f"Started input_dataframe for {method}...")
    
    eigen_l_e = []
    for i in neighborhoods:
        n = norm_eigen(pts, i)
        eigen_l_e.append(n)
    end = time.time()
    print(f"Finished eigen values and vectors calculation for {method} after {round((end - start) / 60, 2)} minutes.")
    df = covariance_features(eigen_l_e, method)
    end = time.time()
    
    print(f"Finished input_dataframe for {method} in {round((end - start) / 60, 2)} minutes.")
    return pd.concat([df_xyz, df], axis=1)

# ----------------------------------------------------------------------------------------------------------------------
#
#   calculate eigen values and normals
#       \******************/
#

def norm_eigen(pts, neighborhood):
    pts_n = pts[neighborhood]
    mean = np.mean(pts_n, axis=0)
    pts_n -= mean
    if (len(neighborhood) < 2):
        cov = np.zeros([3, 3])
    else:
        cov = np.cov(pts_n, rowvar=False)

    l, e = np.linalg.eigh(cov)
    idx = np.argsort(l)[::-1]
    e = e[:, idx]
    l = l[idx]
    l = softmax(l)
    eig_norm = np.zeros([6])
    eig_norm[0] = l[0]
    eig_norm[1] = l[1]
    eig_norm[2] = l[2]
    eig_norm[3] = e[0][2]
    eig_norm[4] = e[1][2]
    eig_norm[5] = e[2][2]
    return eig_norm

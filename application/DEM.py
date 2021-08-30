#
#
#      0=================================0
#      |       TIN mesh and DEM idw      |
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
#      startin TIN and pdal idw raster DEM


# ----------------------------------------------------------------------------------------------------------------------
#
#              Imports - common libs        
#       \**********************************/
#    

import numpy as np
import time
import pandas as pd
import startin
import pdal


# ----------------------------------------------------------------------------------------------------------------------
#
#             TIN mesh
#       \******************/
#
def TIN_mesh(file):

    df = pd.read_csv(file)
    
    ground_gt = df.loc[df['ground_truth'] == 2, ["X","Y", "Z"]].values
    ground_pred = df.loc[df['predictions'] == 2, ["X","Y", "Z"]].values

    df_gt = pd.DataFrame(data=ground_gt, columns=['X','Y','Z'])
    df_gt.to_csv('DEM_results/ground_gt.csv', index=False)
    df_pred = pd.DataFrame(data=ground_pred, columns=['X','Y','Z'])
    df_pred.to_csv('DEM_results/ground_pred.csv', index=False)
    
    DT = startin.DT()
    DT.insert(ground_gt)
    DT.write_obj("DEM_results/ground_gt.obj")

    DT = startin.DT()
    DT.insert(ground_pred)
    DT.write_obj("DEM_results/ground_pred.obj")

    return



# ----------------------------------------------------------------------------------------------------------------------
#
#            raster DEM
#       \******************/
#

def DEM_raster(input_files, output_files):

    size = 0.5
    rad = 1
    pwr = 2
    wnd = 25
    
    for input_file, output_file in zip(input_files, output_files):

        config = ('[\n\t"' + input_file + '",\n' +
                    '\n\t{\n\t\t"output_type": "idw"' +
                    ',\n\t\t"resolution": ' + str(size) +
                    ',\n\t\t"radius": ' + str(rad) +
                    ',\n\t\t"power": ' + str(pwr) +
                    ',\n\t\t"window_size": ' + str(wnd) +
                    ',\n\t\t"filename": "' + output_file +
                    '"\n\t}\n]') 
        pipeline = pdal.Pipeline(config)
        pipeline.execute()

    return



# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#

if __name__ == '__main__':
    start = time.time()
    
    file_path = "../data/predicted_tiles/AHN3_tile_1_2_ground_truth_predictions_error_map_model_RFC_AHN3.csv"
    TIN_mesh(file_path)

    input_files = ['DEM_results/ground_gt.csv', 'DEM_results/ground_pred.csv']
    output_files = ["DEM_results/IDW_interpolation_DTM_radius_1_power_2_pixel_05_gt.tif", "DEM_results/IDW_interpolation_DTM_radius_1_power_2_pixel_05_pred.tif"]
    DEM_raster(input_files, output_files)
    
    end = time.time()
    print(f"Script ended after {round((end - start) / 60, 2)} minutes.")

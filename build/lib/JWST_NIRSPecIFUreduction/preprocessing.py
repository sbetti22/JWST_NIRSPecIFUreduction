import numpy as np
from pathlib import Path

# The entire calwebb_detector1 pipeline
from jwst.pipeline import Detector1Pipeline

# The entire calwebb_spec2 pipeline
from jwst.pipeline import Spec2Pipeline

# JWST pipeline utilities
#from jwst import datamodels
from jwst.associations import asn_from_list
from jwst.associations.lib.rules_level3_base import DMS_Level3_Base

# The entire calwebb_spec3 pipeline
from jwst.pipeline import Spec3Pipeline

def Detector1pipeline(dir_name, output_dir_name):
    #get all the uncal files in a list
    #should only be the nrs_1 files
    uncal_files = Path(dir_name).glob('**/*nrs1_uncal.fits')
    file_list = list(map(Path, uncal_files))

    #Check the files
    print(f"Ingesting {len(file_list)} Files:\n {np.array(file_list)}")

    #make a folder for the rate files if one doesn't exist yet.
    output_dir_path = Path(output_dir_name, 'slope_images')
    output_dir_path.mkdir(parents=True, exist_ok=True)

    for file in file_list:
        #run the pipeline
        result = Detector1Pipeline.call(file, save_results=True, output_dir=str(output_dir_path))
        print("Step 1 is done.")

def Spec1pipeline(output_dir_name):
    #make a folder for the cal files if one doesn't exist yet.
    output_dir_path = Path(output_dir_name, 'cal_images')
    output_dir_path.mkdir(parents=True, exist_ok=True)

    #Get all the rate files in a list
    #path to the files.
    dir_name = Path(output_dir_name, 'slope_images')

    #get all the rate files in a list
    rate_files = dir_name.glob('*rate.fits')
    file_list = list(map(Path, rate_files))
    print(f"Ingesting {len(file_list)} Files:\n {np.array(file_list)}")

    steps=dict()
    steps['bkg_subtract'] = {'skip':True}
    steps['extract_1d'] = {'skip':False}
    steps['cube_build'] = {'coord_system':'ifualign','scalexy':0.1}

    for file in file_list:
        #run the spec2 pipeline
        result = Spec2Pipeline.call(file, output_dir = str(output_dir_path), save_results = True, steps = steps)
    print("Step 2 is done.")


def Spec3pipeline(output_dir_name):
    #make a folder for the cal files if one doesn't exist yet.
    output_dir_path = Path(output_dir_name, 'cube_images')
    output_dir_path.mkdir(parents=True, exist_ok=True)

    #Get all the rate files in a list
    #path to the files.
    dir_name = Path(output_dir_name, 'cal_images')

    #get all the cal files in a list
    cal_files = dir_name.glob('*cal.fits')
    file_list = list(map(Path, cal_files))
    print(f"Ingesting {len(file_list)} Files:\n {np.array(file_list)}")

    #create the asn file
    out_asn = asn_from_list.asn_from_list(file_list, rule = DMS_Level3_Base, product_name = "betaPic_newoutput")

    # Save the association to a json file
    asn_file = Path(dir_name, "manual_calwebb3_asn_08.json")
    with open(str(asn_file), "w") as outfile:
        _, serialized = out_asn.dump(format='json')
        outfile.write(serialized)

    #call step 3
    steps = dict()
    steps['cube_build'] = {'coord_system':'ifualign','scalexy':0.1}

    #run it
    Spec3Pipeline.call(str(asn_file), output_dir = str(output_dir_name), save_results = True, steps = steps)
    print("Step 3 is done!")

def run_preproceessing(dir_name, output_dir_name):
    Detector1pipeline(dir_name, output_dir_name)
    Spec1pipeline(output_dir_name)
    Spec3pipeline(output_dir_name)
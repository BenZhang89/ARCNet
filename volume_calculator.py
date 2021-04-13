import os
# import nibabel
import numpy as np
from util.data_process import *
import argparse
import json

parser = argparse.ArgumentParser(description='Tumor volume calculation')
parser.add_argument('--case_dir', type=str, default='',
                    help='location of the original segmented nifty file')
parser.add_argument('--verbose', type=str, default= False,
                    help='printing the volume case by case')
args = parser.parse_args()
print(args)

verbose = args.verbose
case_dir = os.path.abspath(args.case_dir) #todo change

#get case list, can be one case or many cases
cases = []
if os.path.isfile(case_dir):
    cases.append(case_dir)
if os.path.isdir(case_dir):
    for root, subFolders, case_files in os.walk(case_dir):
        cases += [os.path.join(root, case) for case in case_files]

all_volumes = {}
for each_case in cases:
    case_num = os.path.basename(each_case).split('.')[0]
    if '.nii' not in os.path.basename(each_case):
        print('Error: Input should be nifty file. {} is not a nifty file'.format(os.path.basename(each_case)))
    else:
        tumor_volume = calculate_tumor(each_case)
        all_volumes[case_num] = tumor_volume

        if verbose:
            each_predict = f"""
                The predicted tumor volume of case {case_num} is as follows:
                The edema of tumor is  {tumor_volume['edema']} {tumor_volume['unit']} 
                The edema of tumor is  {tumor_volume['enhancing']} {tumor_volume['unit']}.
                The edema of tumor is  {tumor_volume['core']} {tumor_volume['unit']}.
                The total volume of tumor is  {tumor_volume['total']} {tumor_volume['unit']}. 
                """
            print(each_predict)

with open ('volume_calculation.json', 'w') as v_json:
    json.dump(all_volumes, v_json, indent = 4)
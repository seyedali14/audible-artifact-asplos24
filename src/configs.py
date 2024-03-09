#%%
# Imports
import os
import numpy as np
import yaml
#===========================================================
#                        Constants                         #
#===========================================================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__)).strip('src')

# length of the longest trace
LARGEST_NUM_SAMPLES = 365*288 # equals to 365 days because each day has 288 samples.

#  Number of VMs per baseline based on InfoCom2019 work
NUM_VMS_PER_BASELINE = {0.1: 2251, 1.35: 100, 0.6: 310, 0.2: 1003, 0.4: 486, 0.9: 169, 0.05: 3014, 2.02: 47, 2.7: 48, 3.37: 30} # For InfoCom2019

# 95 percentile of the first/second/third moments of 2021 BVMs
first_mom_per_baseline_p95 = {3.37: 3.413611111111111, 2.7: 2.6755555555555555, 2.02: 2.1933333333333334, 1.35: 1.3188888888888888, 0.9: 0.9090909090909091, 0.6: 0.6479999999999999, 0.4: 0.7472727272727272, 0.2: 0.4827272727272728, 0.1: 0.16499999999999998, 0.05: 0.9699999999999999}
second_mom_per_baseline_p95 = {3.37: 7.484369598765426, 2.7: 4.0988213935720825, 2.02: 1.5069225216262974, 1.35: 0.7217787082950716, 0.9: 0.3229944028395062, 0.6: 0.19680987654320992, 0.4: 0.11521219618055559, 0.2: 0.037852311197916655, 0.1: 0.009180618686868675, 0.05: 0.003025}
third_mom_per_baseline_p95 = {3.37: 24.838953317901154, 2.7: 12.253631556266647, 2.02: 2.7933715884259227, 1.35: 0.9075408293552688, 0.9: 0.20755704311111112, 0.6: 0.08725837410370947, 0.4: 0.030644959287122776, 0.2: 0.006503928911608366, 0.1: 0.0008986163283779145, 0.05: 0.00023153817301097395}

# Threshold for large BVMs (exclusivee)
LARGE_BVM_BASELINE_THRESHOLD = 1.35
#===========================================================
#                     Helper Functions                     #
#===========================================================

def read_yaml(file_path):
    """
    Return content of a yaml file in file_path as a dictionary.
    """

    with open(file_path, "r") as f:
        x = yaml.safe_load(f)
        print('Setting yaml file has been read with', len(x), 'parameters')
        return x

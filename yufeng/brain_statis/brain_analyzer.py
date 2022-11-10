#!/usr/bin/env python

#================================================================
#   Copyright (C) 2022 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : brain_analyzer.py
#   Author       : Yufeng Liu
#   Date         : 2022-10-31
#   Description  : 
#
#================================================================
import os
import glob
import numpy as np
import pandas as pd

from file_io import load_image

def load_region_distr(distr_file, remove_zero=True):
    """
    Args:
        - remove_zero: remove the counts for brain_seg index 0
    """

    distr = pd.read_csv(distr_file)
    if remove_zero:
        # remove statis with id == 0
        distr = distr[distr.region != 0]

    return distr

class BrainSignalAnalyzer(object):
    def __init__(self, res_id, region_file):
        assert (res_id < 0)
        self.res_ds = np.power(2, np.fabs(res_id)-1)
        self.region_distr = load_region_distr(region_file)

    def load_mask(self, mask_file):
        self.mask = load_image(mask_file)
        
    def set_orig_dims(self, orig_dims):
        self.orig_dims = np.array(orig_dims, dtype=np.int32)

    def set_mask_dims(self, mask_dims):
        self.mask_dims = np.array(mask_dims, dtype=np.int32)

    def multiplizer_cur2mask(self):
        return 1.0 * self.orig_dims / self.res_ds / self.mask_dims

    def calc_signal_ratio(self):
        num_sig_voxels = self.region_distr['count'].sum()
        multiplizer = self.multiplizer_cur2mask()
        ratio = num_sig_voxels / np.count_nonzero(self.mask) / np.prod(multiplizer)
        return ratio

class BrainsSignalAnalyzer(object):
    def __init__(self, res_id=-3):
        self.res_id = res_id

    def calc_signal_ratio(self, distr_dir, dim_file, mask_dir):
        dim_f = pd.read_csv(dim_file, index_col='ID', sep='\t')
        for distr_file in glob.glob(os.path.join(distr_dir, '[1-9]*.csv')):
            brain_id = int(os.path.splitext(os.path.split(distr_file)[-1])[0])
            
            max_res_dims = np.array([dim_f.loc[brain_id][0],dim_f.loc[brain_id][1],dim_f.loc[brain_id][2]])
            mask_dims = np.array([dim_f.loc[brain_id][3],dim_f.loc[brain_id][4],dim_f.loc[brain_id][5]])
            bsa = BrainSignalAnalyzer(res_id=self.res_id, region_file=distr_file)
            bsa.set_orig_dims(max_res_dims)
            bsa.set_mask_dims(mask_dims)

            mask_file = os.path.join(mask_dir, f'{brain_id}.v3draw')
            bsa.load_mask(mask_file)
            ratio = bsa.calc_signal_ratio()

            print(f'{brain_id}: {ratio:.6f}')

        return sr_dict
        
        

if __name__ == '__main__':
    distr_dir = './to_del'
    dim_file = './ccf_info/TeraDownsampleMap.csv'
    mask_dir = '/PBshare/SEU-ALLEN/Users/ZhixiYun/data/registration/Inverse'
    res_id = -3

    bssa = BrainsSignalAnalyzer(res_id=res_id)
    bssa.calc_signal_ratio(distr_dir, dim_file, mask_dir)
        
        



#!/usr/bin/env python

#================================================================
#   Copyright (C) 2023 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : convert_eswc_to_swc.py
#   Author       : Yufeng Liu
#   Date         : 2023-01-03
#   Description  : 
#
#================================================================

import os
import glob
import numpy as np
import pandas as pd

from swc_handler import write_swc

def to_swc(eswc, swc):
    COL_NAMES = ['type', 'x', 'y', 'z', 'radius', 'parent', 'seg_id',
                 'level', 'mode', 'timestamp', 'teraflyindex', 'x_ccf',
                 'y_ccf', 'z_ccf', 'flag', 'bou_radius', 'bra_radius_mean',
                 'bra_radius_std', 'bou_intensity', 'bra_intensity_mean',
                 'bra_intensity_std', 'bou_density', 'pdist_to_soma',
                 'unknown', 'edist_to_soma']
    data = pd.read_csv(eswc, sep=' ', header=2, names=COL_NAMES, index_col=0)
    tree = []
    for irow, row in data.iterrows():
        node = (irow, int(row['type']), row['x_ccf']/25., row['y_ccf']/25., row['z_ccf']/25., 
                int(row['flag']), int(row['parent']))
        tree.append(node)
    #print(len(tree), tree[0])
    write_swc(tree, swc)

    
if __name__ == '__main__':
    eswc_dir = '/PBshare/SEU-ALLEN/Projects/fullNeurons/V2023_01_10/boutons'
    swc_dir = 'bouton_v20230110_swc'

    args_list = []
    for eswc_file in glob.glob(os.path.join(eswc_dir, '*eswc')):
        fn = os.path.splitext(os.path.split(eswc_file)[-1])[0]
        #print(fn)
        swc_file = os.path.join(swc_dir, f'{fn}.swc')
        if os.path.exists(swc_file):
            continue
        args_list.append((eswc_file, swc_file))

    from multiprocessing import Pool
    pool = Pool(processes=24)
    pool.starmap(to_swc, args_list)
    pool.close()
    pool.join()



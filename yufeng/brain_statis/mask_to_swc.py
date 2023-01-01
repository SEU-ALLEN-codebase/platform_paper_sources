#!/usr/bin/env python

#================================================================
#   Copyright (C) 2022 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : mask_to_swc.py
#   Author       : Yufeng Liu
#   Date         : 2022-12-31
#   Description  : 
#
#================================================================

import numpy as np
import os
import glob
import pandas as pd

from file_io import load_image

def mask2swc(fg_mask_file, swcfile, reg_dim, max_dim):
    fg_mask = load_image(fg_mask_file)[0]
    fa = max_dim / reg_dim

    coords = np.vstack(np.nonzero(fg_mask)).transpose()
    coords_max = coords * fa

    with open(swcfile, 'w') as fp:
        fp.write("#pseudo-swc by yufeng\n")
        for i, c in enumerate(coords_max):
            fp.write(f'{i+1} 2 {c[2]:.2f} {c[1]:.2f} {c[0]:.2f} 1. -1\n')
    

if __name__ == '__main__':
    sizefile = './ccf_info/TeraDownsampleSize.csv'
    mask_dir = 'statis_out_mask/fMOST-Zeng'

    df = pd.read_csv(sizefile, index_col=0)
    for mask_file in glob.glob(os.path.join(mask_dir, '*nrrd')):
        idx = os.path.split(mask_file)[-1].split('.')[0]
        print(f'--> {idx}')
        swcfile = os.path.join(mask_dir, f'{idx}.swc')

        df_loc = df.loc[idx]
        mz, my, mx, rz, ry, rx = df_loc.loc[['z_ori', 'y_ori', 'x_ori', 'z_down', 'y_down', 'x_down']]
        max_dim = np.array([mz, my, mx])
        reg_dim = np.array([rz, ry, rx])
        mask2swc(mask_file, swcfile, reg_dim, max_dim)


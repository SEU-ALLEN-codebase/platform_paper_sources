#!/usr/bin/env python

#================================================================
#   Copyright (C) 2022 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : get_region_centers.py
#   Author       : Yufeng Liu
#   Date         : 2022-11-16
#   Description  : 
#
#================================================================
import json
from scipy.spatial import distance_matrix

import numpy as np
import pandas as pd
import torch

from file_io import load_image
from anatomy.anatomy_config import MASK_CCF25_FILE
from anatomy.anatomy_core import parse_id_map, parse_ana_tree



def get_mask_hemisphere(mask_left):
    nz_left = torch.nonzero(mask_left).float()
    zcoords_left = nz_left[:,0]
    ycoords_left = nz_left[:,1]
    xcoords_left = nz_left[:,2]
    cz_left = zcoords_left.mean().item()
    cy_left = ycoords_left.mean().item()
    cx_left = xcoords_left.mean().item()
    return cx_left, cy_left, cz_left


center_file = 'region_centers_ccf25.csv'

calc_type = 'pdist' # or region
if calc_type == 'region':
    id_mapper, id_rev_mapper = parse_id_map()
    ana_dict = parse_ana_tree()

    mask = load_image(MASK_CCF25_FILE)
    mask = torch.from_numpy(mask.astype(np.int64)).cuda()
    centers = []
    for i, idx in enumerate(torch.unique(mask)):
        idx = idx.item()

        if idx == 0:
            continue

        region_name = ana_dict[idx]['acronym']

        mask_cur = mask == idx
        mask_left = mask_cur.clone()
        mask_left[mask.shape[0]//2:] = False
        mask_right = mask_cur.clone()
        mask_right[:mask.shape[0]//2] = False

        idx_left_u16, idx_right_u16 = id_rev_mapper[idx]

        cx_left, cy_left, cz_left = get_mask_hemisphere(mask_left)
        cx_right, cy_right, cz_right = get_mask_hemisphere(mask_right)

        if idx == 1051:
            centers.append([idx_right_u16, idx, cx_right, cy_right, cz_right, 1, region_name])
        else:
            centers.append([idx_left_u16, idx, cx_left, cy_left, cz_left, 0, region_name])
            centers.append([idx_right_u16, idx, cx_right, cy_right, cz_right, 1, region_name])

        print(i)

    centers_df = pd.DataFrame(centers, columns=['regionID', 'regionID_CCF', 'centerX', 'centerY', 'centerZ', 'right', 'region_name'])
    centers_df.to_csv(center_file, sep=',', float_format='%.4f', index=False)

elif calc_type == 'pdist':
    pdist_file = 'pdist_region_centers.csv'

    df = pd.read_csv(center_file, usecols=[0,2,3,4], sep=',')
    coords = df.to_numpy()[:,1:]
    pdist = distance_matrix(coords, coords)
    
    pdist = pd.DataFrame(pdist, columns=df['regionID']).set_index(df['regionID'])
    pdist.to_csv(pdist_file, sep=',', float_format='%.4f')


#!/usr/bin/env python

#================================================================
#   Copyright (C) 2023 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : sdmatrix.py
#   Author       : Yufeng Liu
#   Date         : 2023-02-02
#   Description  : 
#
#================================================================
import string
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import sys
sys.path.append('../../common_lib')
from common_utils import plot_sd_matrix, struct_dict
from config import __FEAT_ALL__


# Say, "the default sans-serif font is COMIC SANS"
matplotlib.rcParams['font.sans-serif'] = "Arial"
# Then, "ALWAYS use sans-serif fonts"
matplotlib.rcParams['font.family'] = "sans-serif"
#matplotlib.rcParams['font.weight'] = 'bold'


if 0:
    #DS matrix for all
    nodes = '500-1500'
    mefeature_file = f'../data/micro_env_features_d66_nodes{nodes}_regional.csv'
    regions = ['ACB', 'AId', 'CLA', 'CP', 'LD', 'LGd', 'LP', 'MG', 'MOp', 'MOs', 'OT', 'RSPv', 'RT', 'SMT', 'SSp-bfd', 'SSp-ll', 'SSp-m', 'SSp-n', 'SSp-ul', 'SSp-un', 'SSs', 'VISp', 'VISrl', 'VM', 'VPL', 'VPLpc', 'VPM']
    normalize = True

    df = pd.read_csv(mefeature_file, index_col=0)
    df = df[df['region_name_r316'].isin(regions)]
    # re-standarized
    keys = [f'{fn}_mean' for fn in __FEAT_ALL__]
    brain_structures = df['brain_structure']
    df = df[keys]
    if normalize:
        tmp = df.loc[:, keys]
        df.loc[:, keys] = (tmp - tmp.mean()) / (tmp.std() + 1e-10)
    corr = df.transpose().corr()
    plot_sd_matrix(brain_structures, corr, 'sdmatrix_microenviron', '')



if 1:
    # DS matrix for Cortical regions
    nodes = '500-1500'
    mefeature_file = f'../data/micro_env_features_d66_nodes{nodes}.csv'
    normalize = True
    

    df_raw = pd.read_csv(mefeature_file, index_col=0)
    for structure, regions in struct_dict.items():
        print(f'==> Processing for structure: {structure}')
        df = df_raw[df_raw['region_name_r316'].isin(regions)]
        outname = f'sdmatrix_microenviron_{structure.lower()}'

        if normalize:
            tmp = df.loc[:, __FEAT_ALL__]
            df_feat = (tmp - tmp.mean()) / (tmp.std() + 1e-10)
        else:
            df_feat = df[__FEAT_ALL__]
        print(df_feat.mean().mean())
        corr = df_feat.transpose().corr()
        structs = df['region_name_r316']
        plot_sd_matrix(structs, regions, corr, outname, '', vmin=-0.4, vmax=0.8, annot=False)



#!/usr/bin/env python

#================================================================
#   Copyright (C) 2023 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : sdmatrix.py
#   Author       : Yufeng Liu
#   Date         : 2023-01-30
#   Description  : 
#
#================================================================
import os
import sys
import numpy as np
import pandas as pd

sys.path.append('../../common_lib')
from common_utils import load_type_from_excel, stype2struct, plot_sd_matrix, struct_dict, CorticalLayers, PstypesToShow

from config import __FEAT_NAMES__, __FEAT_ALL__

__FEAT_MEAN__ = [f'{fn}_mean_mean' for fn in __FEAT_NAMES__]
__FEAT_STD__ = [f'{fn}_mean_std' for fn in __FEAT_NAMES__]
__FEAT_MEAN_STD__ = __FEAT_MEAN__ + __FEAT_STD__

def load_feature(feat_file):
    df = pd.read_csv(feat_file, index_col=1)
    tmp = df[__FEAT_MEAN_STD__]
    df.loc[:, __FEAT_MEAN_STD__] = (tmp - tmp.mean()) / (tmp.std() + 1e-10)
    return df

def calc_regional_features(df, out_dir):
    # s-type regional mean features
    rn = 'region_name_r316'
    stypes = struct_dict['CTX'] + struct_dict['TH'] + struct_dict['STR']
    mean_st = df[df.index.isin(stypes)][__FEAT_MEAN__]
    std_st = df[df.index.isin(stypes)][__FEAT_STD__]
    mean_st.rename(columns=lambda x: x.replace('mean_mean', 'mean'), inplace=True)
    std_st.rename(columns=lambda x: x.replace('mean_std', 'std'), inplace=True)
    mean_st.to_csv(os.path.join(out_dir, 'stype_mean_features_microenviron.csv'))
    std_st.to_csv(os.path.join(out_dir, 'stype_std_features_microenviron.csv'))


if __name__ == '__main__':
    feat_file = '../data/micro_env_features_d66_nodes500-1500_regional.csv'
    out_dir = '/home/lyf/Research/cloud_paper/sd_features/data'

    df = load_feature(feat_file)
    calc_regional_features(df, out_dir)
    


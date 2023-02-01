#!/usr/bin/env python

#================================================================
#   Copyright (C) 2023 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : calc_sdmatrix.py
#   Author       : Yufeng Liu
#   Date         : 2023-01-28
#   Description  : 
#
#================================================================

import numpy as np
import pickle
import pandas as pd

import sys
sys.path.append('../../common_lib')
from common_utils import stype2struct, plot_sd_matrix


def load_data(feat_files):
    df_ax = pd.read_csv(feat_files[0])
    df_ba = pd.read_csv(feat_files[1])
    df = df_ax.merge(df_ba, how='inner', on='prefix')

    include_apical = len(feat_files) == 3
    if include_apical:
        print('Include apical...')
        df_ap = pd.read_csv(feat_files[2])
        fnames = ['max_density', 'num_nodes', 'total_path_length', 'volume', 
                  'num_branches', 'dist_to_soma', 'dist_to_soma2', 'num_hubs', 
                  'variance_ratio']
        cidx = df.shape[1]
        df.loc[:, fnames] = np.zeros((df.shape[0], len(fnames)))
        df_ap_prefixs = set(df_ap['prefix'].to_numpy().tolist())
        for prefix in df['prefix']:
            if prefix in df_ap_prefixs:
                idx = np.nonzero((df['prefix'] == prefix).to_numpy())[0][0]
                idx2 = np.nonzero((df_ap['prefix'] == prefix).to_numpy())[0][0]
                df.iloc[idx, cidx:] = df_ap.iloc[idx2, -len(fnames):]

    df.set_index('region_x', inplace=True)
    df.drop(['Unnamed: 0_x', 'prefix', 'region_y', "Unnamed: 0_y"], axis=1, inplace=True)
    return df


def plot_sdmatrix_full(feat_files, figname, title, normalize=True):
    df = load_data(feat_files)
    structs = [stype2struct[region] for region in df.index]
    df.reset_index(inplace=True)
    df.drop('region_x', axis=1, inplace=True)
    
    if normalize:
        df = (df - df.mean()) / (df.std() + 1e-10)

    corr = df.transpose().corr()
    plot_sd_matrix(structs, corr, figname, title)
        
def plot_sdmatrix_ctx(feat_files, figname, title, normalize=True):
    regions = ['ACB', 'AId', 'CLA', 'MOp', 'MOs', 'RSPv', 'SSp-bfd', 'SSp-ll', 'SSp-m', 'SSp-n', 'SSp-ul', 'SSp-un', 'SSs', 'VISp', 'VISrl']
    df = load_data(feat_files)
    df = df[df.index.isin(regions)]

    structs = [region for region in df.index]
    df.reset_index(inplace=True)
    df.drop('region_x', axis=1, inplace=True)
    
    if normalize:
        df = (df - df.mean()) / (df.std() + 1e-10)

    corr = df.transpose().corr()
    plot_sd_matrix(structs, corr, figname, title)
   

if __name__ == '__main__':
   
    # full morph
    feat_files = ['min_num_neurons10_l2/features_r2_somaTypes_axonal.csv',
                  'min_num_neurons10_l2/features_r2_somaTypes_basal.csv', 
                  'min_num_neurons10_l2/features_r2_somaTypes_apical.csv']

    if 0:
        figname = 'sdmatrix_arbors.png'
        title = ''
        
        plot_sdmatrix_full(feat_files, figname, title, True)

    if 1:
        figname = 'sdmatrix_arbors_ctx.png'
        title = ''
        
        plot_sdmatrix_ctx(feat_files, figname, title, True)


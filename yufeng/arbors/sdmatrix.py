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
from common_utils import stype2struct, plot_sd_matrix, struct_dict, CorticalLayers, PstypesToShow


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
    df.drop(['Unnamed: 0_x', 'region_y', "Unnamed: 0_y"], axis=1, inplace=True)
    return df


def plot_sdmatrix_full(feat_files, figname, title, normalize=True):
    df = load_data(feat_files)
    structs = [stype2struct[region] for region in df.index]
    df.reset_index(inplace=True)
    df.drop(['region_x', 'prefix'], axis=1, inplace=True)
    
    if normalize:
        df = (df - df.mean()) / (df.std() + 1e-10)

    corr = df.transpose().corr()
    plot_sd_matrix(structs, corr, figname, title)
        
def plot_sdmatrix_struct(feat_files, figname, regions, title, normalize=True, vmin=-0.4, vmax=0.8, annot=False):
    df = load_data(feat_files)
    df = df[df.index.isin(regions)]

    structs = [region for region in df.index]
    df.reset_index(inplace=True)
    df.drop(['region_x', 'prefix'], axis=1, inplace=True)
    
    if normalize:
        df = (df - df.mean()) / (df.std() + 1e-10)

    corr = df.transpose().corr()
    plot_sd_matrix(structs, regions, corr, figname, title, vmin=vmin, vmax=vmax, annot=annot)

def plot_sdmatrix_struct_with_cortical_layer(feat_files, celltype_file, figname, regions, title, normalize=True, vmin=-0.4, vmax=0.8, annot=False):
    df = load_data(feat_files)
    df = df[df.index.isin(regions)]

    df_ct = pd.read_csv(celltype_file, index_col=0, usecols=('Cell name', 'Manually_corrected_soma_region', 'Cortical_layer'))
    df = df.merge(df_ct, how='inner', left_on='prefix', right_on='Cell name')
    print(df.shape)

    cstype = []
    for stype, cl in zip(df.Manually_corrected_soma_region, df.Cortical_layer):
        if cl is np.NaN:
            cl = ''
        else:
            cl = f'-{cl}'
        cstype.append(f'{stype}{cl}')
    df['cstype'] = cstype

    # remove with
    #rs, counts = np.unique(cstype, return_counts=True)
    #rs = rs[counts >= 5]
    #rs = sorted(rs, key=lambda x: x.split('-')[-1])
    rs = CorticalLayers
    df = df[df.cstype.isin(rs)]

    structs = [region for region in cstype if region in rs]
    df.reset_index(inplace=True)
    df.drop(['Manually_corrected_soma_region', 'Cortical_layer', 'prefix', 'cstype'], axis=1, inplace=True)
    
    if normalize:
        df = (df - df.mean()) / (df.std() + 1e-10)

    corr = df.transpose().corr()
    print(rs)
    plot_sd_matrix(structs, rs, corr, figname, title, vmin=vmin, vmax=vmax, annot=annot)
   
def plot_sdmatrix_struct_with_ptype(feat_files, celltype_file, figname, regions, title, normalize=True, vmin=-0.4, vmax=0.8, annot=False):
    def sort_lambda(x):
        sp = x.split('-')
        if len(sp) == 1:
            return f'1_{x}'
        else:
            return f'0_{sp[-1]}_{x}'

    df = load_data(feat_files)
    df_ct = pd.read_csv(celltype_file, index_col=0, usecols=('Cell name', 'Manually_corrected_soma_region', 'Subclass_or_type'))
    df = df.merge(df_ct, how='inner', left_on='prefix', right_on='Cell name')
    print(df.shape)

    ptypes = []
    for stype, pt in zip(df.Manually_corrected_soma_region, df.Subclass_or_type):
        if pt is np.NaN:
            pt = ''
        else:
            pt = pt.split('_')[-1]
            pt = f'-{pt}'
        ptypes.append(f'{stype}{pt}')
    df['ptype'] = ptypes

    # remove CP_others
    df = df[df['ptype'].isin(regions)]

    # remove with
    #rs, counts = np.unique(ptypes, return_counts=True)
    #rs = rs[counts >= 5]
    #rs = sorted(rs, key=sort_lambda)
    #df = df[df.ptype.isin(rs)]

    structs = [region for region in ptypes if region in regions]
    df.reset_index(inplace=True)
    df.drop(['Manually_corrected_soma_region', 'Subclass_or_type', 'prefix', 'ptype'], axis=1, inplace=True)
    
    if normalize:
        df = (df - df.mean()) / (df.std() + 1e-10)

    corr = df.transpose().corr()
    plot_sd_matrix(structs, regions, corr, figname, title, vmin=vmin, vmax=vmax, annot=annot)

if __name__ == '__main__':
   
    # full morph
    feat_files = ['min_num_neurons10_l2/features_r2_somaTypes_axonal.csv',
                  'min_num_neurons10_l2/features_r2_somaTypes_basal.csv', 
                  'min_num_neurons10_l2/features_r2_somaTypes_apical.csv']
    celltype_file = '../../common_lib/41586_2021_3941_MOESM4_ESM.csv'

    if 0:
        figname = 'sdmatrix_arbors'
        title = ''
        
        plot_sdmatrix_full(feat_files, figname, title, True)

    if 1:
        for structure, regions in struct_dict.items():
            #figname = f'sdmatrix_arbor_{structure.lower()}'
            #plot_sdmatrix_struct(feat_files, figname, regions, '', True, vmin=-0.4, vmax=0.8, annot=False)

            #if structure == 'CTX':
            #    figname = f'sdmatrix_arbor_{structure.lower()}_withLayer'
            #    plot_sdmatrix_struct_with_cortical_layer(feat_files, celltype_file, figname, regions, '', True, vmin=-0.4, vmax=0.8, annot=False)

            figname = f'sdmatrix_arbor_{structure.lower()}_withPtype'
            plot_sdmatrix_struct_with_ptype(feat_files, celltype_file, figname, PstypesToShow[structure], '', True, vmin=-0.4, vmax=0.8, annot=False)
                


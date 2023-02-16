#!/usr/bin/env python

#================================================================
#   Copyright (C) 2023 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : distance_vs_correlation.py
#   Author       : Yufeng Liu
#   Date         : 2023-02-15
#   Description  : 
#
#================================================================

import os
import sys
import random
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

__NUM_BINS__ = 25
__LABEL_FONTS__ = 18

def load_spos_corr(spos_file, df_corr, is_spos=True):
    df_spos = pd.read_csv(spos_file, index_col=0)
    if is_spos:
        # to 1um resolution
        df_spos.loc[:, ['x_pos', 'y_pos', 'z_pos']] *= 0.04
    else:
        df_spos.loc[:] *= 0.001

    # keep only the target neurons
    df_spos = df_spos.loc[df_corr.index]
    
    pdists = distance_matrix(df_spos, df_spos)
    indices = np.triu_indices_from(pdists)
    dists = pdists[indices]
    corrs = df_corr.to_numpy()[indices]

    return dists, corrs

def soma_dist_vs_corr(spos_file, corr_file):
    df_corr = pd.read_csv(corr_file, index_col=0)
    df_corr.drop(['type'], axis=1, inplace=True)
    dists, corrs = load_spos_corr(spos_file, df_corr)    

    nsample = 10000
    sample_indices = random.sample(range(len(dists)), nsample)
    dists_sample = dists[sample_indices]
    corrs_sample = corrs[sample_indices]

    data = np.array([dists_sample, corrs_sample]).transpose().astype(float)
    #data = np.array([dists, corrs]).transpose().astype(float)

    xname = 'Soma-soma distance'
    yname = 'Correlation'
    df = pd.DataFrame(data, columns=[xname, yname])
    glm = sns.lmplot(data=df, x=xname, y=yname,
               scatter_kws={'s':.5, 'color':'green'}, fit_reg=False
    )

    # show mean average
    bin_means, bin_edges, binnumber = stats.binned_statistic(dists, corrs,
        statistic='mean', bins=__NUM_BINS__)
    bin_stds, bin_edges, binnumber = stats.binned_statistic(dists, corrs,
        statistic='std', bins=__NUM_BINS__)
    bin_width = (bin_edges[1] - bin_edges[0])
    bin_centers = bin_edges[1:] - bin_width/2
    plt.plot(bin_centers, bin_means, color='blue')
    plt.fill_between(bin_centers, bin_means-bin_stds, bin_means+bin_stds, color='blue', alpha=0.2)
    
    plt.xlim(0, 10)
    plt.xlabel(f'{xname} (mm)', fontsize=__LABEL_FONTS__)
    plt.ylabel(f'{yname}', fontsize=__LABEL_FONTS__)

    plt.savefig('soma_distance_vs_correlation.png', dpi=300)
    plt.close('all')
 
def proj_dist_vs_corr(proj_file, corr_file):
    df_corr = pd.read_csv(corr_file, index_col=0)
    df_corr.drop(['type'], axis=1, inplace=True)
    dists, corrs = load_spos_corr(proj_file, df_corr, is_spos=False)

    nsample = 10000
    sample_indices = random.sample(range(len(dists)), nsample)
    dists_sample = dists[sample_indices]
    corrs_sample = corrs[sample_indices]

    data = np.array([dists_sample, corrs_sample]).transpose().astype(float)
    #data = np.array([dists, corrs]).transpose().astype(float)

    xname = 'Axon-axon distance'
    yname = 'Correlation'
    df = pd.DataFrame(data, columns=[xname, yname])
    glm = sns.lmplot(data=df, x=xname, y=yname,
               scatter_kws={'s':.5, 'color':'green'}, fit_reg=False
    )

    # show mean average
    bin_means, bin_edges, binnumber = stats.binned_statistic(dists, corrs,
        statistic='mean', bins=__NUM_BINS__)
    bin_stds, bin_edges, binnumber = stats.binned_statistic(dists, corrs,
        statistic='std', bins=__NUM_BINS__)
    bin_width = (bin_edges[1] - bin_edges[0])
    bin_centers = bin_edges[1:] - bin_width/2
    plt.plot(bin_centers, bin_means, color='blue')
    plt.fill_between(bin_centers, bin_means-bin_stds, bin_means+bin_stds, color='blue', alpha=0.2)
    
    plt.xlim(0, 60)
    plt.xlabel(f'{xname} (mm)', fontsize=__LABEL_FONTS__)
    plt.ylabel(f'{yname}', fontsize=__LABEL_FONTS__)

    plt.savefig('axon_distance_vs_correlation.png', dpi=300)
    plt.close('all')

def soma_proj_dist_vs_corr(spos_file, proj_file, corr_file):
    df_corr = pd.read_csv(corr_file, index_col=0)
    df_corr.drop(['type'], axis=1, inplace=True)
    spos_dists, _ = load_spos_corr(spos_file, df_corr, is_spos=True)
    proj_dists, corrs = load_spos_corr(proj_file, df_corr, is_spos=False)

    dists = spos_dists + proj_dists / 5.    # reweight for balance

    nsample = 10000
    sample_indices = random.sample(range(len(dists)), nsample)
    dists_sample = dists[sample_indices]
    corrs_sample = corrs[sample_indices]

    data = np.array([dists_sample, corrs_sample]).transpose().astype(float)
    #data = np.array([dists, corrs]).transpose().astype(float)

    xname = 'Neuron-neuron distance'
    yname = 'Correlation'
    df = pd.DataFrame(data, columns=[xname, yname])
    glm = sns.lmplot(data=df, x=xname, y=yname,
               scatter_kws={'s':.5, 'color':'green'}, fit_reg=False
    )

    # show mean average
    bin_means, bin_edges, binnumber = stats.binned_statistic(dists, corrs,
        statistic='mean', bins=__NUM_BINS__)
    bin_stds, bin_edges, binnumber = stats.binned_statistic(dists, corrs,
        statistic='std', bins=__NUM_BINS__)
    bin_width = (bin_edges[1] - bin_edges[0])
    bin_centers = bin_edges[1:] - bin_width/2
    plt.plot(bin_centers, bin_means, color='blue')
    plt.fill_between(bin_centers, bin_means-bin_stds, bin_means+bin_stds, color='blue', alpha=0.2)
    
    plt.xlim(0, 20)
    plt.xlabel(f'{xname} (mm)', fontsize=__LABEL_FONTS__)
    plt.ylabel(f'{yname}', fontsize=__LABEL_FONTS__)

    plt.savefig('neuron_distance_vs_correlation.png', dpi=300)
    plt.close('all')
   

if __name__ == '__main__':
    spos_file = '../common_lib/misc/soma_pos_1891_v20230110.csv'
    proj_file = '../common_lib/41586_2021_3941_MOESM4_ESM_proj.csv'
    corr_file = './multi-scale/corr_neuronLevel_sdmatrix_heatmap_stype_all.csv'

    soma_dist_vs_corr(spos_file, corr_file)
    proj_dist_vs_corr(proj_file, corr_file)
    soma_proj_dist_vs_corr(spos_file, proj_file, corr_file)
    
    
    



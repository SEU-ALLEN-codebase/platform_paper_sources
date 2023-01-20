#!/usr/bin/env python

#================================================================
#   Copyright (C) 2023 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : visualize.py
#   Author       : Yufeng Liu
#   Date         : 2023-01-19
#   Description  : 
#
#================================================================

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from config import __FEAT_NAMES__, __FEAT_ALL__

if 0:
    # Visualize the overall node distribution, and test the diversity and 
    # convergence among brain regions
    import umap

    mefeature_file = '../data/micro_env_features_d66_all.csv'
    classes = ['MOp', 'MOs', 'VPM', 'VPL', 'CP']
    kname = 'region_name_r316'
    n_comp = 5

    print('Loading the data')
    df = pd.read_csv(mefeature_file)
    # plot the node distribution
    #sns.displot(data=df, x='Nodes', kind='hist')
    #plt.savefig('nodes_distr_all.png', dpi=200)
    #plt.close('all')

    df_vis = df[df[kname].isin(classes)]
    print(f'>> Number samples before removal: {df_vis.shape[0]}')
    df_vis = df_vis[(df_vis['Nodes'] > 300) & (df_vis['Nodes'] < 1500)]
    print(f'>> Number samples after removal: {df_vis.shape[0]}')
    
    # map to low resolution
    print(f'Do umap for the data')
    reducer = umap.UMAP(n_components=n_comp)
    embedding = reducer.fit_transform(df_vis[__FEAT_ALL__].to_numpy())
    # visualization
    print('Visualization')
    vnames = [f'umap_{i}' for i in range(n_comp)]
    df_vis.loc[:,vnames] = embedding
    g = sns.pairplot(df_vis, vars=vnames, hue=kname, plot_kws={'s':10})

    plt.savefig('mefeature_umap_example.png', dpi=200)


if 1:
    # Plot the regional similarity
    rmefeature_file = '../data/micro_env_features_d66_nodes300-1500_regional.csv'
    mean_feat_names = [f'{fn}_mean' for fn in __FEAT_ALL__]
    
    rmef = pd.read_csv(rmefeature_file, index_col=0)
    rmef_hm = rmef[['region_name_r316', *mean_feat_names]]
    rmef_hm.set_index('region_name_r316', inplace=True)
    rmef_hm = rmef_hm.transpose()
    # clip
    #rmef_hm.clip(-2, 2, inplace=True)
    
    rmef_hm.reset_index(inplace=True, drop=True)
    #sns.clustermap(rmef_hm)
    corr = rmef_hm.corr()
    corr[corr > 0.8] = 0.8
    sns.clustermap(corr, cmap='coolwarm')
    plt.savefig('temp.png', dpi=200)
    



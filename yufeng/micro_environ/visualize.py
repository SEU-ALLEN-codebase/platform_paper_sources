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
import matplotlib
import matplotlib.pyplot as plt

from config import __FEAT_NAMES__, __FEAT_ALL__

# Say, "the default sans-serif font is COMIC SANS"
matplotlib.rcParams['font.sans-serif'] = "Arial"
# Then, "ALWAYS use sans-serif fonts"
matplotlib.rcParams['font.family'] = "sans-serif"
#matplotlib.rcParams['font.weight'] = 'bold'

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
    def mean_clustermap(rmef_hm, figname='temp.png'):
        rmef_hm.set_index('region_name_r316', inplace=True)
        brain_structures = rmef_hm.pop('brain_structure')
        rmef_hm = rmef_hm.transpose()
        #rmef_hm.clip(-2, 2, inplace=True)
        
        rmef_hm.reset_index(inplace=True, drop=True)
        corr = rmef_hm.corr()
        corr[corr > 0.9] = 0.9

        lut = dict(zip(np.unique(brain_structures), "rbgy"))
        row_colors = brain_structures.map(lut)
        #print(row_colors)

        cm = sns.clustermap(data=corr, cmap='coolwarm', xticklabels=1, yticklabels=1,
                row_colors=row_colors)

        # change the ticklabels
        cm.ax_heatmap.set_title('mean-features', fontsize=25)
        cm.ax_heatmap.set_xlabel('Regions', fontdict={'fontsize': 18})
        cm.ax_heatmap.set_ylabel('', fontdict={'fontsize': 18})
        tlabels = corr.columns[cm.dendrogram_col.reordered_ind]
        x_tlabels = []
        y_tlabels = []
        for i in range(len(tlabels)):
            if i % 2 == 0:
                y_tlabels.append(f'----------{tlabels[i]}')
                x_tlabels.append(f'{tlabels[i]}----------')
            else:
                x_tlabels.append(tlabels[i])
                y_tlabels.append(tlabels[i])

        cm.ax_heatmap.set_xticklabels(x_tlabels)
        cm.ax_heatmap.set_yticklabels(y_tlabels)
        plt.setp(cm.ax_heatmap.yaxis.get_majorticklabels(), fontsize=8)
        plt.setp(cm.ax_heatmap.xaxis.get_majorticklabels(), fontsize=8)

        cm.cax.set_visible(False)
        cm.ax_row_dendrogram.set_visible(False)
        cm.ax_col_dendrogram.set_visible(False)
        plt.tight_layout()
        plt.savefig(figname, dpi=600)
        plt.close('all')

        return tlabels

    def std_heatmap(rmef_hm, figname='temp.png', defined_order=None):
        rmef_hm.set_index('region_name_r316', inplace=True)
        brain_structures = rmef_hm.pop('brain_structure')
        rmef_hm = rmef_hm.transpose()
        #rmef_hm.clip(-2, 2, inplace=True)
        
        rmef_hm.reset_index(inplace=True, drop=True)
        #import ipdb; ipdb.set_trace()
        rmef_hm = rmef_hm[defined_order]
        corr = rmef_hm.corr()
        corr[corr > 0.9] = 0.9

        fig, ax_heatmap = plt.subplots(figsize=(8,9.5))
        sns.heatmap(data=corr, ax=ax_heatmap, cmap='coolwarm', 
            xticklabels=1, yticklabels=False, cbar=False)

        # change the ticklabels
        ax_heatmap.set_title('std-features', fontsize=25)
        ax_heatmap.set_xlabel('Regions', fontdict={'fontsize': 18})
        ax_heatmap.set_ylabel('', fontdict={'fontsize': 18})
        x_tlabels = []
        y_tlabels = []
        for i in range(len(defined_order)):
            if i % 2 == 0:
                y_tlabels.append(f'----------{defined_order[i]}')
                x_tlabels.append(f'{defined_order[i]}----------')
            else:
                x_tlabels.append(defined_order[i])
                y_tlabels.append(defined_order[i])

        ax_heatmap.set_xticklabels(x_tlabels)
        #ax_heatmap.set_yticklabels(y_tlabels)
        plt.setp(ax_heatmap.yaxis.get_majorticklabels(), fontsize=8)
        #plt.setp(ax_heatmap.xaxis.get_majorticklabels(), fontsize=8)

        plt.tight_layout()
        plt.savefig(figname, dpi=600)

        return defined_order


    # Plot the regional similarity
    nodes = '500-1500'
    rmefeature_file = f'../data/micro_env_features_d66_nodes{nodes}_regional.csv'
    std_normalize = False # normalize by std

    mean_feat_names = [f'{fn}_mean' for fn in __FEAT_ALL__]
    std_feat_names = [f'{fn}_std' for fn in __FEAT_ALL__]
    
    rmef = pd.read_csv(rmefeature_file, index_col=0)
    rmef_hm = rmef[['region_name_r316', 'brain_structure', *mean_feat_names]]
    if std_normalize:
        rmef_hm.loc[:, mean_feat_names] = rmef.loc[:,mean_feat_names].to_numpy() / (rmef.loc[:,std_feat_names].to_numpy() + 1e-10)
    rmef_hm_std = rmef[['region_name_r316', 'brain_structure', *std_feat_names]]

    orders = mean_clustermap(rmef_hm, figname=f'corr_clustermap_mean_nodes{nodes}.png')
    orders = std_heatmap(rmef_hm_std, figname=f'corr_clustermap_std_nodes{nodes}.png', 
        defined_order=orders)
    



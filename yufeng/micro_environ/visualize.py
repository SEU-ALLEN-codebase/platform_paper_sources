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
from scipy.spatial import distance_matrix

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


# Heatmap & clustermap of whole-train correlation coefficients
if 1:
    def plot_sd_matrix(brain_structures, corr_raw, figname):
        fig, ax_sd = plt.subplots(figsize=(6,6))
        structs = np.unique(brain_structures)
        nstructs = len(structs)
        sd_matrix = np.zeros((nstructs, nstructs))
        for i in range(nstructs):
            struct1 = corr_raw.index[brain_structures == structs[i]]
            for j in range(i, nstructs):
                struct2 = corr_raw.index[brain_structures == structs[j]]
                cc = corr_raw.loc[struct1, struct2].values.mean()
                sd_matrix[i][j] = cc
                sd_matrix[j][i] = cc
        df_sd = pd.DataFrame(sd_matrix, columns=structs, index=structs)
        sns.heatmap(data=df_sd, ax=ax_sd, cmap='coolwarm', annot=True, 
                    annot_kws={"size": 25}, cbar=False)
        ax_sd.set_title('SD matrix', fontsize=40)
        ax_sd.set_xlabel('', fontdict={'fontsize': 18})
        ax_sd.set_ylabel('', fontdict={'fontsize': 18})
        plt.setp(ax_sd.xaxis.get_majorticklabels(), fontsize=25)
        plt.setp(ax_sd.yaxis.get_majorticklabels(), fontsize=25)
        plt.savefig(figname, dpi=300)
        plt.close('all')

    def mean_clustermap(rmef_hm, figname='temp.png'):
        rmef_hm = rmef_hm.copy()
        brain_structures = rmef_hm.pop('brain_structure')
        rmef_hm = rmef_hm.transpose()
        #rmef_hm.clip(-2, 2, inplace=True)
        
        rmef_hm.reset_index(inplace=True, drop=True)
        corr = rmef_hm.corr()
        corr_raw = corr.copy()
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
        #cm.ax_row_dendrogram.set_visible(False)
        cm.ax_col_dendrogram.set_visible(False)
        plt.tight_layout()
        plt.savefig(figname, dpi=300)
        plt.close('all')

        # quantitative estimation of diversity and stereotypy
        plot_sd_matrix(brain_structures, corr_raw, f'{figname[:-4]}_sd.png')

        return tlabels

    def std_heatmap(rmef_hm, figname='temp.png', defined_order=None):
        rmef_hm = rmef_hm.copy()
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
        plt.savefig(figname, dpi=300)
        plt.close('all')

        return defined_order

    def plot_regional_pdist(center_file, defined_order):
        df_c = pd.read_csv(center_file)
        df_c = df_c[df_c['right'] == 0]
        df_c.set_index('region_name', inplace=True)
        df_c = df_c.loc[defined_order, ['centerX', 'centerY', 'centerZ']] * 25.
        pdist = distance_matrix(df_c, df_c)

        fig, ax_heatmap = plt.subplots(figsize=(8,9.5))
        ax_heatmap = sns.heatmap(data=pdist, ax=ax_heatmap, cmap='coolwarm_r', 
            xticklabels=1, yticklabels=False, cbar=False)

        # change the ticklabels
        ax_heatmap.set_title('spatial distance', fontsize=25)
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
        plt.savefig('spatial_distance.png', dpi=300)
        plt.close('all')

    def plot_spatial_enhanced_corr(rmef_hm, center_file, figname):
        rmef_hm = rmef_hm.copy()
        brain_structures = rmef_hm.pop('brain_structure')
        rmef_hm = rmef_hm.transpose()
        #rmef_hm.clip(-2, 2, inplace=True)

        rmef_hm.reset_index(inplace=True, drop=True)
        corr = rmef_hm.corr()
        #corr[corr > 0.9] = 0.9

        # load the position
        df_c = pd.read_csv(center_file)
        df_c = df_c[df_c['right'] == 0]
        df_c.set_index('region_name', inplace=True)
        df_c = df_c.loc[corr.columns, ['centerX', 'centerY', 'centerZ']] * 25.

        # enhance by pdist
        pdist = distance_matrix(df_c, df_c)
        pdist /= pdist.max()
        epdist = np.exp(-pdist)
        corr = epdist * corr
        corr_raw = corr.copy()
        #corr[corr > 0.7] = 0.7
        #corr[corr < 0] = 0

        lut = dict(zip(np.unique(brain_structures), "rbgy"))
        row_colors = brain_structures.map(lut)
        cm = sns.clustermap(data=corr, cmap='coolwarm', xticklabels=1, yticklabels=1, row_colors=row_colors)

        # change the ticklabels
        cm.ax_heatmap.set_title('spatial_enhanced-mean-features', fontsize=25)
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
        #cm.ax_row_dendrogram.set_visible(False)
        cm.ax_col_dendrogram.set_visible(False)
        plt.tight_layout()
        plt.savefig(figname, dpi=300)
        plt.close('all')

        plot_sd_matrix(brain_structures, corr_raw, f'{figname[:-4]}_sd.png')


        
    # Plot the regional similarity
    nodes = '500-1500'
    rmefeature_file = f'../data/micro_env_features_d66_nodes{nodes}_regional.csv'
    std_normalize = False # normalize by std
    center_file = '../../brain_statistics/region_centers/region_centers_ccf25_r316.csv'

    mean_feat_names = [f'{fn}_mean' for fn in __FEAT_ALL__]
    std_feat_names = [f'{fn}_std' for fn in __FEAT_ALL__]
    
    rmef = pd.read_csv(rmefeature_file, index_col=0)
    rmef_hm = rmef[['region_name_r316', 'brain_structure', *mean_feat_names]]
    rmef_hm.set_index('region_name_r316', inplace=True)
    
    if std_normalize:
        rmef_hm.loc[:, mean_feat_names] = rmef.loc[:,mean_feat_names].to_numpy() / (rmef.loc[:,std_feat_names].to_numpy() + 1e-10)
    rmef_hm_std = rmef[['region_name_r316', 'brain_structure', *std_feat_names]]
    rmef_hm_std.set_index('region_name_r316', inplace=True)

    orders = mean_clustermap(rmef_hm, figname=f'corr_clustermap_mean_nodes{nodes}.png')
    plot_spatial_enhanced_corr(rmef_hm, center_file, figname=f'corr_clustermap_mean_nodes{nodes}_spatialEnhanced.png')
    orders = std_heatmap(rmef_hm_std, figname=f'corr_clustermap_std_nodes{nodes}.png', 
        defined_order=orders)
    
# feature distribution 
if 0:
    nodes = '500-1500'
    rmefeature_file = f'../data/micro_env_features_d66_nodes{nodes}_regional.csv'
    #rmefeature_file = 'non-environ_features_nodes500-1500.csv'
    region = 'CP'
    
    vis_features = ['Stems', 'Bifurcations', 'Branches', 'Tips', 'Length', 'Volume',
              'MaxEuclideanDistance', 'MaxPathDistance', 'MaxBranchOrder',
              'AverageContraction', 'AverageFragmentation', 'AverageBifurcationAngleLocal',
              'AverageBifurcationAngleRemote']
    fnames = [f'{fn}_mean_mean' for fn in vis_features]
    #fnames = __FEAT_NAMES__
    
    df = pd.read_csv(rmefeature_file, index_col=0).set_index('region_name_r316')
    rcoords = pd.DataFrame(list(zip(range(len(fnames)), df.loc[region, fnames])), columns=['Region', 'feature'])
    
    
    fs = list(zip(np.repeat(df.index.to_list(), df.shape[0]), 
                fnames * df.shape[0], 
                df.loc[:,fnames].to_numpy().reshape(-1)))
    df_f = pd.DataFrame(data=fs, columns=['Region', 'feature_name', 'feature'])
    #sns.violinplot(data=df_f, x='feature_name', y='feature')
    sns.boxplot(data=df_f, x='feature_name', y='feature', showfliers=False)

    sns.scatterplot(data=rcoords, x='Region', y='feature', marker='o', color='red', label=region)
    plt.xticks(ticks=range(len(fnames)), labels=vis_features, rotation=90, fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('')
    plt.ylabel('mean-features', fontsize=18)
    plt.ylim(-1.6, 1.6)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{region}_feature_distr.png', dpi=300)
    plt.close('all')



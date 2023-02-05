#!/usr/bin/env python

#================================================================
#   Copyright (C) 2023 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : plot_features.py
#   Author       : Yufeng Liu
#   Date         : 2023-02-03
#   Description  : 
#
#================================================================
import os
import sys
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

sys.path.append('../../common_lib')
from common_utils import struct_dict, CorticalLayers, PstypesToShow

def rename_func(x, key=''):
    x = x.replace('_', ' ').split(' ')
    x = ''.join([xi[0].upper() + xi[1:] for xi in x])
    
    return x + key

def rename_mean(x):
    return rename_func(x, 'Mean')

def rename_std(x):
    return rename_func(x, 'Std')

def get_stype_categories():
    index = struct_dict['CTX'] + struct_dict['TH'] + struct_dict['STR']
    cgs = ['CTX' for i in range(len(struct_dict['CTX']))] + \
          ['TH' for i in range(len(struct_dict['TH']))] + \
          ['STR' for i in range(len(struct_dict['STR']))]
    return pd.Series(cgs, name='s-type', index=index)

def get_ptype_categories():
    index = PstypesToShow['CTX'] + PstypesToShow['TH'] + PstypesToShow['STR']
    cgs = ['CTX-'+ptype.split('-')[-1] for ptype in PstypesToShow['CTX']] + \
          ['TH-'+ptype.split('-')[-1] for ptype in PstypesToShow['TH']] + \
          ['STR-'+ptype.split('-')[-1] for ptype in PstypesToShow['STR']]
    return pd.Series(cgs, name='p-type', index=index)

def get_cstype_categories():
    index = CorticalLayers
    cgs = [cst.split('-')[-1] for cst in index]
    return pd.Series(cgs, name='cs-type', index=index)


levels = ['microenviron', 'fullMorpho', 'arbor', 'motif', 'bouton']
#levels = ['fullMorpho', 'arbor', 'motif', 'bouton']
data_dir = '../data'
use_std_feature = False
vmin = -2
vmax = 2
do_prominence_estimation = True
do_plot = True

#colors = ['orangered', 'gold', 'medianblue', 'lime', 'magenta', 'brown']
colors = ['b', 'r', 'k', 'm', 'c', 'g', 'y']
rcolors_dict = dict(zip(levels, colors[-len(levels):]))

for ct in ['stype', 'ptype', 'cstype']:
    print(f'--> {ct}')
    df_feats = pd.DataFrame([])
    ltypes = []
    for level in levels:
        print(f'level: {level}')
        mfile = os.path.join(data_dir, f'{ct}_mean_features_{level}.csv')
        sfile = os.path.join(data_dir, f'{ct}_std_features_{level}.csv')
        if (not os.path.exists(mfile)) or (not os.path.exists(sfile)):
            continue
        if use_std_feature:
            if level == 'microenviron':
                mean_feat = pd.read_csv(mfile, index_col=0).rename(columns=rename_func)
                std_feat = pd.read_csv(sfile, index_col=0).rename(columns=rename_func)
            else:
                mean_feat = pd.read_csv(mfile, index_col=0).rename(columns=rename_mean)
                std_feat = pd.read_csv(sfile, index_col=0).rename(columns=rename_std)
            feats = pd.concat([mean_feat, std_feat], axis=1)
        else:
            feats = pd.read_csv(mfile, index_col=0).rename(columns=rename_func)

        df_feats = pd.concat([df_feats, feats], axis=1)
        ltypes.extend([level for i in range(feats.shape[1])])

    df_feats = df_feats.transpose()
    # customize row colors
    ltypes = pd.Series(ltypes, name='Level', index=df_feats.index)
    row_colors = ltypes.map(rcolors_dict)
    # col colors
    if ct == 'stype':
        ctypes = get_stype_categories()
        lut = dict(zip(np.unique(ctypes), sns.hls_palette(len(np.unique(ctypes)), l=0.5, s=0.8)))
        col_colors = ctypes.map(lut)
        ct_label = 's-type'
    elif ct == 'ptype':
        ctypes = get_ptype_categories()
        lut = dict(zip(np.unique(ctypes), sns.hls_palette(len(np.unique(ctypes)), l=0.5, s=0.8)))
        col_colors = ctypes.map(lut)
        ct_label = 'p-type'
    elif ct == 'cstype':
        ctypes = get_cstype_categories()
        lut = dict(zip(np.unique(ctypes), sns.hls_palette(len(np.unique(ctypes)), l=0.5, s=0.8)))
        col_colors = ctypes.map(lut)
        ct_label = 's-type-layer'
    else:
        raise ValueError

    #------------- The following section is for visualization -----------------#
    if do_plot:
        print('===> Plotting feature map')
        df_feats.clip(vmin, vmax, inplace=True)
        print(row_colors.shape, col_colors.shape, df_feats.shape)
        g2 = sns.clustermap(df_feats, cmap='coolwarm', row_colors=row_colors, 
                            col_colors=col_colors, row_cluster=False, 
                            col_cluster=True, vmin=vmin, vmax=vmax,
                            xticklabels=1, yticklabels=1, 
                            figsize=(10,20))
        # Move the row colors to right
        #ax_row_colors = cm.ax_row_colors
        #box_row_colors = ax_row_colors.get_position()
        #box_heatmap = cm.ax_heatmap.get_position()
        #ax_row_colors.set_position([box_heatmap.max[0], box_row_colors.y0, box_row_colors.width*1.5, box_row_colors.height])

        # row legend
        for label in rcolors_dict.keys():
            g2.ax_row_dendrogram.bar(0, 0, color=rcolors_dict[label],
                                    label=label, linewidth=0)
        g2.ax_row_dendrogram.legend(title='Level', loc="upper right", ncol=1, 
                                    bbox_to_anchor=(2., 1.28))

        # column legend
        for label in lut.keys():
            g2.ax_col_dendrogram.bar(0, 0, color=lut[label],
                                    label=label, linewidth=0)
        g2.ax_col_dendrogram.legend(title=ct_label, loc="upper right", ncol=1, 
                                    bbox_to_anchor=(0.9, 0.9))

        # colorbar
        g2.cax.set_position([0.9, .1, .03, .2])
        g2.cax.set_ylabel('Standardized feature value')
        g2.cax.set_yticks([-2,-1,0,1,2])
        
        plt.savefig(f'{ct}_featuremap.png')
        plt.close('all')
 


    #------------- The following section is for data mining -------------------#
    if do_plot and do_prominence_estimation:
        print('==> Estimating prominance features')
        # sort across the class-level
        indices_x = np.argsort(df_feats, axis=1).to_numpy()
        nfeatures, nclasses = indices_x.shape
        orders_x = np.zeros(indices_x.shape)
        for i in range(nfeatures):
            orders_x[i][indices_x[i]] = range(nclasses)
        orders_x = np.fabs(orders_x - (nclasses+1)/2.)
        # sort across the feature-level
        indices_y = np.argsort(orders_x, axis=0)
        orders_y = np.zeros(indices_y.shape)
        for i in range(nclasses):
            orders_y[:,i][indices_y[:,i]] = range(nfeatures)
            
        df_orders = df_feats.copy()
        df_orders.iloc[:,:] = orders_y
        # visualization
        #df_orders.clip(0, 10, inplace=True)  # show only the top10 
        #df_prominence = 1 - df_orders / 10.
        df_prominence = (df_orders - df_orders.shape[0] + 11) / 10.
        df_prominence.clip(0, 1, inplace=True)

        # reorder the columns to keep it the same as clustermap before
        reordered_ind = g2.dendrogram_col.reordered_ind
        df_prominence = df_prominence.iloc[:, reordered_ind]

        g1 = sns.clustermap(df_prominence, cmap='OrRd', row_colors=row_colors,
                            col_colors=col_colors, row_cluster=False,
                            col_cluster=False, 
                            xticklabels=1, yticklabels=1,
                            figsize=(10,20))
        # colorbar
        g1.cax.set_position([0.9, .1, .03, .2])
        g1.cax.set_ylabel('Feature prominence')
        
        plt.savefig(f'{ct}_prominence.png', dpi=300)
        plt.close('all')


    



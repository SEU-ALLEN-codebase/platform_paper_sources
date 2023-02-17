#!/usr/bin/env python

#================================================================
#   Copyright (C) 2023 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : multi-levels_analyzer.py
#   Author       : Yufeng Liu
#   Date         : 2023-02-07
#   Description  : 
#
#================================================================
import os
import glob
import numpy as np
from scipy.spatial import distance_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage

from ds_matrix_condense import summary_class_level_ds

import sys
sys.path.append('../common_lib')
from common_utils import struct_dict, CorticalLayers, PstypesToShow

__LEVELS__ = ('microenviron', 'fullMorpho', 'arbor', 'motif', 'bouton')
sns.set_theme(style="whitegrid", rc={'legend.labelspacing': 1.0, 'font.weight': 'bold'})

def calc_ds_similarity_among_levels(data_dir='./levels', type_str='stype', struct='all'):
    vecs = []
    for level in __LEVELS__:
        print(level)
        csvfile = os.path.join(data_dir, f'corr_regionLevel_sdmatrix_{level}_{type_str}_{struct}.csv')
        matrix = pd.read_csv(csvfile, index_col=0)
        vec = matrix.to_numpy().reshape(-1)
        #vec = vec / np.linalg.norm(vec)
        vecs.append(vec)
    vecs = np.array(vecs)
    #pdists = np.matmul(vecs, vecs.transpose())
    #pdists = pd.DataFrame(pdists, index=__LEVELS__, columns=__LEVELS__)
    pdists = pd.DataFrame(vecs.transpose(), columns=__LEVELS__)
    pdists = pdists.corr()

    ds_value = 'distance'
    pdists = pdists.stack().reset_index(name=ds_value)
    tmp = pdists[ds_value]
    tmp.clip(0, 1)
    pdists.loc[:, ds_value] = 1 - tmp
    
    g = sns.relplot(
        data = pdists, 
        x="level_0", y="level_1", hue=ds_value, size=ds_value,
        palette="icefire", edgecolor="0.7", hue_norm=(0, 0.8),
        height=4, sizes=(10, 500), size_norm=(0, 0.8)
    )
    g.set(xlabel="", ylabel="", aspect="equal")
    g.despine(left=True, bottom=True)
    g.ax.margins(0.1)
    for label in g.ax.get_xticklabels():
        label.set_rotation(90)
    for artist in g.legend.legendHandles:
        artist.set_edgecolor("1.")

    plt.subplots_adjust(left=0.25, right=0.8, bottom=0.3)

    plt.savefig(f'ds_similarity_{type_str}_{struct}.png', dpi=300)
    plt.close('all')


def calc_interregional_stereotypy(data_dir='./levels', type_str='stype'):
    if type_str == 'stype':
        cols = (14,9,3)
        cnames = ('CTX', 'TH', 'STR')
    elif type_str == 'ptype':
        cols = (10, 9, 5, 3, 1, 1, 1, 1, 1)
        cnames = ('CTX-ET', 'CTX-IT', 'TH-core', 'TH-matrix', 'TH-RT', 'CP-GPe', 'CP-SNr', 'STR-ACB', 'STR-OT')
    elif type_str == 'cstype':
        cols = (6, 5, 11, 2)
        cnames = ('L2/3', 'L4', 'L5', 'L6')

    data = []
    data_reg = []
    for level in __LEVELS__:
        feat_file = os.path.join(data_dir, f'corr_regionLevel_sdmatrix_{level}_{type_str}_all.csv')
        df = pd.read_csv(feat_file, index_col=0)
        names = df.columns

        intras, inters, diffs = summary_class_level_ds(df, cols)
        print(f'{level:<15} {intras} {inters} {diffs}')
        data.append(intras)

        intras_reg, inters_reg, diffs_reg = summary_class_level_ds(df, np.ones(df.shape[1], dtype=np.int32))
        #print(f'{level:<15} {intras_reg} {inters_reg} {diffs_reg}')
        data_reg.append(intras_reg)

    

    # region-level
    if type_str == 'ptype':
        height = 15
    else:
        height = 12

    df_reg = pd.DataFrame(data_reg, columns=names, index=__LEVELS__)
    ds_value = 'intra-type\nDS value'
    df_reg_stack = df_reg.transpose().stack().reset_index(name=ds_value)
    orders = [0, 0.2, 0.4, 0.6, 0.8]
    g = sns.relplot(
        data = df_reg_stack, 
        x="level_0", y="level_1", hue=ds_value, size=ds_value,
        palette="icefire", edgecolor="1.", hue_norm=(0.2, 1.0),    # icefire
        height=height, sizes=(30, 500), size_norm=(0.2, 1.0),
    )
    g.set(xlabel="", ylabel="", aspect="equal")
    g.despine(left=True, bottom=True)
    g.ax.margins(x=0.02, y=0.15)
    for label in g.ax.get_xticklabels():
        label.set_rotation(90)
    for artist in g.legend.legendHandles:
        artist.set_edgecolor("1.")
    
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    # adjust the legend
    plt.setp(g._legend.get_texts(), fontsize=16)
    plt.setp(g._legend.get_title(), fontsize=16)

    if type_str == 'ptype':
        print(f'Adjust figure layout for {type_str}')
        plt.subplots_adjust(left=0.12, right=0.92)
    else:
        plt.subplots_adjust(left=0.15, right=0.9)

    plt.savefig(f'intra-region_stereotypy_{type_str}.png', dpi=300)
    plt.close('all')

    
def calc_interregional_stereotypy_ridge_plot(data_dir='./levels', type_str='stype'):
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    vname = 'DS value'
    fontsize = 30
    pal = {
        'microenviron': 'orangered',
        'fullMorpho': 'darkviolet',
        'arbor': 'blue',
        'motif': 'peru',
        'bouton': 'forestgreen'
    }
    type_names = {
        'stype': 's-type',
        'ptype': 'sp-type',
        'cstype': 'sl-type'
    }

    if type_str == 'stype':
        cnames = struct_dict['CTX'] + struct_dict['TH'] + struct_dict['STR']
        hspace = -0.7
        height = 1.0
    elif type_str == 'ptype':
        cnames = PstypesToShow['CTX']# + PstypesToShow['TH'] + PstypesToShow['STR']
        hspace = -0.6
        height = 1.0
    elif type_str == 'cstype':
        cnames = CorticalLayers
        hspace = -0.5
        height = 1.0
    
    
    data = []
    for level in __LEVELS__:
        print(f'--> Processing for level: {level}')
        feat_file = os.path.join(data_dir, f'corr_neuronLevel_sdmatrix_{level}_{type_str}_all.csv')
        df = pd.read_csv(feat_file, index_col=0)

        c_cnt = 0
        for cname in cnames:
            print(f'    --> class: {cname}')
            indices = np.nonzero((df.type == cname).to_numpy())[0]
            dfs = df.iloc[indices, indices].to_numpy()
            triu_indices = np.triu_indices_from(dfs, k=1)
            vds = dfs[triu_indices[0], triu_indices[1]]
            levels = np.tile(level, vds.shape[0])
            names = np.tile(cname, vds.shape[0])
            data.append(np.vstack((vds, levels, names)).transpose())
            
            c_cnt += 1
            #if c_cnt == 5:
            #    break
        
    data = np.vstack(data)
    df = pd.DataFrame(data, columns=[vname, 'level', 'type'])
    df = df.astype({vname: float})
    print(f'Total shape of data: {df.shape}')
    
    # Initialize the facegrid
    g = sns.FacetGrid(df, row='type', col='level', hue='level', aspect=5, 
                      height=height, palette=pal)
    
    # Draw the densities in a few steps
    print('Draw kdeplot...')
    g.map(sns.kdeplot, vname,
          bw_adjust=.5, clip_on=False,
          fill=True, alpha=0.3, linewidth=1.5)
    g.map(sns.kdeplot, vname, clip_on=False, lw=2, bw_adjust=.5, alpha=0.7)

    # passing color=None to refline() uses the hue mapping
    #g.refline(y=0, linewidth=2, linestyle="-", color='k', clip_on=False)
    #g.map(plt.axhline, y=0, linewidth=2, linestyle="-", color='k', clip_on=False)

    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        if label == 'microenviron':
            ax = plt.gca()
            ax.text(0, (1+hspace)*0.4, x.iloc[0], fontweight="bold", color='k',
                    ha="left", va="center", transform=ax.transAxes,
                    fontsize=fontsize)

    g.map(label, 'type')

    # Set the subplots to overlap
    g.figure.subplots_adjust(hspace=hspace, wspace=-0.1)

    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.set(xticks=np.arange(-1,1+0.5,0.5))
    g.set_xticklabels(np.arange(-1,1.5,0.5), fontsize=fontsize-2, fontweight='bold')
    #g.set_axis_labels('Correlation', fontsize=fontsize, fontweight='bold')
    g.set_axis_labels('', fontsize=0, fontweight='bold')
    g.add_legend()
    sns.move_legend(g, 'upper center', ncols=5)
    plt.setp(g._legend.get_title(), fontsize=0, fontweight='bold')
    plt.setp(g._legend.get_texts(), fontsize=fontsize, fontweight='bold')
    
    g.despine(bottom=True, left=True)

    plt.savefig(f'intra-regional_ridge_plot_{type_str}.png', dpi=300)
    plt.close('all')

   
if __name__ == '__main__':

    for type_str in ['stype', 'ptype', 'cstype']:
        #calc_ds_similarity_among_levels(type_str=type_str)
        #calc_interregional_stereotypy(type_str=type_str)
        #if type_str != 'ptype': continue
        calc_interregional_stereotypy_ridge_plot(type_str=type_str)


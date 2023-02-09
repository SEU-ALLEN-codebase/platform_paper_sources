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

    ds_value = 'correlation'
    pdists = pdists.stack().reset_index(name=ds_value)
    
    g = sns.relplot(
        data = pdists, 
        x="level_0", y="level_1", hue=ds_value, size=ds_value,
        palette="icefire", edgecolor="0.7", hue_norm=(0.2, 1),
        height=4, sizes=(10, 500), size_norm=(0.2, 1)
    )
    g.set(xlabel="", ylabel="", aspect="equal")
    g.despine(left=True, bottom=True)
    g.ax.margins(0.1)
    for label in g.ax.get_xticklabels():
        label.set_rotation(90)
    for artist in g.legend.legendHandles:
        artist.set_edgecolor("1.")

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
    '''
    df = pd.DataFrame(data, columns=cnames, index=__LEVELS__)

    ds_value = 'DS value'
    df_stack = df.transpose().stack().reset_index(name=ds_value)
    g = sns.relplot(
        data = df_stack, 
        x="level_0", y="level_1", hue=ds_value, size=ds_value,
        palette="vlag", edgecolor="1.", hue_norm=(0.2, 0.8),
        height=4, sizes=(10, 500), size_norm=(0.2, 0.8)
    )
    g.set(xlabel="", ylabel="", aspect="equal")
    g.despine(left=True, bottom=True)
    g.ax.margins(0.1)
    for label in g.ax.get_xticklabels():
        label.set_rotation(90)
    for artist in g.legend.legendHandles:
        artist.set_edgecolor("1.")
    plt.savefig(f'intra-class_stereotypy_{type_str}.png', dpi=300)
    plt.close('all')
    '''

    # region-level
    if type_str == 'ptype':
        height = 15
    else:
        height = 12

    df_reg = pd.DataFrame(data_reg, columns=names, index=__LEVELS__)
    ds_value = 'inter-type\nDS value'
    df_reg_stack = df_reg.transpose().stack().reset_index(name=ds_value)
    g = sns.relplot(
        data = df_reg_stack, 
        x="level_0", y="level_1", hue=ds_value, size=ds_value,
        palette="icefire", edgecolor="1.", hue_norm=(0.2, 1.0),    # icefire
        height=height, sizes=(30, 500), size_norm=(0.2, 1.0)
    )
    g.set(xlabel="", ylabel="", aspect="equal")
    g.despine(left=True, bottom=True)
    g.ax.margins(x=0.02, y=0.15)
    for label in g.ax.get_xticklabels():
        label.set_rotation(90)
    for artist in g.legend.legendHandles:
        artist.set_edgecolor("1.")
    
    plt.savefig(f'intra-region_stereotypy_{type_str}.png', dpi=300)
    plt.close('all')
    
    
if __name__ == '__main__':

    for type_str in ['stype', 'ptype', 'cstype']:
        calc_ds_similarity_among_levels(type_str=type_str)
        calc_interregional_stereotypy(type_str=type_str)


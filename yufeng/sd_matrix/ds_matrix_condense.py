#!/usr/bin/env python

#================================================================
#   Copyright (C) 2023 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : ds_matrix_condense.py
#   Author       : Yufeng Liu
#   Date         : 2023-02-08
#   Description  : 
#
#================================================================
import numpy as np

def summary_class_level_ds(df, cols=(14,9,3)):
    if type(df) is str:
        df = pd.read_csv(df, index_col=0)

    icols = [0]
    icol = 0
    for ncol in cols:
        icol += ncol
        icols.append(icol)

    intras = []
    inters = []
    diffs = []
    for i in range(len(icols)-1):
        ix = icols[i]
        iy = icols[i+1]
        #print(len(df.columns[ix:iy]), df.columns[ix:iy])
        intra = df.iloc[ix:iy, ix:iy].sum().sum()
        inter = df.iloc[ix:iy].sum().sum() - intra
        intra /= (iy - ix)**2
        inter /= ((df.shape[0] - iy + ix) * (iy - ix))
        intras.append(intra)
        inters.append(inter)
        diffs.append(intra - inter)
    intras, inters, diffs = np.round(np.array(intras), 4), np.round(np.array(inters), 4), np.round(np.array(diffs), 4)
    return intras, inters, diffs

def plot_dsmatrix_illustration():
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    data = pd.DataFrame([[0.6, -0.3],[-0.3, 0.8]], 
                        columns=['Cell type1', 'Cell type2'], 
                        index=['Cell type1', 'Cell type2'])
    labels = [['intra-type', 'inter-type'],
            ['inter-type', 'intra-type']]
    fig, axes = plt.subplots(figsize=(8,6))
    g = sns.heatmap(data=data, cmap='vlag', annot=labels, fmt='', cbar=True, ax=axes, 
                    annot_kws={"size": 25, 'color':'black'}, 
                    vmin=-.5, vmax=1)
    plt.setp(axes.xaxis.get_majorticklabels(), fontsize=25)
    plt.setp(axes.yaxis.get_majorticklabels(), fontsize=25)
    g.figure.axes[-1].set_yticks([-0.5,0,0.5,1.0], [-0.5,0,0.5,1.0], fontsize=20)

    plt.savefig('example_dsmatrix.png', dpi=300)
    plt.close('all')

if __name__ == '__main__':
    plot_dsmatrix_illustration()
        


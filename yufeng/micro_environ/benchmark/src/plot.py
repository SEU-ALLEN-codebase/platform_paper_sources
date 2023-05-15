#!/usr/bin/env python

#================================================================
#   Copyright (C) 2023 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : plot.py
#   Author       : Yufeng Liu
#   Date         : 2023-04-18
#   Description  : 
#
#================================================================

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def load_data(gsfile, reconfile, matchfile, min_nodes=300):
    gs = pd.read_csv(gsfile, index_col=0)
    recon = pd.read_csv(reconfile, index_col=0)
    match = pd.read_csv(matchfile, index_col=0)
    # select only matched
    gsm = gs[gs.index.isin(match.filename_y)].drop(['region_name_r316'], axis=1)
    reconm = recon[recon.index.isin(match.index)].drop(['region_name_r316'], axis=1)
    if min_nodes != -1:
        # filtering by nodes
        reconm = reconm[reconm.Nodes > min_nodes]
        mdict = dict(match.filename_y)
        new_files = [mdict[f] for f in reconm.index]
        gsm = gsm[gsm.index.isin(set(new_files))]

    return gsm, reconm

def plot_lmeasure_old(gsm, reconm, figname='temp.png'):
    df = pd.concat([gsm, reconm])
    data = ['gs' for i in range(len(gsm))] + ['recon' for i in range(len(reconm))]
    df['data'] = data

    # merge data
    h, w = 3, 6
    font = 22
    fig = plt.figure(figsize=(w*5, h*5))
    show_indices = list(range(2,9))+list(range(10,17))+list(range(19,22))
    
    feat_names = gsm.columns
    sf = 4
    xlabel_list = [
        'Number',
        'Number',
        'Number',
        'Number',
        'Voxel',
        'Voxel',
        'Voxel',
        'Voxel',
        f'$Voxel^2 (x10^{sf})$',
        '$Voxel^3$',
        'Voxel',
        'Voxel',
        'Number',
        '',
        'Degree',
        'Degree',
        ''
    ]
    df.loc[:,'Surface'] = df.loc[:,'Surface'] / np.power(10,sf)
    for i,f in enumerate(show_indices):
        feature = feat_names[f]
        log = False
        if f in [0,12,17]: log=True

        axes = fig.add_subplot(h,w,i+1)
        kdp = sns.kdeplot(data=df, x=feature, hue="data",
                    log_scale=log,
                    bw_adjust=1.0, fill=False, alpha=0.6)
        
        dfs = df[feature]
        dfs_g = dfs[df.data == 'gs']
        dfs_r = dfs[df.data == 'recon']

        # chisqure test
        #stat, p_value = stats.ttest_ind(dfs_g, dfs_r)
        #print(f'Chi-square Test for {feature}: statistic={stat:.4f}, p-value={p_value}')
        
        # Kolmogorov-Smirnov Test
        stat, p_value = stats.mannwhitneyu(dfs_g, dfs_r, alternative='two-sided')
        print(f'KS Test for {feature}: statistic={stat:.4f}, p-value={p_value}')
        

        if feature == 'AverageBifurcationAngleLocal':
            title = 'AvgBifAngLocal'
        elif feature == 'AverageBifurcationAngleRemote':
            title = 'AvgBifAngRemote'
        elif feature == 'AverageContraction':
            title = 'AvgContraction'
        else:
            title = feature

        axes.set_title(title, fontsize=font)
        #axes.text(0,2,feature,va='top',ha='center',fontsize=font)
        print(i, f, xlabel_list[i])
        axes.set_xlabel(xlabel_list[i])
        axes.set_ylabel('Density',fontsize=font)
        axes.spines['top'].set_visible(False)
        axes.spines['right'].set_visible(False)
        axes.spines['bottom'].set_linewidth(2)
        axes.spines['left'].set_linewidth(2)

        if i not in [0, 6, 12]:
            #axes.set_xlabel('')
            axes.set_ylabel('')
        if f != 21:
            axes.get_legend().remove()

    plt.subplots_adjust(wspace=0.16,hspace=0.25)
    plt.savefig(figname, dpi=300)
    
def plot_lmeasure(gsm, reconm, figname='temp.png'):
    df = pd.concat([gsm, reconm])
    data = ['Manual Annotation' for i in range(len(gsm))] + ['Reconstruction' for i in range(len(reconm))]
    df['data'] = data

    sf = 4
    df.loc[:,'Surface'] = df.loc[:,'Surface'] / np.power(10,sf)

    t_font = 22
    l_font = 18
    font = {'size': t_font}
    matplotlib.rc('font', **font)

    pfeatures = ['Branches', 'Surface']
    for pf in pfeatures:
        fig = plt.figure(figsize=(8,8))
        #bp = sns.boxplot(data=df, y=pf, x='data', hue='data', 
        #                 orient='v')
        kdp = sns.kdeplot(data=df, x=pf, hue="data",
                    bw_adjust=1.0, fill=True, alpha=0.6)
        
        axes = plt.gca()
        #axes.set_title(pf, fontsize=font)
        #axes.text(0,2,feature,va='top',ha='center',fontsize=font)
        axes.spines['top'].set_visible(False)
        axes.spines['right'].set_visible(False)
        axes.spines['bottom'].set_linewidth(2)
        axes.spines['left'].set_linewidth(2)
        axes.xaxis.set_tick_params(width=2, direction='out')
        axes.yaxis.set_tick_params(width=2, direction='out')

        if pf == 'Surface':
            xlabel = f'Surface (x$10^{sf}  voxel^2$)'
        else:
            xlabel = 'Number of Branches'
        axes.set_xlabel(xlabel, fontsize=t_font)
        axes.set_ylabel('Density', fontsize=t_font)
        axes.legend_.set_frame_on(False)
        axes.legend_.set_title('')

        fig.subplots_adjust(left=0.16, bottom=0.16)
        plt.savefig(f'{figname}_{pf}', dpi=300)


if __name__ == '__main__':
    gsfile = 'lm_gs.csv'
    reconfile = 'lm_weak1854.csv'
    matchfile = 'utils/file_mapping1854.csv'
    min_nodes = -1
    figname = 'comp_lmeasure'

    gsm, reconm = load_data(gsfile, reconfile, matchfile, min_nodes=min_nodes)
    plot_lmeasure(gsm, reconm, figname)
    



#!/usr/bin/env python

#================================================================
#   Copyright (C) 2023 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : plot_features.py
#   Author       : Yufeng Liu
#   Date         : 2023-01-03
#   Description  : 
#
#================================================================

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import sys
sys.path.append('../common_lib')
from common_utils import load_pstype_from_excel, load_type_from_excel


FEAT_NAMES = ["Bouton Number", "TEB Ratio", 
                "Bouton Density", "Geodesic Distance",
                "Bouton Interval", "Project Regions"]

def load_features(feat_file):
    df = pd.read_csv(feat_file)
    return df

def load_data(feat_file, ctype_file, normalize=False, remove_unwanted_regions=True):
    df_ct = pd.read_csv(ctype_file)
    df_f = pd.read_csv(feat_file)
    df = df_f.merge(df_ct, how='inner', on='Cell name')
    # normalize
    if normalize:
        for feat in FEAT_NAMES:
            df[feat] /= df[feat].max()


    # Process the type
    ctypes = []
    for x in df['Subclass_or_type']:
        if x is np.NaN:
            ctypes.append(x)
        else:
            ctypes.append(x.split('_')[0])
    df['Brain structure'] = ctypes

    if remove_unwanted_regions:
        df = df[(df['Brain structure'] == 'TH') | (df['Brain structure'] == 'CTX') | (df['Brain structure'] == 'CP')]
    
    return df


def load_ctx_layers_data(feat_file, ctype_file, normalize=False):
    df_ct = pd.read_csv(ctype_file)
    df_f = pd.read_csv(feat_file)
    df = df_f.merge(df_ct, how='inner', on='Cell name')
    # normalize
    if normalize:
        for feat in FEAT_NAMES:
            df[feat] /= df[feat].max()


    df = df[~df['Cortical_layer'].isnull()]

    df = df[df['Subclass_or_type'] != 'Car3']
    
    return df

def draw_line(df, subclass='CTX_ET', min_counts=9):
    feat_names = FEAT_NAMES
    
    df_sub = df[df['Subclass_or_type'] == subclass]
    stypes, counts = np.unique(df_sub['Manually_corrected_soma_region'], return_counts=True)
    print(stypes, counts)
    colors = 'rgbcmykrgbcmyk'
    styles = 'ooooooo^^^^^^^'
    ic = 0
    for stype, count in zip(stypes, counts):
        if count < min_counts or stype in ['fiber tracts', '', 'unknown']:
            continue
        feats = df_sub[df_sub['Manually_corrected_soma_region'] == stype][feat_names]
        mean = feats.mean()
        std = feats.std()
        #plt.errorbar(range(feats), mean, fmt=f'{colors[ic]}o-', yerr=std.to_numpy(), label=stype)
        plt.errorbar(range(len(feat_names)), mean.to_numpy(), fmt=f'{colors[ic]}{styles[ic]}-', yerr=std.to_numpy(), label=stype)

        ic += 1

    plt.legend()
    plt.ylim(0, 1.0)
    plt.xticks(ticks=range(len(feat_names)), labels=feat_names)
    plt.title(subclass)
    plt.savefig(f'bouton_features_{subclass}.png', dpi=200)
    plt.close('all')

def draw_lines(df, min_counts=9):
    feat_names = FEAT_NAMES
    xlabels = FEAT_NAMES
    subclasses = ['CP_SNr', 'CP_GPe', 'CTX_ET', 'CTX_IT', 'TH_core', 'TH_matrix']

    colors = 'rgbcmykrgbcmyk'
    styles = 'ooooooo^^^^^^^'

    fig, axes = plt.subplots(3,2, figsize=(10,10))
    for i, subclass in enumerate(subclasses):
        print(i, subclass)
        i1 = i // 2
        i2 = i % 2
        
        df_sub = df[df['Subclass_or_type'] == subclass]
        stypes, counts = np.unique(df_sub['Manually_corrected_soma_region'], return_counts=True)
        ic = 0
        for stype, count in zip(stypes, counts):
            if count < min_counts or stype in ['fiber tracts', '', 'unknown']:
                continue
            feats = df_sub[df_sub['Manually_corrected_soma_region'] == stype][feat_names]
            mean = feats.mean()
            std = feats.std()
            #plt.errorbar(range(feats), mean, fmt=f'{colors[ic]}o-', yerr=std.to_numpy(), label=stype)
            axes[i1,i2].errorbar(range(len(feat_names)), mean.to_numpy(), fmt=f'{colors[ic]}{styles[ic]}-', yerr=std.to_numpy(), label=stype)

            ic += 1
        axes[i1,i2].legend(fontsize=8, ncol=2, loc='upper left')
        axes[i1,i2].set_title(subclass, fontsize=22, pad=-40)
        axes[i1,i2].grid(axis='y', alpha=0.5, ls='--')
        axes[i1,i2].set_ylim(0, 0.9)
        if i1 == 2:
            axes[i1,i2].set_xticks(ticks=range(len(feat_names)), labels=xlabels, rotation=30, fontsize=20)
        else:
            axes[i1,i2].set_xticks(ticks=range(len(feat_names)), labels=['','','','','',''])

        [axes[i1,i2].spines[s].set_visible(False) for s in ['top','right']]
        axes[i1,i2].spines['left'].set_linewidth(3)
        axes[i1,i2].spines['left'].set_alpha(0.5)
        axes[i1,i2].spines['bottom'].set_linewidth(3)
        axes[i1,i2].spines['bottom'].set_alpha(0.5)
        
    plt.tight_layout()   
    plt.savefig(f'bouton_features.png', dpi=200)
    plt.close('all')

def draw_pairplot(df):
    cname = 'Brain structure'
    # remove some types
    palette = {
        'CTX': 'red',
        'CP': 'cornflowerblue',
        'TH': 'lime'
    }

    mapper = {}
    for key in FEAT_NAMES:
        nkey = key.replace(' ', '\n')
        mapper[key] = nkey
    df = df.rename(columns=mapper)
    sns.set_context("notebook", rc={"axes.labelsize":22})
    g = sns.pairplot(df, vars=list(mapper.values()), hue=cname, palette=palette)
    sns.move_legend(g, "upper left", bbox_to_anchor=(0.12,1.0))
    plt.setp(g._legend.get_title(), fontsize=15)
    plt.setp(g._legend.get_texts(), fontsize=16)
    plt.savefig('temp.png', dpi=200)
    
    print('aa')

def draw_displot(df, n_neurons=10):
    st_key = 'Brain structure'
    stype_key = "Manually_corrected_soma_region"

    # keep only s-types with at least 10 neurons
    structures = np.unique(df[st_key])
    outlist = []
    for st in structures:
        df_st = df[df[st_key] == st]
        stypes, counts = np.unique(df_st[stype_key], return_counts=True)
        for stype, count in zip(stypes, counts):
            if count >= n_neurons:
                outlist.append(df_st[df_st[stype_key] == stype].to_numpy())
    
    df_filtered = pd.DataFrame(np.vstack(outlist), columns=df.columns)
    
    '''df_stack = pd.DataFrame(df_filtered[FEAT_NAMES].to_numpy().transpose().reshape(-1), columns=['Features'])
    df_stack['Feature type'] = [fn for fn in FEAT_NAMES for i in range(df_filtered.shape[0])]
    df_stack[st_key] = np.tile(df_filtered[st_key].to_numpy(), len(FEAT_NAMES))
    df_stack[stype_key] = np.tile(df_filtered[stype_key].to_numpy(), len(FEAT_NAMES))

    sns.set_style("darkgrid")
    g = sns.displot(
        data=df_stack, x="Features", hue=stype_key, row=st_key, col='Feature type',
        kind="kde", height=5, common_norm=False
    )
    plt.savefig('temp.png', dpi=200)
    '''

    sns.set_style("whitegrid")
    sns.set_context("notebook", rc={"axes.labelsize": 20})
    for feat in FEAT_NAMES:
        g = sns.displot(data=df_filtered, x=feat, hue=stype_key, row=st_key, 
            common_norm=False, kind='kde', linewidth=3)
        g._legend.set_title(None)
        plt.setp(g._legend.get_texts(), fontsize=15)
        g.set(ylabel=None)
        g.set(yticklabels=[])
        plt.savefig(f"{'_'.join(feat.split())}.png", dpi=200)
        plt.close('all')


def draw_stypes_by_ctx_layers(df):
    cname = 'Stype_layer'

    # use only subset of data for better visualization
    slayers = []
    for irow, row in df.iterrows():
        stype = row['Manually_corrected_soma_region']
        layer = row['Cortical_layer']
        slayers.append(f'{stype}-L{layer}')

    df['Stype_layer'] = slayers
    '''

    #import ipdb; ipdb.set_trace()
    df = df[(df[cname] == 'MOs-L2/3') | (df[cname] == 'MOs-L5') | (df[cname] == 'MOp-L2/3') | (df[cname] == 'MOp-L5')]
    

    # remove some types
    palette = {
        'MOs-L2/3': 'red',
        'MOs-L5': 'cornflowerblue',
        'MOp-L2/3': 'lime',
        'MOp-L5': 'darkviolet'
    }

    mapper = {}
    for key in FEAT_NAMES:
        nkey = key.replace(' ', '\n')
        mapper[key] = nkey
    df = df.rename(columns=mapper)
    sns.set_context("notebook", rc={"axes.labelsize":22})
    g = sns.pairplot(df, vars=list(mapper.values()), hue=cname, palette=palette)
    sns.move_legend(g, "upper left", bbox_to_anchor=(0.12,1.0))
    plt.setp(g._legend.get_title(), fontsize=15)
    plt.setp(g._legend.get_texts(), fontsize=16)
    plt.savefig('temp.png', dpi=200)
    '''

    st_key = 'Cortical_layer'
    stype_key = cname

    # keep only s-types with at least 10 neurons
    structures = np.unique(df[st_key])
    outlist = []
    for st in structures:
        df_st = df[df[st_key] == st]
        stypes, counts = np.unique(df_st[stype_key], return_counts=True)
        for stype, count in zip(stypes, counts):
            if count >= 10:
                outlist.append(df_st[df_st[stype_key] == stype].to_numpy())
    
    df_filtered = pd.DataFrame(np.vstack(outlist), columns=df.columns)

    sns.set_style("whitegrid")
    sns.set_context("notebook", rc={"axes.labelsize": 20})
    for feat in FEAT_NAMES:
        g = sns.displot(data=df_filtered, x=feat, hue=stype_key, row=st_key,
            common_norm=False, kind='kde', linewidth=3)
        g._legend.set_title(None)
        plt.setp(g._legend.get_texts(), fontsize=15)
        g.set(ylabel=None)
        g.set(yticklabels=[])
        plt.savefig(f"{'_'.join(feat.split())}_ctx_layers.png", dpi=200)
        plt.close('all')
    

if __name__ == '__main__':
    ctype_file = '../common_lib/41586_2021_3941_MOESM4_ESM.csv'
    feat_file = './bouton_features/bouton_features.csv'
    
    
    #df = load_data(feat_file, ctype_file, normalize=False)
    #draw_pairplot(df)
    #draw_displot(df)

    # draw ctx_layers
    df = load_ctx_layers_data(feat_file, ctype_file, normalize=False)
    draw_stypes_by_ctx_layers(df)
    
    




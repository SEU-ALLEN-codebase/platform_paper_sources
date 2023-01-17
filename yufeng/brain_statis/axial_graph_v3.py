#!/usr/bin/env python

#================================================================
#   Copyright (C) 2022 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : axial_graph_v2.py
#   Author       : Yufeng Liu
#   Date         : 2022-12-21
#   Description  : 
#
#================================================================

import os
import glob
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN

import networkx as nx

from file_io import load_image
from anatomy.anatomy_core import parse_regions316, parse_ana_tree

from brain_analyzer_v2 import BrainsSignalAnalyzer, format_precomputed_df

def get_col_by_axis(axis='AP'):
    if axis == 'AP':    
        col_name = 'centerX'
    elif axis == 'DV':
        col_name = 'centerY'
    elif axis == 'LR':
        col_name = 'centerZ'
    else:
        raise NotImplementedError(f"param @axis only receive 'AP', 'DV' and 'LR', but got {axis}")
    return col_name

def graph_with_corr(ndict, df_centers, corr, col_name, ana_dict):
    G = nx.DiGraph()
    for k, v in ndict.items():
        for vi in v:
            ck = df_centers.at[k, col_name]
            cvi = df_centers.at[vi, col_name]
            kname = ana_dict[k]['acronym']
            viname = ana_dict[vi]['acronym']
            cc = corr.at[k, vi]
            pw = max(10 * cc - 4, 0.5)
            sa = max(pw / 3, 0.5)

            if ck < cvi:
                G.add_edge(kname, viname, penwidth=pw, arrowsize=sa)
            else:
                G.add_edge(viname, kname, penwidth=pw, arrowsize=sa)

    # do transitive reduction
    TR = nx.transitive_reduction(G)
    TR.add_nodes_from(G.nodes(data=True))
    TR.add_edges_from((u, v, G.edges[u, v]) for u, v in TR.edges)

    vis = True
    if vis:
        A = nx.nx_agraph.to_agraph(TR)  # convert to a graphviz graph
        A.draw("init_graph.png", prog='dot')  # Draw with pygraphviz

    return TR


def get_color(id_path, cmap):
    for idx in id_path:
        if idx in cmap:
            return cmap[idx], idx


def initialize_graph(df_centers, ndict, ana_dict, col_name, pw=4, asize=4):
    ctx = 688  # cortex, CTX

    G = nx.DiGraph()
    for k, v in ndict.items():
        #id_path1 = ana_dict[k]['structure_id_path']
        #if ctx not in id_path1:
        #    continue
        for vi in v:
            #id_path2 = ana_dict[vi]['structure_id_path']
            #if ctx not in id_path2:
            #    continue

            ck = df_centers.at[k, col_name]
            cvi = df_centers.at[vi, col_name]
            kname = ana_dict[k]['acronym']
            viname = ana_dict[vi]['acronym']
            if ck < cvi:
                G.add_edge(kname, viname, arrowsize=asize, penwidth=pw)
            else:
                G.add_edge(viname, kname, arrowsize=asize, penwidth=pw)

    # do transitive reduction
    TR = nx.transitive_reduction(G)
    TR.add_nodes_from(G.nodes(data=True))
    TR.add_edges_from((u, v, G.edges[u, v]) for u, v in TR.edges)

    return TR

def graph_with_label(ndict, df_centers, df_distr, col_name, ana_dict):
    cmap = {
        688: 'orange', # CTX
        623: 'green', # CNU: STR + PAL
        512: 'magenta',  # CB, 
        343: 'cyan',  # BS: IB(549(TH)+HY), MB, HB
    }

    TR = initialize_graph(df_centers, ndict, ana_dict, col_name)
    labels = [l for l in np.unique(df_distr['label']) if l not in ('', 'Ai139')]

    for cur_label in labels:
        print(f'==> Processing for {cur_label}')
        df_labels = df_distr[df_distr['label'] == cur_label].drop(['label'], axis=1).median(axis=0)
        nodes = TR.nodes
        TR.graph['label'] = f'\n{cur_label}'
        stats = []
        for rid in df_labels.index:
            v = df_labels.at[rid]
            rname = ana_dict[rid]['acronym']
            if df_centers.shape[0] < 300:
                pw = max(min(125 * v + 0.5, 10), 0.5)
            else:
                pw = max(min(250 * v + 0.5, 10), 0.5)

            id_path = ana_dict[rid]['structure_id_path']
            color, sid = get_color(id_path, cmap)
            if pw > 3:
                stats.append((ana_dict[sid]['acronym'], rname, v))
            if rname in nodes:
                nodes[rname]['penwidth'] = pw
                nodes[rname]['fillcolor'] = color
                nodes[rname]['style'] = 'filled'

        stats = sorted(stats, key=lambda x:x[-1], reverse=True)
        print(f'#regions: {len(stats)}')
        for st in stats:
            print(f'{st[0]} {st[1]} {st[2]:.4f}')
        
        A = nx.nx_agraph.to_agraph(TR)  # convert to a graphviz graph
        lstr = cur_label.replace(';', '-').replace('+', '--')
        A.draw(f"graph_label{lstr}.png", prog='dot')  # Draw with pygraphviz   

    return TR

def graph_for_label_merged(ndict, df_centers, df_distr, col_name, ana_dict):
    cmap = {
        688: 'orange', # CTX
        623: 'green', # CNU: STR + PAL
        512: 'magenta',  # CB, 
        343: 'cyan',  # BS: IB(549(TH)+HY), MB, HB
    }

    TR = initialize_graph(df_centers, ndict, ana_dict, col_name)

    print(f'==> Processing for somata')
    df_labels = df_distr.drop(['label'], axis=1).mean(axis=0)
    df_ln = df_labels / df_labels.sum()
    nodes = TR.nodes
    TR.graph['label'] = f'\nlabel'
    stats = []
    for rid in df_labels.index:
        v = df_labels.at[rid]
        vn = df_ln.at[rid]
        rname = ana_dict[rid]['acronym']
        if vn > 0.01:
            bcolor = 'black'
        else:
            bcolor = 'white'
        
        id_path = ana_dict[rid]['structure_id_path']
        color, sid = get_color(id_path, cmap)
        pw = 10
        stats.append((ana_dict[sid]['acronym'], rname, v, vn))
        if rname in nodes:
            nodes[rname]['penwidth'] = pw
            nodes[rname]['fillcolor'] = color
            nodes[rname]['style'] = 'filled'
            nodes[rname]['color'] = bcolor

    stats = sorted(stats, key=lambda x:x[-1], reverse=True)
    print(f'#regions: {len(stats)}')
    for st in stats:
        print(f'{st[0]} {st[1]} {int(st[2]):d} {st[3]:.4f}')
    
    A = nx.nx_agraph.to_agraph(TR)  # convert to a graphviz graph
    A.draw(f"graph_label.png", prog='dot')  # Draw with pygraphviz   

    return TR

def graph_for_somata(ndict, df_centers, df_distr, col_name, ana_dict):
    cmap = {
        688: 'orange', # CTX
        623: 'green', # CNU: STR + PAL
        512: 'magenta',  # CB, 
        343: 'cyan',  # BS: IB(549(TH)+HY), MB, HB
    }

    TR = initialize_graph(df_centers, ndict, ana_dict, col_name)
    labels = [l for l in np.unique(df_distr['label']) if l not in ('', 'Ai139')]

    for cur_label in labels:
        print(f'==> Processing for {cur_label}')
        df_labels = df_distr[df_distr['label'] == cur_label].drop(['label'], axis=1).median(axis=0)
        df_ln = df_labels / df_labels.sum()
        nodes = TR.nodes
        TR.graph['label'] = f'\n{cur_label}'
        stats = []
        for rid in df_labels.index:
            v = df_labels.at[rid]
            vn = df_ln.at[rid]
            rname = ana_dict[rid]['acronym']
            if v < 10:
                pw = 0.5
            elif df_centers.shape[0] < 300:
                pw = max(min(125 * vn + 0.5, 10), 0.5)
            else:
                pw = max(min(250 * vn + 0.5, 10), 0.5)
            
            id_path = ana_dict[rid]['structure_id_path']
            color, sid = get_color(id_path, cmap)
            if pw > 3:
                stats.append((ana_dict[sid]['acronym'], rname, v, vn))
            if rname in nodes:
                nodes[rname]['penwidth'] = pw
                nodes[rname]['fillcolor'] = color
                nodes[rname]['style'] = 'filled'

        stats = sorted(stats, key=lambda x:x[-1], reverse=True)
        print(f'#regions: {len(stats)}')
        for st in stats:
            print(f'{st[0]} {st[1]} {int(st[2]):d} {st[3]:.4f}')
        
        A = nx.nx_agraph.to_agraph(TR)  # convert to a graphviz graph
        lstr = cur_label.replace(';', '-').replace('+', '--')
        A.draw(f"graph_label{lstr}.png", prog='dot')  # Draw with pygraphviz   

    return TR

def graph_for_somata_merged(ndict, df_centers, df_distr, col_name, ana_dict):
    cmap = {
        688: 'orange', # CTX
        623: 'green', # CNU: STR + PAL
        512: 'magenta',  # CB, 
        343: 'cyan',  # BS: IB(549(TH)+HY), MB, HB
    }

    TR = initialize_graph(df_centers, ndict, ana_dict, col_name)

    print(f'==> Processing for somata')
    df_labels = df_distr.drop(['label'], axis=1).mean(axis=0)
    df_ln = df_labels / df_labels.sum()
    nodes = TR.nodes
    TR.graph['label'] = f'\nsomata'
    stats = []
    for rid in df_labels.index:
        v = df_labels.at[rid]
        vn = df_ln.at[rid]
        rname = ana_dict[rid]['acronym']
        if vn < 0.0002:
            bcolor = 'white'
        elif vn < 0.001:
            bcolor = 'red'
        else:
            bcolor = 'white'
        
        id_path = ana_dict[rid]['structure_id_path']
        color, sid = get_color(id_path, cmap)
        pw = 10
        stats.append((ana_dict[sid]['acronym'], rname, v, vn))
        if rname in nodes:
            nodes[rname]['penwidth'] = pw
            nodes[rname]['fillcolor'] = color
            nodes[rname]['style'] = 'filled'
            nodes[rname]['color'] = bcolor

    stats = sorted(stats, key=lambda x:x[-1], reverse=True)
    print(f'#regions: {len(stats)}')
    for st in stats:
        print(f'{st[0]} {st[1]} {int(st[2]):d} {st[3]:.4f}')
    
    A = nx.nx_agraph.to_agraph(TR)  # convert to a graphviz graph
    A.draw(f"graph_somata.png", prog='dot')  # Draw with pygraphviz   

    return TR

def graph_for_corr_separate(ndict, df_centers, df_distr, col_name, ana_dict):
    cmap = {
        688: 'orange', # CTX
        623: 'green', # CNU: STR + PAL
        512: 'magenta',  # CB,
        343: 'cyan',  # BS: IB(549(TH)+HY), MB, HB
    }

    module1 = "GPi, VTA, PPN".split(', ')
    module2 = "LIN, SPVI, SPVO, SOC, NTS, RO, TRN, PAS, P5, PGRNl, I5, Acs5, ICB, y, PARN, ISN, MDRNv, MARN, DMX, PC5, AMB, PRP, IRN, LRN, ACVII".split(', ')
    module3 = "SCs, NPC, LGd, IntG, PO, MPT, NOT, PPT".split(', ')

    TR = initialize_graph(df_centers, ndict, ana_dict, col_name)

    print(f'==> Processing for somata')
    nodes = TR.nodes
    TR.graph['label'] = f'\nlabel'

    for prid in df_distr.columns.drop(['label']):
        rname = ana_dict[prid]['acronym']

        if rname in module1:
            pw = 25
            bcolor = 'red'
        elif rname in module2:
            pw = 25
            bcolor = 'orange'
        elif rname in module3:
            pw = 25
            bcolor = 'black'
        else:
            pw = 1
            bcolor = 'black'

        id_path = ana_dict[prid]['structure_id_path']
        color, sid = get_color(id_path, cmap)
        if rname in nodes:
            nodes[rname]['penwidth'] = pw
            nodes[rname]['fillcolor'] = color
            nodes[rname]['style'] = 'filled'
            nodes[rname]['height'] = 2
            nodes[rname]['width'] = 3.5
            nodes[rname]['fontsize'] = 80
            nodes[rname]['color'] = bcolor

    A = nx.nx_agraph.to_agraph(TR)  # convert to a graphviz graph
    A.draw(f"modularized_corr.png", prog='dot')  # Draw with pygraphviz

    return TR

def graph_for_all_modules(ndict, df_centers, df_distr, col_name, ana_dict, module_file='modules.txt'):
    cmap = {
        688: 'orange', # CTX
        623: 'green', # CNU: STR + PAL
        512: 'magenta',  # CB,
        343: 'cyan',  # BS: IB(549(TH)+HY), MB, HB
    }

    modules = []
    with open('modules.txt') as fp:
        for line in fp.readlines():
            line = line.strip()
            if not line: continue
            regions = line.split(', ')
            modules.append(regions)

    TR = initialize_graph(df_centers, ndict, ana_dict, col_name)

    print(f'==> Processing for somata')
    nodes = TR.nodes
    TR.graph['label'] = f'\nlabel'

    for i, module in enumerate(modules):
        print(f'--> Module {i+1}')
        for prid in df_distr.columns.drop(['label']):
            rname = ana_dict[prid]['acronym']

            if rname in module:
                pw = 25
                bcolor = 'red'
            else:
                pw = 1
                bcolor = 'black'

            id_path = ana_dict[prid]['structure_id_path']
            color, sid = get_color(id_path, cmap)
            if rname in nodes:
                nodes[rname]['penwidth'] = pw
                nodes[rname]['fillcolor'] = color
                nodes[rname]['style'] = 'filled'
                nodes[rname]['height'] = 2
                nodes[rname]['width'] = 3.5
                nodes[rname]['fontsize'] = 80
                nodes[rname]['color'] = bcolor

        A = nx.nx_agraph.to_agraph(TR)  # convert to a graphviz graph
        A.draw(f"module{i+1}_graph.png", prog='dot')  # Draw with pygraphviz

    return TR

def graph_for_correlated_regions(ndict, df_centers, df_distr, col_name, ana_dict):
    cmap = {
        688: 'orange', # CTX
        623: 'green', # CNU: STR + PAL
        512: 'magenta',  # CB,
        343: 'cyan',  # BS: IB(549(TH)+HY), MB, HB
    }

    inode = 'MOp'
    #onodes = "IG, ACAd, ILA, ACAv, AD, AId, AIp, AIv, AM, FRP, AT, SSp-bfd, SSp-ll, SSp-m, SSp-n, SSp-tr, SSp-ul, SSs, VISp, VISam, VISal, VISl, VISrl, SPVO, VISpm, TEa, CL, V, VISC, VM, ORBl, VPLpc, ORBm, VPMpc, ORBvl, PARN, RSPagl, PCG, ENTl, PERI, PF, PL, MOp, MOs, AUDp, AUDd, AUDv, AUDpo, POL, GU, PRNc, IAD, SSp-un, VISa, VISli, VISpor, VMPO".split(', ')   # cc thresh 0.2
    onodes = "ACAd, ACAv, AD, AId, AIv, FRP, SSp-bfd, SSp-ll, SSp-m, SSp-n, SSp-tr, SSp-ul, SSs, VISal, VM, MOp, MOs, AUDd, AUDpo, GU, SSp-un, VISa, VISli".split(', ') # cc thresh 0.4

    TR = initialize_graph(df_centers, ndict, ana_dict, col_name)

    print(f'==> Processing for somata')
    nodes = TR.nodes
    TR.graph['label'] = f'\nlabel'

    for prid in df_distr.columns.drop(['label']):
        rname = ana_dict[prid]['acronym']

        if rname == 'MOp':
            pw = 25
            bcolor = 'red'
        elif rname in onodes:
            pw = 25
            bcolor = 'black'
        else:
            pw = 1
            bcolor = 'black'

        id_path = ana_dict[prid]['structure_id_path']
        color, sid = get_color(id_path, cmap)
        if rname in nodes:
            nodes[rname]['penwidth'] = pw
            nodes[rname]['fillcolor'] = color
            nodes[rname]['style'] = 'filled'
            nodes[rname]['height'] = 2
            nodes[rname]['width'] = 3.5
            nodes[rname]['fontsize'] = 80
            nodes[rname]['color'] = bcolor

    A = nx.nx_agraph.to_agraph(TR)  # convert to a graphviz graph
    A.draw(f"{inode}_corr.png", prog='dot')  # Draw with pygraphviz

    return TR



def draw_AP_graph_with_corr(center_file, precomputed_signal, neighbor_file, ignore_lr=True, last_column='modality', nr=70):
    print(f'Estimate center information')
    df_centers = pd.read_csv(center_file, index_col='regionID_CCF')
    if ignore_lr:
        df_centers = df_centers[df_centers['right'] == 1]

    print(f'Loading the graph')
    with open(neighbor_file, 'rb') as fp:
        ndict = pickle.load(fp)

    col_name = get_col_by_axis(axis='AP')
    ana_dict = parse_ana_tree()

    print('Estimate distribution information')
    bssa = BrainsSignalAnalyzer(res_id=-3, plot=False)
    df_distr = pd.read_csv(precomputed_signal, index_col=0)
    format_precomputed_df(df_distr, last_column='modality')
    nreg = df_centers.shape[0]
    if nreg < 300:
        level = 1
    else:
        level = 0
    df_distr = bssa.map_to_coarse_regions(df_distr, level=level)

    print('Calculate brain-wide correlation coefficient')
    df_corr = df_distr.drop([last_column], axis=1)
    df_corr.replace(0, np.nan, inplace=True)
    corr = df_corr.corr(min_periods=min(df_corr.shape[0]//4, 10))
    corr.fillna(0, inplace=True)

    # make sure the regions are in the same order
    match = df_corr.columns.to_numpy() == df_centers.index.to_numpy()
    assert((~match).sum() == 0)
    # assign connection according to pairwise CC
    graph_with_corr(ndict, df_centers, corr, col_name, ana_dict)

def draw_AP_graph_with_label(center_file, distr_dir, neighbor_file, ignore_lr=True, last_column='modality', nr=70):
    print(f'Estimate center information')
    df_centers = pd.read_csv(center_file, index_col='regionID_CCF')
    if ignore_lr:
        df_centers = df_centers[df_centers['right'] == 1]

    print(f'Loading the graph')
    with open(neighbor_file, 'rb') as fp:
        ndict = pickle.load(fp)

    col_name = get_col_by_axis(axis='AP')

    print('Estimate distribution information')
    bssa = BrainsSignalAnalyzer(res_id=-3, plot=False)
    df_distr = bssa.parse_distrs(distr_dir)
    nreg = df_centers.shape[0]
    if nreg < 300:
        level = 1
    else:
        level = 0
    df_distr = bssa.map_to_coarse_regions(df_distr, level=level)
    ana_dict = bssa.ana_dict

    print('Calculate brain-wide correlation coefficient')
    df_distr = bssa.convert_modality_to_label(df_distr, normalize=True)
    
    # assign connection according to pairwise CC
    graph_for_label_merged(ndict, df_centers, df_distr, col_name, ana_dict)

def draw_AP_graph_for_somata(center_file, precomputed_file, neighbor_file, ignore_lr=True, nr=70, minimal_somata=100):
    """
    :params center_file: pre-calculated center for 70(76)/316(315) regions
    :params precomputed_file: precomputed somata distribution
    :params neighbor_file: direct neighboring regions
    :params ignore_lr: whether to discriminate left and right
    :params nr: deprecated
    :params minimal_somata: brains with somata less than minimal_somata will be discarded
    """
    print(f'Estimate center information')
    df_centers = pd.read_csv(center_file, index_col='regionID_CCF')
    if ignore_lr:
        df_centers = df_centers[df_centers['right'] == 1]

    print(f'Loading the graph')
    with open(neighbor_file, 'rb') as fp:
        ndict = pickle.load(fp)

    col_name = get_col_by_axis(axis='AP')

    print('Estimate distribution information')
    bssa = BrainsSignalAnalyzer(res_id=-3, plot=False)
    df_distr = bssa.load_somata(precomputed_file=precomputed_file)
    # drop brains with number of somata less than minimal_somata
    df_distr = df_distr[df_distr.sum(axis=1) > minimal_somata]
    # convert the dtype of column names to int
    rnames = [rname for rname in df_distr.columns if rname != 'modality']
    cname_mapper = dict(zip(df_distr, map(int, rnames)))
    df_distr.rename(columns=cname_mapper, inplace=True)
    df_distr.rename(index=str, inplace=True)

    nreg = df_centers.shape[0]
    if nreg < 300:
        level = 1
    else:
        level = 0
    
    df_distr = bssa.map_to_coarse_regions(df_distr, level=level)

    ana_dict = bssa.ana_dict

    print('Calculate brain-wide correlation coefficient')
    df_distr = bssa.convert_modality_to_label(df_distr, normalize=False)
    
    # assign connection according to pairwise CC
    graph_for_somata_merged(ndict, df_centers, df_distr, col_name, ana_dict)

def draw_AP_graph_corr(center_file, precomputed_signal, neighbor_file, ignore_lr=True, last_column='modality', nr=70):
    print(f'Estimate center information')
    df_centers = pd.read_csv(center_file, index_col='regionID_CCF')
    if ignore_lr:
        df_centers = df_centers[df_centers['right'] == 1]

    print(f'Loading the graph')
    with open(neighbor_file, 'rb') as fp:
        ndict = pickle.load(fp)

    col_name = get_col_by_axis(axis='AP')

    print('Estimate distribution information')
    bssa = BrainsSignalAnalyzer(res_id=-3, plot=False)
    df_distr = pd.read_csv(precomputed_signal, index_col=0)
    format_precomputed_df(df_distr, last_column='modality')

    nreg = df_centers.shape[0]
    if nreg < 300:
        level = 1
    else:
        level = 0
    df_distr = bssa.map_to_coarse_regions(df_distr, level=level)
    ana_dict = bssa.ana_dict

    print('Calculate brain-wide correlation coefficient')
    df_distr = bssa.convert_modality_to_label(df_distr, normalize=False)

    # assign connection according to pairwise CC
    #graph_for_corr_separate(ndict, df_centers, df_distr, col_name, ana_dict)
    #graph_for_correlated_regions(ndict, df_centers, df_distr, col_name, ana_dict)
    graph_for_all_modules(ndict, df_centers, df_distr, col_name, ana_dict)

if __name__ == '__main__':
    nr = 316
    center_file = f'./region_centers/region_centers_ccf25_r{nr}.csv'
    neighbor_file = f'./region_centers/regional_neibhgors_n{nr}.pkl'
    distr_dir = '/home/lyf/Research/cloud_paper/brain_statistics/statis_out/statis_out_adaThr_all'
    precomputed_somata = 'precomputed_somata.csv'
    precomputed_signal = 'precomputed_distrs.csv'
    
    #draw_AP_graph_with_label(center_file, distr_dir, neighbor_file, nr=nr)
    #draw_AP_graph_for_somata(center_file, precomputed_somata, neighbor_file, nr=nr, minimal_somata=100)
    draw_AP_graph_corr(center_file, precomputed_signal, neighbor_file, nr=nr)
    

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

from brain_analyzer_v2 import BrainsSignalAnalyzer

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

def graph_with_label_bak(ndict, df_centers, df_distr, col_name, ana_dict):
    G = nx.DiGraph()
    for k, v in ndict.items():
        for vi in v:
            ck = df_centers.at[k, col_name]
            cvi = df_centers.at[vi, col_name]
            if ck < cvi:
                G.add_edge(k, vi)
            else:
                G.add_edge(vi, k)

    # do transitive reduction
    TR = nx.transitive_reduction(G)
    TR.add_nodes_from(G.nodes(data=True))
    TR.add_edges_from((u, v, G.edges[u, v]) for u, v in TR.edges)

    # reweight the size of node according to its signaling
    cur_label = 'Calb2-CreERT2;Ai166'
    df_labels = df_distr[df_distr['label'] == cur_label].drop(['label'], axis=1)
    ncl = df_labels.shape[0]
    for bid in df_labels.index:
        print(f'==> Drawing {bid}')
        nodes = TR.nodes
        for rid in df_labels.columns:
            v = df_labels.at[bid, rid]
            pw = max(min(500 * v + 0.5, 10), 0.5)
            if rid not in nodes:
                continue
            nodes[rid]['penwidth'] = pw

        vis = True
        if vis:
            A = nx.nx_agraph.to_agraph(TR)  # convert to a graphviz graph
            A.draw(f"graph_brain{bid}_label{cur_label.replace(';', '_')}.png", prog='dot')  # Draw with pygraphviz

    return TR

def graph_with_label(ndict, df_centers, df_distr, col_name, ana_dict):

    def get_color(id_path, cmap):
        color = 'black'
        for idx in id_path:
            if idx in cmap:
                return cmap[idx]
        return color

    cmap = {
        688: 'orange', # CTX
        623: 'green', # CNU: STR + PAL
        512: 'magenta',  # CB, 
        343: 'cyan',  # BS: IB(549(TH)+HY), MB, HB
    }
    ctx = 688  # cortex, CTX

    G = nx.DiGraph()
    for k, v in ndict.items():
        id_path1 = ana_dict[k]['structure_id_path']
        #if ctx not in id_path1:
        #    continue
        for vi in v:
            id_path2 = ana_dict[vi]['structure_id_path']
            #if ctx not in id_path2:
            #    continue

            ck = df_centers.at[k, col_name]
            cvi = df_centers.at[vi, col_name]
            kname = ana_dict[k]['acronym']
            viname = ana_dict[vi]['acronym']
            if ck < cvi:
                G.add_edge(kname, viname)
            else:
                G.add_edge(viname, kname)

    # do transitive reduction
    TR = nx.transitive_reduction(G)
    TR.add_nodes_from(G.nodes(data=True))
    TR.add_edges_from((u, v, G.edges[u, v]) for u, v in TR.edges)

    labels = [l for l in np.unique(df_distr['label']) if l not in ('', 'Ai139')]

    for cur_label in labels:
        print(f'==> Processing for {cur_label}')
        df_labels = df_distr[df_distr['label'] == cur_label].drop(['label'], axis=1).median(axis=0)
        nodes = TR.nodes
        TR.graph['label'] = f'\n{cur_label}'
        for rid in df_labels.index:
            v = df_labels.at[rid]
            rname = ana_dict[rid]['acronym']
            if df_centers.shape[0] < 300:
                pw = max(min(125 * v + 0.5, 10), 0.5)
            else:
                pw = max(min(500 * v + 0.5, 10), 0.5)
            id_path = ana_dict[rid]['structure_id_path']
            color = get_color(id_path, cmap)
            if rname in nodes:
                nodes[rname]['penwidth'] = pw
                nodes[rname]['fillcolor'] = color
                nodes[rname]['style'] = 'filled'
        
        A = nx.nx_agraph.to_agraph(TR)  # convert to a graphviz graph
        lstr = cur_label.replace(';', '-').replace('+', '--')
        A.draw(f"graph_label{lstr}.png", prog='dot')  # Draw with pygraphviz   

    return TR

def draw_AP_graph_with_corr(center_file, distr_dir, neighbor_file, ignore_lr=True, last_column='modality', nr=70):
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
    df_distr = bssa.parse_distrs(distr_dir)
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
    ana_dict = parse_ana_tree()

    print('Estimate distribution information')
    bssa = BrainsSignalAnalyzer(res_id=-3, plot=False)
    df_distr = bssa.parse_distrs(distr_dir)
    nreg = df_centers.shape[0]
    if nreg < 300:
        level = 1
    else:
        level = 0
    df_distr = bssa.map_to_coarse_regions(df_distr, level=level)

    print('Calculate brain-wide correlation coefficient')
    df_distr = bssa.convert_modality_to_label(df_distr, normalize=True)
    
    # assign connection according to pairwise CC
    graph_with_label(ndict, df_centers, df_distr, col_name, ana_dict)

if __name__ == '__main__':
    nr = 70
    center_file = f'./region_centers/region_centers_ccf25_r{nr}.csv'
    neighbor_file = f'./region_centers/regional_neibhgors_n{nr}.pkl'
    distr_dir = '/home/lyf/Research/cloud_paper/brain_statistics/statis_out/statis_out_adaThr_all'
    
    draw_AP_graph_with_label(center_file, distr_dir, neighbor_file, nr=nr)
    

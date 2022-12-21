#!/usr/bin/env python

#================================================================
#   Copyright (C) 2022 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : axial_graph.py
#   Author       : Yufeng Liu
#   Date         : 2022-12-20
#   Description  : 
#
#================================================================

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN

from file_io import load_image
from anatomy.anatomy_core import parse_regions316

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


def axial_clustering(df_centers, axis='AP', vis=False, nclust=15):
    '''
    :params axis:       'AP', 'DV' and 'LR'
    '''
    col_name = get_col_by_axis(axis)
    axial_coords = df_centers[col_name].to_numpy()
    yp = np.ones(len(axial_coords))
    composite = np.vstack((axial_coords, yp)).transpose()
    
    fcluster = KMeans(nclust)
    clusters = fcluster.fit_predict(composite)
    
    if vis:
        for i in range(nclust):
            mask = clusters == i
            plt.scatter(axial_coords[mask], yp[mask], marker='o', label=f'cluster{i}')

        plt.ylim(0.98, 1.05)
        plt.xlabel(f'Coordinate along axis {axis}')
        plt.legend(ncol=3)
        plt.savefig(f'regions_axis{axis}.png', dpi=200)
        plt.close()

    return clusters
    
def write_graphviz(gfile, connections):
    with open(gfile, 'w') as fp:
        fp.write('digraph G {\n')
        fp.write('    fontname="Helvetica,Arial,sans-serif"\n')
        fp.write('    node [fontname="Helvetica,Arial,sans-serif"]\n')
        fp.write('    edge [fontname="Helvetica,Arial,sans-serif"]\n')
	
        for cm in connections:
            ids1, ids2 = np.nonzero(cm.to_numpy())
            regions1 = cm.index[ids1]
            regions2 = cm.columns[ids2]
            for r1, r2 in zip(regions1, regions2):
                fp.write(f'    {r1} -> {r2};\n')

        fp.write('}\n')
            

def get_connections(df_centers, clusters, corr, axis='AP', connection_thresh=0.5):
    # sort by cluster center
    col_name = get_col_by_axis(axis)
    nclust = len(np.unique(clusters))
    coords = df_centers[col_name].to_numpy()
    argids = np.argsort([coords[clusters==i].mean() for i in range(nclust)])

    # assign connection
    connections = []
    for pcid, ccid in zip(argids[:-1], argids[1:]):
        regions_p = np.nonzero(clusters == pcid)[0]
        regions_c = np.nonzero(clusters == ccid)[0]
        pids = df_centers.loc[regions_p]['regionID_CCF']
        cids = df_centers.loc[regions_c]['regionID_CCF']
        cur_cc = corr.loc[pids, cids]
        connections.append(cur_cc > connection_thresh)
    return connections
    

def draw_AP_graph(center_file, distr_dir, ignore_lr=True, nclust=15, last_column='modality'):
    print(f'Estimate center information')
    df_centers = pd.read_csv(center_file)
    if ignore_lr:
        df_centers = df_centers[df_centers['right'] == 1].reset_index(drop=True)
    clusters = axial_clustering(df_centers, axis='AP', nclust=nclust)

    print('Estimate distribution information')
    bssa = BrainsSignalAnalyzer(res_id=-3, plot=False)
    df_distr = bssa.parse_distrs(distr_dir)
    nreg = df_centers.shape[0]
    if nreg == 315:
        nreg = 316
    df_distr = bssa.map_to_coarse_regions(df_distr, num_regions=nreg)

    print('Calculate brain-wide correlation coefficient')
    df_corr = df_distr.drop([last_column], axis=1)
    corr = df_corr.corr(min_periods=10)
    corr.fillna(0)

    # make sure the regions are in the same order
    match = df_corr.columns.to_numpy() == df_centers['regionID_CCF'].to_numpy()
    assert((~match).sum() == 0)
    # assign connection according to pairwise CC
    connections = get_connections(df_centers, clusters, corr, axis='AP', connection_thresh=0.7)
    write_graphviz('temp_graphviz.txt', connections)



if __name__ == '__main__':
    center_file = './region_centers/region_centers_ccf25_r316.csv'
    distr_dir = '/home/lyf/Research/cloud_paper/brain_statistics/statis_out/statis_out_adaThr_all'
    
    draw_AP_graph(center_file, distr_dir)
    

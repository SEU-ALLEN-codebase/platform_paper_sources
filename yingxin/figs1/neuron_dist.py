#!/usr/bin/env python

#================================================================
#   Copyright (C) 2021 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : neuron_dist_vaa3d.py
#   Author       : Yufeng Liu
#   Date         : 2021-10-28
#   Description  : 
#
#================================================================

import os, sys, glob
import math
import numpy as np
import subprocess
from skimage.draw import line_nd
from scipy.spatial import distance_matrix

from swc_handler import parse_swc, write_swc, scale_swc, is_in_box



def memory_safe_min_distances(voxels1, voxels2, num_thresh=50000):
    nv1 = len(voxels1)
    nv2 = len(voxels2)
    if (nv1 > num_thresh) or (nv2 > num_thresh):
        vq1 = [voxels1[i*num_thresh:(i+1)*num_thresh] for i in range(int(math.ceil(nv1/num_thresh)))]
        vq2 = [voxels2[i*num_thresh:(i+1)*num_thresh] for i in range(int(math.ceil(nv2/num_thresh)))]
        dists1 = np.ones(nv1) * 1000000.
        dists2 = np.ones(nv2) * 1000000.
        for i,v1 in enumerate(vq1):
            idx00 = i * num_thresh
            idx01 = i * num_thresh + len(v1)
            for j,v2 in enumerate(vq2):
                idx10 = j * num_thresh
                idx11 = j * num_thresh + len(v2)
                d = distance_matrix(v1, v2)
                dists1[idx00:idx01] = np.minimum(d.min(axis=1), dists1[idx00:idx01])
                dists2[idx10:idx11] = np.minimum(d.min(axis=0), dists2[idx10:idx11])
    else:
        pdist = distance_matrix(voxels1, voxels2)
        dists1 = pdist.min(axis=1)
        dists2 = pdist.min(axis=0)
    return dists1, dists2
               
def calc_DMs(voxels1, voxels2, para):
    dist_results = {
        'ESA': -1,
        'DSA': -1,
        'PDS': -1}
    if len(voxels1) == 0 or len(voxels2) == 0: return dist_results
    
    dists1, dists2 = memory_safe_min_distances(voxels1, voxels2)
    for key in dist_results:
        if key == 'DSA':#far nodes distance
            dists1_ = dists1[dists1 > para]
            dists2_ = dists2[dists2 > para]
            if dists1_.shape[0] == 0:
                dists1_ = np.array([0.])
            if dists2_.shape[0] == 0:
                dists2_ = np.array([0.])
        elif key == 'PDS':#far nodes ratio
            dists1_ = (dists1 > para).astype(np.float32)
            dists2_ = (dists2 > para).astype(np.float32)
        elif key == 'ESA':#distance
            dists1_ = dists1
            dists2_ = dists2
        dist_results[key] = (dists1_.mean(), dists2_.mean(), (dists1_.mean() + dists2_.mean())/2.)
    return dist_results

def tree_to_voxels(tree, crop_box=(1000000,1000000,1000000)):
    pos_dict = {}
    new_tree = []
    for i, leaf in enumerate(tree):
        idx, type_, x, y, z, r, p = leaf
        leaf_new = (*leaf, is_in_box(x,y,z,crop_box))
        pos_dict[leaf[0]] = leaf_new
        new_tree.append(leaf_new)
    tree = new_tree

    xl, yl, zl = [], [], []
    for _, leaf in pos_dict.items():
        idx, type_, x, y, z, r, p, ib = leaf
        if p == -1: continue   
        if p not in pos_dict: continue
        parent_leaf = pos_dict[p]
        if (not ib) and (not parent_leaf[ib]):
            print('All points are out of box! do trim_swc before!')
            raise ValueError

        cur_pos = leaf[2:5]
        par_pos = parent_leaf[2:5]
        lin = line_nd(cur_pos[::-1], par_pos[::-1], endpoint=True)
        xl.extend(list(lin[2]))
        yl.extend(list(lin[1]))
        zl.extend(list(lin[0]))

    voxels = []
    for (xi,yi,zi) in zip(xl,yl,zl):
        if is_in_box(xi,yi,zi,crop_box):
            voxels.append((xi,yi,zi))
    voxels = np.array(list(set(voxels)), dtype=np.float32)
    return voxels

def get_specific_neurite(tree, type_id):
    if (not isinstance(type_id, list)) and (not isinstance(type_id, tuple)):
        type_id = (type_id,)
    new_tree = []
    for leaf in tree:
        if leaf[1] in type_id:
            new_tree.append(leaf)
    return new_tree

def calc_score(swc_file1, swc_file2, para, neurite_type='all', dist_type='DM'):
    if dist_type == 'DM':
        tree1 = parse_swc(swc_file1)
        tree2 = parse_swc(swc_file2)
        if neurite_type == 'all':
            pass
        elif neurite_type == 'dendrite':
            type_id = (3,4)
            tree1 = get_specific_neurite(tree1, type_id)
        elif neurite_type == 'axon':
            type_id = 2
            tree1 = get_specific_neurite(tree1, type_id)
        else:
            raise NotImplementedError

        voxels1 = tree_to_voxels(tree1)
        voxels2 = tree_to_voxels(tree2)
        dist = calc_DMs(voxels1, voxels2, para)
    return dist    

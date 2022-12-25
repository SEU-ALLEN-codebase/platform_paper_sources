#!/usr/bin/env python

#================================================================
#   Copyright (C) 2022 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : regional_arbors.py
#   Author       : Yufeng Liu
#   Date         : 2022-12-08
#   Description  : 
#
#================================================================

import os
import glob
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections  as mc
from matplotlib import colors as mcolors

from arbor_analyzer_r2 import load_soma_pos, ArborFeatures, load_swc_as_array

sys.path.append('../../common_lib')
from common_utils import load_pstype_from_excel, load_type_from_excel

def regional_mapping(soma_file, arbor_dir, prefixes, plot=True, stype=''):
    spos_dict = load_soma_pos(soma_file)
    pos_arr = []
    for prefix in prefixes:
        print(prefix)
        soma_pos = np.array(spos_dict[prefix])
        arbor_file = os.path.join(arbor_dir, f'{prefix}_axon.swc._m3_lt.eswc')
        try:
            swc_array = load_swc_as_array(arbor_file)
        except IOError:
            print(f'Error: {stype}')
            continue
        
        arbor_indices = np.unique(swc_array[:,1])
        num_arbors = len(arbor_indices)

        min_dist = 1000000
        for i in range(num_arbors):
            flag = swc_array[:,1] == i
            af = ArborFeatures(swc_array[flag], arbor_identifier='')
            af.max_density()
            mdc = af.max_density_coord
            #mdc = af.calc_center()
            #print(mdc, soma_pos)
            dist = np.linalg.norm(mdc - soma_pos)
            if min_dist > dist:
                min_dist = dist
                min_dist_pos = mdc
        
        # write to file
        pos_arr.append([soma_pos, min_dist_pos])

    pos_arr = np.array(pos_arr) # (-1, 2, 3)
    print(pos_arr.shape)

    if plot:
        # plot pathes
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111, projection='3d')
    
        #import ipdb; ipdb.set_trace()    
        for line_seg in pos_arr:
            ax.scatter(line_seg[:,0], line_seg[:,1], line_seg[:,2])
            ax.plot(line_seg[:,0], line_seg[:,1], line_seg[:,2], alpha=0.2)
        
        plt.title(stype)
        plt.savefig(f'{stype}.png')
        plt.close()


if __name__ == '__main__':
    soma_file = '../data/soma_pos.txt'
    arbor_dir = '../data/axon_arbors_round2_ln'
    celltype_file = '../../common_lib/41586_2021_3941_MOESM4_ESM.xlsx'

    soma_types, _, _ = load_type_from_excel(celltype_file, use_abstract_ptype=False)
    for stype in soma_types.keys():
        print(stype)
        prefixes = soma_types[stype]
        if len(prefixes) > 20:
            regional_mapping(soma_file, arbor_dir, prefixes, stype=stype)
    


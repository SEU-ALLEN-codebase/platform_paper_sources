#!/usr/bin/env python

#================================================================
#   Copyright (C) 2022 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : neurite_arbors.py
#   Author       : Yufeng Liu
#   Date         : 2022-09-26
#   Description  : 
#
#================================================================

import os
import glob
import numpy as np
import pandas as pd

from skimage.draw import line_nd
import matplotlib.pyplot as plt

from swc_handler import parse_swc, write_swc, get_specific_neurite, NEURITE_TYPES
from morph_topo import morphology


class NeuriteArbors:
    def __init__(self, swcfile):
        tree = parse_swc(swcfile)
        self.morph = morphology.Morphology(tree)
        self.morph.get_critical_points()

    def get_paths(self, mip='z'):
        """
        Tip: set
        """
        if mip == 'z':
            idx1, idx2 = 2, 3
        elif mip == 'x':
            idx1, idx2 = 3, 4
        elif mip == 'y':
            idx1, idx2 = 2, 4
        else:
            raise NotImplementedError

        paths = []
        for tip in self.morph.tips:
            path = []
            node = self.morph.pos_dict[tip]
            #if node[1] not in type_id: continue
            path.append([node[idx1], node[idx2]])
            while node[6] in self.morph.pos_dict:
                pid = node[6]
                pnode = self.morph.pos_dict[pid]
                path.append([pnode[idx1], pnode[idx2]])
                #if pnode[1] not in type_id:
                #    break
                node = self.morph.pos_dict[pid]

            paths.append(np.array(path))

        return paths

       
    def plot_morph_mip(self, xxyy=None, mip='z', color='r', figname='temp.png', out_dir='.'):
        paths = self.get_paths(mip=mip)
        
        plt.figure(figsize=(8,8))
        for path in paths:
            plt.plot(path[:,0], path[:,1], color=color, lw=5)
            
        try:
            all_paths = np.vstack(paths)
            if xxyy is None:
                xxyy = (all_paths.min(axis=0)[0], all_paths.max(axis=0)[0], all_paths.min(axis=0)[1], all_paths.max(axis=0)[1])

            plt.xlim([xxyy[0], xxyy[1]])
            plt.ylim([xxyy[2], xxyy[3]])
        except ValueError:
            pass

        plt.tick_params(left = False, right = False, labelleft = False ,
                labelbottom = False, bottom = False)
        # Iterating over all the axes in the figure
        # and make the Spines Visibility as False
        for pos in ['right', 'top', 'bottom', 'left']:
            plt.gca().spines[pos].set_visible(False)
        
        # title
        #plt.title(figname, fontsize=20)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'{figname}.png'), dpi=200)
        plt.close()
        

if __name__ == '__main__':
    swc_dir = '/PBshare/SEU-ALLEN/Users/yfliu/transtation/Research/platform/micro_environ/data/42k_local_morphology_gcoord_registered'
    lm_file = '../data/lm_features_d22_all.csv'
    neuron_type = 'MOp'
    out_dir = f'../data/mip_nodes500-1500_{neuron_type}'
    color = 'crimson'

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # filter with node number
    df = pd.read_csv(lm_file, index_col=0)
    df_n = df[(df['region_name_r316'] == neuron_type) & (df['Nodes'] >= 500) & (df['Nodes'] <= 1500)]
    print(df_n.shape)

    num = 0
    for irow, row in df_n.iterrows():
        swcfile = os.path.join(swc_dir, row['dataset_name'], str(row['brain_id']), irow)
        na = NeuriteArbors(swcfile)
        figname = os.path.split(swcfile)[-1][:-16]
        na.plot_morph_mip(color=color, figname=figname, out_dir=out_dir)
        
        num += 1
        if num % 10 == 0:
            print(f'--> processed: {num}, current: {swcfile}')
    



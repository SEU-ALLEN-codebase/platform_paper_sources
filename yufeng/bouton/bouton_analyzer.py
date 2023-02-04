#!/usr/bin/env python

#================================================================
#   Copyright (C) 2023 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : bouton_analyzer.py
#   Author       : Yufeng Liu
#   Date         : 2023-01-03
#   Description  : 
#
#================================================================

import os
import glob
import numpy as np
import pandas as pd

import seaborn as sns

from file_io import load_image
from swc_handler import parse_swc, scale_swc, get_specific_neurite, NEURITE_TYPES
from morph_topo.morphology import Morphology
from anatomy.anatomy_config import MASK_CCF25_FILE

class BoutonFeatures:
    def __init__(self, swcfile):
        tree = self.load_swc(swcfile)
        self.morph = Morphology(tree)

        self.axons = get_specific_neurite(tree, NEURITE_TYPES['axon'] + [5])
        self.boutons = get_specific_neurite(self.axons, 5)
        self.bids = set([bnode[0] for bnode in self.boutons])

    @staticmethod
    def load_swc(swcfile, scale = 25):
        tree = parse_swc(swcfile)
        tree = scale_swc(tree, scale)
        return tree

    def get_num_bouton(self):
        return len(self.boutons)

    def get_teb_ratio(self):
        num_teb = len([node for node in self.boutons if node[5] == 3])
        teb_ratio = num_teb / self.get_num_bouton()
        return teb_ratio

    def bouton_density_by_axon(self, frag_lengths_dict):
        num_bouton = self.get_num_bouton()
        axon_length = 0
        for node in self.axons:
            idx = node[0]
            axon_length += frag_lengths_dict[idx]
        den1 = num_bouton / axon_length * 100.
        return den1

    def geodesic_distances(self, path_dict, frag_lengths):
        plen_dict = self.morph.get_path_len_dict(path_dict, frag_lengths)
        try:
            lengths = [plen_dict[idx[0]] for idx in self.boutons]
        except:
            print('!!! Error!')
            lengths = [-1 for idx in self.boutons]
        return lengths

    def bouton_intervals(self, frag_lengths_dict):
        int_dict = {}
        for bnode in self.boutons:
            pid = bnode[6]
            cid = bnode[0]
            intervel = 0
            while pid in self.morph.pos_dict:
                intervel += frag_lengths_dict[cid]

                if pid in self.bids:
                    int_dict[cid] = intervel
                    break

                cid = pid
                pid = self.morph.pos_dict[pid][6]
        return int_dict

    def get_number_of_segments(self):
        if not hasattr(self.morph, 'tips'):
            self.morph.get_critical_points()

        return len(self.morph.tips) + len(self.morph.bifurcation) + len(self.morph.multifurcation)

    def get_targets(self, mask):
        bcoords = np.array([node[2:5] for node in self.boutons])
        bcoords = np.round(bcoords / 25.).astype(np.int32)
        bcoords = np.clip(bcoords, a_min=0, a_max=[mask.shape[2]-1, mask.shape[1]-1, mask.shape[0]-1])
        vm = mask[bcoords[:,2], bcoords[:,1], bcoords[:,0]]
        vu, vc = np.unique(vm, return_counts=True)
        return len(vu)
                

    def calc_overall_features(self, mask):
        frag_lengths, frag_lengths_dict = self.morph.calc_frag_lengths()
        path_dict = self.morph.get_path_idx_dict()

        num_bouton = self.get_num_bouton()
        teb_ratio = self.get_teb_ratio()
        print(f'#bouton={num_bouton}, teb_ratio={teb_ratio:.4f}')
        # overall density
        den1 = self.bouton_density_by_axon(frag_lengths_dict)
        print(f'overall density={den1:.4f} n/100um')
    
        # geodesic distances
        gdists = self.geodesic_distances(path_dict, frag_lengths)
        gdist = sum(gdists) / len(gdists)
        print(f'Average gdist={gdist:.4f} um')
        
        # interval
        int_dict = self.bouton_intervals(frag_lengths_dict)
        mean_interval = sum(int_dict.values()) / len(int_dict)
        print(f'Average interval={mean_interval:.4f}')
        
        # targets
        num_targets = self.get_targets(mask)
        print(f'Number of targeting regions: {num_targets}\n')

        # num of segments
        num_segs = self.get_number_of_segments()

        return num_bouton, teb_ratio, den1, gdist, mean_interval, num_targets, num_segs


def single_processor(swcfile, mask):
    fn = os.path.splitext(os.path.split(swcfile)[-1])[0]
    outfile = f'bouton_features/{fn}.txt'
    if os.path.exists(outfile):
        return

    bf = BoutonFeatures(swcfile)
    fs = bf.calc_overall_features(mask)
    with open(outfile, 'w') as fp:
        fp.write(fn)
        for fi in fs:
            fp.write(f', {fi}')
        fp.write('\n')
    
def calc_overall_features_all(bouton_dir):
    mask = load_image(MASK_CCF25_FILE)
 
    args_list = []
    for swcfile in glob.glob(os.path.join(bouton_dir, '*.swc')):   
        args_list.append((swcfile, mask))

    from multiprocessing import Pool
    pool = Pool(processes=24)
    pool.starmap(single_processor, args_list)
    pool.close()
    pool.join()


if __name__ == '__main__':
    bouton_dir = 'bouton_v20230110_swc'

    calc_overall_features_all(bouton_dir)

    



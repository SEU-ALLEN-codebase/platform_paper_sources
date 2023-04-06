#!/usr/bin/env python

#================================================================
#   Copyright (C) 2023 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : benchmark.py
#   Author       : Yufeng Liu
#   Date         : 2023-04-04
#   Description  : 
#
#================================================================
import numpy as np
import pandas as pd
from swc_handler import parse_swc, get_specific_neurite, NEURITE_TYPES, flip_swc
from neuron_quality.metrics import DistanceEvaluation
from morph_topo.morphology import Morphology

np.set_printoptions(precision=4)

def evaluate(match_file, dsa_thr=2., outfile='temp.csv', only_dendrite=False, flip_y=True):
    de = DistanceEvaluation(dsa_thr=dsa_thr, resample2=False)
    df = pd.read_csv(match_file)

    dmatrix = []
    for idx, row in df.iterrows():
        swc1 = row['path_x']
        swc2 = row['path_y']
        tree1 = parse_swc(swc1)
        if flip_y:
            tree1 = flip_swc(tree1, axis='y', dim=512)
        tree2 = parse_swc(swc2)

        nodes1 = len(tree1)
        if only_dendrite:
            dendrites = get_specific_neurite(tree2, NEURITE_TYPES['dendrite'])
            nodes2 = len(dendrites)
            ds = de.run(tree1, dendrites)
        else:
            nodes2 = len(tree2)
            ds = de.run(tree1, tree2)
        # path length
        morph1 = Morphology(tree1)
        morph2 = Morphology(tree2)
        pl1 = morph1.calc_total_length()
        if only_dendrite:
            seg_lengths, lengths_dict = morph2.calc_frag_lengths()
            is_dendrite = np.array([node[1] in [3,4] for node in tree2])
            pl2 = seg_lengths[is_dendrite].sum()
        else:
            pl2 = morph2.calc_total_length()
        
        if idx % 10 == 0:
            print(f'[{idx}] #nodes1={nodes1}, nodes2={nodes2}, metrics=\n{ds}\n')

        dmatrix.append([*ds[2], nodes1, nodes2, pl1, pl2])
    dmatrix = pd.DataFrame(dmatrix, columns=['pds12', 'pds21', 'pds', 'nodes1', 'nodes2', 'path_length1', 'path_length2'])
    dmatrix.to_csv(outfile, index=False)

    return dmatrix
    

if __name__ == '__main__':
    match_file = './utils/file_mapping1854.csv'
    dsa_thr = 4
    only_dendrite = False
    outfile = f'recon1854_dist{dsa_thr}.csv'

    evaluate(match_file, dsa_thr=dsa_thr, outfile=outfile, only_dendrite=only_dendrite)


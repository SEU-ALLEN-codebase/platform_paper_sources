#!/usr/bin/env python

#================================================================
#   Copyright (C) 2022 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : main_path_analyzer.py
#   Author       : Yufeng Liu
#   Date         : 2022-08-06
#   Description  : 
#
#================================================================

import os
import glob
import numpy as np

from sklearn.cross_decomposition import CCA
import matplotlib.pyplot as plt
import umap

from swc_handler import parse_swc
from morph_topo.morphology import Morphology

class MainPathAnalyzer(object):
    def __init__(self):
        pass

    def start_end_variance(self, path_swcs, class_name):
        termini = []
        somata = []
        for path_swc in path_swcs:
            tree = parse_swc(path_swc)
            terminal = tree[0][2:5]
            assert(tree[-1][-1] == -1)
            soma = tree[-1][2:5]

            termini.append(terminal)
            somata.append(soma)

        termini = np.array(termini)
        somata = np.array(somata)
        tstd = termini.std()
        sstd = somata.std()
        print(f'Std of somata, termini and their ratio are: {sstd}, {tstd}, {tstd/sstd}')

        # do Canonical Correlation Analysis (CCA)
        cca = CCA()
        #termini = (termini - termini.mean(axis=0)) / termini.std(axis=0)
        #somata = (somata - somata.mean(axis=0)) / somata.std(axis=0)
        cca.fit(termini, somata)
        t_c, s_c = cca.transform(termini, somata)

        coeff = np.corrcoef(t_c[:,0], s_c[:,0])[0,1]
        print(f'coeff: {coeff}')

        fig, axes = plt.subplots(1,2)
        plt.suptitle(f'{class_name}(num={len(path_swcs)})')
        # visualization with the first PC
        print(t_c[:,0].shape)
        axes[0].scatter(t_c[:,0], s_c[:,0])
        axes[0].set_title(f'Corr(Termini,Somata)={coeff:.2f}')
        axes[0].set_xlabel('CC1 of Termini')
        axes[0].set_ylabel('CC1 of Somata')

        # visualization with UMAP
        ts = np.hstack((termini, somata))
        # do whitening
        ts_whitened = (ts - ts.mean(axis=0)) / ts.std(axis=0)
        # UMAP reduction
        n_neighbors = min(15, len(ts_whitened)//2)
        reducer = umap.UMAP(n_neighbors=n_neighbors)
        embedding = reducer.fit_transform(ts_whitened)
        np.savetxt(f'{class_name}_umap_embedding.txt', embedding, fmt='%.4f')
        axes[1].scatter(
                        embedding[:, 0],
                        embedding[:, 1],
                        )
        axes[1].set_title(f'UMAP(Somata,termini)')

        fig.tight_layout()
        figname = f'{class_name}_somata_termini_corr.png'
        fig.savefig(figname, dpi=150)
        plt.close()


if __name__ == '__main__':
    ctypes = ['ACB', 'Car3_AId', 'Car3_SSs', 'CLA', 'CP_GPe', 'CP_others', 'CP_SNr', 'ET_MOp', 'ET_MOs', 'ET_RSP', 'ET_SSp-bfd', 'ET_SSp-m', 'ET_SSp-n', 'ET_SSp-ul', 'ET_SSs', 'IT_MOp', 'IT_MOs', 'IT_SSp-bfd', 'IT_SSp-m', 'IT_SSp-n', 'IT_SSs', 'IT_VIS', 'LD', 'LGd', 'LP', 'MG', 'RT', 'SMT', 'VM', 'VPL', 'VPLpc', 'VPM']
    
    for ctype in ctypes:
        path_swcs = glob.glob(os.path.join(f'../main_tracts_types/{ctype}*tract.swc'))
        print(f'--> {ctype}: {len(path_swcs)}')
        mpa = MainPathAnalyzer()
        mpa.start_end_variance(path_swcs, ctype)



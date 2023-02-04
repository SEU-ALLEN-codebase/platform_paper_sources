#!/usr/bin/env python

#================================================================
#   Copyright (C) 2023 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : analyze.py
#   Author       : Yufeng Liu
#   Date         : 2023-02-03
#   Description  : 
#
#================================================================

import numpy as np
import pandas as pd

def calc_ctx_diff_ptype(csvfile, icols=(0, 3, 13, 22)):
    df = pd.read_csv(csvfile, index_col=0)
    
    ds_self = np.diag(df).mean()
    ds_intraclass = 0
    n_intra = 0
    for i in range(len(icols)-1):
        for j in range(len(icols)-1):
            if i == j:
                sub = df.iloc[icols[i]:icols[i+1], icols[j]:icols[j+1]]
                n_intra += sub.size
                ds_intraclass += sub.sum().sum()
    ds_interclass = (df.sum().sum() - ds_intraclass) / df.size
    ds_intraclass /= n_intra
    return ds_self, ds_intraclass, ds_interclass



if __name__ == '__main__':
    print('Ptype: ')
    for level in ['fullMorpho', 'arbor', 'motif', 'bouton']:
        ptype_csv = f'./withPtype/corr_regionLevel_sdmatrix_{level}_ctx_withPtype.csv'
        ds_self, ds_intraclass, ds_interclass = calc_ctx_diff_ptype(ptype_csv)
        print(f'[{level}] DS_self={ds_self:.3f}, DS_intraclass={ds_intraclass:.3f}, DS_interclass={ds_interclass:.3f}')

    print('\n\nCortical Layer:')
    for level in ['fullMorpho', 'arbor', 'motif', 'bouton']:
        ptype_csv = f'./withCorticalLayer/corr_regionLevel_sdmatrix_{level}_ctx_withLayer.csv'
        ds_self, ds_intraclass, ds_interclass = calc_ctx_diff_ptype(ptype_csv, icols=(0,6,11,22,24))
        print(f'[{level}] DS_self={ds_self:.3f}, DS_intraclass={ds_intraclass:.3f}, DS_interclass={ds_interclass:.3f}')

                


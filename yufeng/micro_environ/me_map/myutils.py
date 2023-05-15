#!/usr/bin/env python

#================================================================
#   Copyright (C) 2023 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : myutils.py
#   Author       : Yufeng Liu
#   Date         : 2023-05-11
#   Description  : 
#
#================================================================

import pandas as pd

# broadcast the ptype and ctype to other section
ref_file = './data/lm_features_d22_15441_with_ptype_cstype.csv'
subject_file = './data/micro_env_features_nodes300-1500_statis.csv'
outfile = subject_file[:-4] + '_with_ptype_cstype.csv'

ref = pd.read_csv(ref_file, index_col=0)
sub = pd.read_csv(subject_file, index_col=0)
common = [*(set(ref.index) & set(sub.index))]

out = sub.loc[common].copy()
out.loc[:, ['ptype', 'cstype']] = ref.loc[common, ['ptype', 'cstype']]
out.to_csv(outfile)


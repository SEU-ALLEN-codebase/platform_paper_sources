#!/usr/bin/env python

#================================================================
#   Copyright (C) 2023 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : micro_env_features.py
#   Author       : Yufeng Liu
#   Date         : 2023-01-18
#   Description  : 
#
#================================================================
import time
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix

import sys
sys.path.append('../src')
from config import __FEAT_NAMES__

class MEFeatures:
    def __init__(self, feature_file, region_num='316'):
        self.region_num = region_num    # not the actual number of regions
        self.df = self.load_features(feature_file)

    def load_features(self, feature_file, remove_nan=True, min_num_recons=10):
        df = pd.read_csv(feature_file, index_col=0)
        print(f'Initial number of recons: {df.shape[0]}')
        if remove_nan:
            err_f = ~(df.isna().sum(axis=1).astype(np.bool))
            df = df[err_f]
        print(f'Number after remove_nan: {df.shape[0]}')

        # keep regions with at leat `min_num_recons` neurons
        rkey = f'region_id_r{self.region_num}'
        regions, counts = np.unique(df[rkey], return_counts=True)
        abd_regions = regions[counts >= min_num_recons]
        df = df[df[rkey].isin(abd_regions)]
        print(f'Number after removing regions less thant {min_num_recons}: {df.shape[0]}')

        return df

    def calc_micro_env_features(self, mefeature_file, topk=5, nodes_range=(500, 1500), 
                                min_num_recons=10):
        # filter out the regions with number of nodes out of range `nodes_range`
        nmin, nmax = nodes_range
        df = self.df[(self.df['Nodes'] >= nmin) & (self.df['Nodes'] <= nmax)]
        print(f'Number of samples after nodes pruning: {df.shape[0]}')
        # should re-filtering the data
        rkey = f'region_id_r{self.region_num}'
        regions, counts = np.unique(df[rkey], return_counts=True)
        abd_regions = regions[counts >= min_num_recons]
        df = df[df[rkey].isin(abd_regions)]
        print(f'Number of samples after regional counts: {df.shape[0]}')

        df_mef = df.copy()
        feat_names = __FEAT_NAMES__ + ['pc11', 'pc12', 'pc13', 'pca_vr1', 'pca_vr2', 'pca_vr3']
        mefeat_names = [f'{fn}_me' for fn in feat_names]

        # we should pre-normalize each feature for topk extraction
        feats = df.loc[:, feat_names]
        feats = (feats - feats.mean()) / (feats.std() + 1e-10)
        df[feat_names] = feats

        # Use the statistical features of the topk of neurons with the same region
        rkey = f'region_id_r{self.region_num}'
        regions = np.unique(df[rkey])
        fdim = len(feat_names)
        
        df_mef[mefeat_names] = 0
        print(df_mef.shape)
        t0 = time.time()
        for region in regions:
            #if region != 985: continue  # for test only!
            print(f'==> Processing region: {region}')
            # all neurons in current region
            region_mask = df[rkey] == region
            region_index = df_mef.index[region_mask]
            # pairwise distance estimation
            feat = df.loc[region_index, feat_names]
            pdist = distance_matrix(feat, feat)
            indices_topk = np.argpartition(pdist, topk+1, axis=1)[:, :topk+1]
            # get all features at once
            mef_raw = self.df.loc[region_index, feat_names].iloc[indices_topk.reshape(-1)].to_numpy().reshape(-1, topk+1, fdim)
            mef_mean = mef_raw.mean(axis=1)
            #mef_median = np.median(mef_raw, axis=1)
            #mef_std = mef_raw.std(axis=1)
            # micro_env_features
            #mef = np.stack((mef_mean, mef_median, mef_std), axis=1).reshape((mef_mean.shape[0],-1))
            df_mef.loc[region_index, mefeat_names] = mef_mean
        print(f'--> Total time used: {time.time() - t0:.2f} sec')

        df_mef.to_csv(mefeature_file, float_format='%g')

       

if __name__ == '__main__':
    if 1:
        nodes_range = (300, 1500)
        feature_file = './data/lm_features_d22_all.csv'
        mefile = f'./data/micro_env_features_nodes{nodes_range[0]}-{nodes_range[1]}_withoutNorm.csv'
        mef = MEFeatures(feature_file)
        mef.calc_micro_env_features(mefile, nodes_range=nodes_range)
    
    
        

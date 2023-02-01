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

from config import __FEAT_NAMES__, __FEAT_ALL__

class MEFeatures:
    def __init__(self, feature_file, region_num='316'):
        self.region_num = region_num    # not the actual number of regions
        self.df = self.load_features(feature_file)

    def load_features(self, feature_file, remove_nan=True, min_num_recons=20):
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

    def calc_micro_env_features(self, mefeature_file, topk=5, nodes_range=(500, 1500), min_num_recons=10):
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

        # we should pre-normalize each feature for topk extraction
        feats = df.loc[:, __FEAT_NAMES__]
        feats = (feats - feats.mean()) / (feats.std() + 1e-10)
        df[__FEAT_NAMES__] = feats

        # Use the statistical features of the topk of neurons with the same region
        rkey = f'region_id_r{self.region_num}'
        regions = np.unique(df[rkey])
        fdim = len(__FEAT_NAMES__)
        
        df_mef[__FEAT_ALL__] = 0
        print(df_mef.shape)
        t0 = time.time()
        for region in regions:
            if region != 985: continue  # for test only!
            print(f'==> Processing region: {region}')
            # all neurons in current region
            region_mask = df[rkey] == region
            region_index = df_mef.index[region_mask]
            # pairwise distance estimation
            feat = df.loc[region_index, __FEAT_NAMES__]
            pdist = distance_matrix(feat, feat)
            indices_topk = np.argpartition(pdist, topk+1, axis=1)[:, :topk+1]
            # get all features at once
            mef_raw = self.df.loc[region_index, __FEAT_NAMES__].iloc[indices_topk.reshape(-1)].to_numpy().reshape(-1, topk+1, fdim)
            import ipdb; ipdb.set_trace()
            mef_mean = mef_raw.mean(axis=1)
            mef_median = np.median(mef_raw, axis=1)
            mef_std = mef_raw.std(axis=1)
            # micro_env_features
            mef = np.stack((mef_mean, mef_median, mef_std), axis=1).reshape((mef_mean.shape[0],-1))
            df_mef.loc[region_index, __FEAT_ALL__] = mef
        print(f'--> Total time used: {time.time() - t0:.2f} sec')

        # normalize by removing background
        feats = df_mef.loc[:, __FEAT_ALL__]
        df_mef.loc[:, __FEAT_ALL__] = (feats - feats.mean()) / (feats.std() + 1e-10)

        df_mef.to_csv(mefeature_file, float_format='%g')

    def calc_regional_mefeatures(self, mefeature_file, rmefeature_file):
        mef = pd.read_csv(mefeature_file, index_col=0)
        print(f'Feature shape: {mef.shape}')
        # calculate the mean and average features
        rkey = f'region_id_r{self.region_num}'
        rnkey = f'region_name_r{self.region_num}'
        regions = np.unique(mef[rkey])
        output = []
        index = []
        for region in regions:
            region_index = mef.index[mef[rkey] == region]
            feat = mef.loc[region_index, __FEAT_ALL__]
            rid = mef.loc[region_index[0], rkey]
            rname = mef.loc[region_index[0], rnkey]
            struct = mef.loc[region_index[0], 'brain_structure']
            fmean = feat.mean().to_numpy().tolist()
            fstd = feat.std().to_numpy().tolist()
            index.append(rid)
            output.append([rname, struct, len(region_index), *fmean, *fstd])
        
        columns = [rnkey, 'brain_structure', 'NumRecons']
        columns.extend([f'{fn}_mean' for fn in __FEAT_ALL__])
        columns.extend([f'{fn}_std' for fn in __FEAT_ALL__])
        rmef = pd.DataFrame(output, index=index, columns=columns)
        rmef.to_csv(rmefeature_file, float_format='%g')

    def calc_regional_single_features(self, mefeature_file, sfeature_file):
        rkey = f'region_id_r{self.region_num}'
        rnkey = f'region_name_r{self.region_num}'
        mef = pd.read_csv(mefeature_file, index_col=0)
        print(f'Feature shape: {mef.shape}')
        # normalize
        mef = mef[[rkey, rnkey, 'brain_structure', *__FEAT_NAMES__]]
        temp = mef.loc[:, __FEAT_NAMES__]
        mef.loc[:, __FEAT_NAMES__] = (temp - temp.mean()) / (temp.std() + 1e-10)

        regions = np.unique(mef[rkey])
        output = []
        index = []
        for region in regions:
            region_index = mef.index[mef[rkey] == region]
            feat = mef.loc[region_index, __FEAT_NAMES__]
            rid = mef.loc[region_index[0], rkey]
            rname = mef.loc[region_index[0], rnkey]
            struct = mef.loc[region_index[0], 'brain_structure']
            fmean = feat.mean().to_numpy().tolist()
            index.append(rid)
            output.append([rname, struct, len(region_index), *fmean])
        
        columns = [rnkey, 'brain_structure', 'NumRecons']
        columns.extend(__FEAT_NAMES__)
        rmef = pd.DataFrame(output, index=index, columns=columns)
        rmef.to_csv(sfeature_file, float_format='%g')
            
        

if __name__ == '__main__':
    nodes_range = (500, 1500)
    feature_file = '../data/lm_features_d22_all.csv'
    mefeature_file = f'../data/micro_env_features_d66_nodes{nodes_range[0]}-{nodes_range[1]}.csv'
    rmefeature_file = f'{mefeature_file[:-4]}_regional.csv'
    mef = MEFeatures(feature_file)
    mef.calc_micro_env_features(mefeature_file, nodes_range=nodes_range)
    #mef.calc_regional_mefeatures(mefeature_file, rmefeature_file)
    
    # temporal, for test only!
    #mef.calc_regional_single_features(mefeature_file, f'non-environ_features_nodes{nodes_range[0]}-{nodes_range[1]}.csv')
        

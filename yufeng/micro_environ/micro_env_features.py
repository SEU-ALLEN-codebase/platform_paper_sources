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

class MEFeatures:
    __FEAT_NAMES__ = [
        'Nodes', 'SomaSurface', 'Stems', 'Bifurcations',
        'Branches', 'Tips', 'OverallWidth', 'OverallHeight', 'OverallDepth',
        'AverageDiameter', 'Length', 'Surface', 'Volume',
        'MaxEuclideanDistance', 'MaxPathDistance', 'MaxBranchOrder',
        'AverageContraction', 'AverageFragmentation',
        'AverageParent-daughterRatio', 'AverageBifurcationAngleLocal',
        'AverageBifurcationAngleRemote', 'HausdorffDimension']
    __FEAT_ALL__ = [
        'Nodes_mean', 'SomaSurface_mean', 'Stems_mean', 'Bifurcations_mean', 
        'Branches_mean', 'Tips_mean', 'OverallWidth_mean', 'OverallHeight_mean', 
        'OverallDepth_mean', 'AverageDiameter_mean', 'Length_mean', 'Surface_mean', 
        'Volume_mean', 'MaxEuclideanDistance_mean', 'MaxPathDistance_mean', 
        'MaxBranchOrder_mean', 'AverageContraction_mean', 'AverageFragmentation_mean', 
        'AverageParent-daughterRatio_mean', 'AverageBifurcationAngleLocal_mean', 
        'AverageBifurcationAngleRemote_mean', 'HausdorffDimension_mean',
        'Nodes_median', 'SomaSurface_median', 'Stems_median', 'Bifurcations_median', 
        'Branches_median', 'Tips_median', 'OverallWidth_median', 'OverallHeight_median', 
        'OverallDepth_median', 'AverageDiameter_median', 'Length_median', 'Surface_median', 
        'Volume_median', 'MaxEuclideanDistance_median', 'MaxPathDistance_median', 
        'MaxBranchOrder_median', 'AverageContraction_median', 'AverageFragmentation_median', 
        'AverageParent-daughterRatio_median', 'AverageBifurcationAngleLocal_median', 
        'AverageBifurcationAngleRemote_median', 'HausdorffDimension_median',
        'Nodes_std', 'SomaSurface_std', 'Stems_std', 'Bifurcations_std', 
        'Branches_std', 'Tips_std', 'OverallWidth_std', 'OverallHeight_std', 
        'OverallDepth_std', 'AverageDiameter_std', 'Length_std', 'Surface_std', 
        'Volume_std', 'MaxEuclideanDistance_std', 'MaxPathDistance_std', 
        'MaxBranchOrder_std', 'AverageContraction_std', 'AverageFragmentation_std', 
        'AverageParent-daughterRatio_std', 'AverageBifurcationAngleLocal_std', 
        'AverageBifurcationAngleRemote_std', 'HausdorffDimension_std'
        ]

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

    def calc_micro_env_features(self, mefeature_file, topk=5):
        # we should pre-normalize each feature for topk extraction
        df = self.df.copy()
        feats = df.loc[:, self.__FEAT_NAMES__]
        feats = (feats - feats.mean()) / (feats.std() + 1e-10)
        df[self.__FEAT_NAMES__] = feats

        # Use the statistical features of the topk of neurons with the same region
        rkey = f'region_id_r{self.region_num}'
        regions = np.unique(df[rkey])
        fdim = len(self.__FEAT_NAMES__)
        df_mef = self.df.copy() #df.drop(self.__FEAT_NAMES__, axis=1)
        
        df_mef[self.__FEAT_ALL__] = 0
        print(df_mef.shape)
        t0 = time.time()
        for region in regions:
            print(f'==> Processing region: {region}')
            # all neurons in current region
            region_mask = df[rkey] == region
            region_index = df_mef.index[region_mask]
            # pairwise distance estimation
            feat = df.loc[region_index, self.__FEAT_NAMES__]
            pdist = distance_matrix(feat, feat)
            indices_topk = np.argpartition(pdist, topk+1, axis=1)[:, :topk+1]
            # get all features at once
            mef_raw = self.df.loc[region_index, self.__FEAT_NAMES__].iloc[indices_topk.reshape(-1)].to_numpy().reshape(-1, topk+1, fdim)
            mef_mean = mef_raw.mean(axis=1)
            mef_median = np.median(mef_raw, axis=1)
            mef_std = mef_raw.std(axis=1)
            # micro_env_features
            mef = np.stack((mef_mean, mef_median, mef_std), axis=1).reshape((mef_mean.shape[0],-1))
            df_mef.loc[region_index, self.__FEAT_ALL__] = mef
        print(f'--> Total time used: {time.time() - t0:.2f} sec')

        # normalize by removing background
        feats = df_mef.loc[:, self.__FEAT_ALL__]
        df_mef.loc[:, self.__FEAT_ALL__] = (feats - feats.mean()) / (feats.std() + 1e-10)

        df_mef.to_csv(mefeature_file, float_format='%g')

    def calc_normalized_regional_mefeatures(self, mefeature_file, nmefeature_file):
        pass

if __name__ == '__main__':
    feature_file = '../data/lm_features_d22_all.csv'
    mefeature_file = '../data/micro_env_features_d66_all.csv'
    nmefeature_file = '../data/micro_env_features_d66_all_normalized.csv'
    mef = MEFeatures(feature_file)
    #mef.calc_micro_env_features(mefeature_file)
    mef.calc_normalized_regional_mefeatures(mefeature_file, nmefeature_file)
        

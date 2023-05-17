#!/usr/bin/env python

#================================================================
#   Copyright (C) 2022 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#
#   Filename     : get_axonal_hubs.py
#   Author       : Yufeng Liu
#   Date         : 2022-10-19
#   Description  :
#
#================================================================

import os
import glob
import sys
import math
import pandas as pd
from itertools import permutations
import pickle

import numpy as np
from scipy.spatial import distance_matrix
from sklearn import decomposition
from skimage import morphology
import skimage.io, skimage.transform

import cc3d
import hdbscan
import umap

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

sys.path.append('../../common_lib')
from common_utils import load_celltypes, struct_dict, stype2struct
from select_neuron_types import select_neurons_by_stype, select_neurons_by_sptype, select_neurons_by_sltype


matplotlib.rcParams['font.sans-serif'] = "Arial"
matplotlib.rcParams['font.size'] = 15


def load_swc_as_array(swcfile):
    data = np.loadtxt(swcfile, usecols=(0,1,2,3,4,5,6), comments='#')
    return data

class AxonalHubs(object):
    def __init__(self, swcfile, fn=None, density_dist_thresh=20, scaling=25.):
        if type(swcfile) is str:
            self.swc_array = load_swc_as_array(swcfile)
        else:
            self.swc_array = np.array(swcfile)
        self.swc_array[:, 2:5] = self.swc_array[:, 2:5] / scaling
        
        self.density_dist_thresh = density_dist_thresh
        self.fn = fn

    def calc_density(self, return_max_info=True, density_dist_th=None):
        if density_dist_th is None:
            density_dist_th = self.density_dist_thresh

        coords = self.swc_array[:, 2:5]
        pdists = distance_matrix(coords, coords)
        density = (pdists < density_dist_th).sum(axis=0)
        if return_max_info:
            self.max_density = round(density.max(), self.ndigits)
            max_id = np.argmax(density)
            self.max_density_coord = coords[max_id]
        return density

    def get_soma_pos(self):
        soma_mask = self.swc_array[:, -1] == -1
        return self.swc_array[soma_mask][0,2:5]

    def detect_hubs(self):
        """
        A hub is defined as:
            - connective regions high density nodes
        Algorithms:
            1. convert the swc into 3D image space
            2. get all the high density nodes
            3. Expand each node
        """
        density_dist_th = 20
        density_th = 70
        density = self.calc_density(return_max_info=False, density_dist_th=density_dist_th)
        high_density = density > density_th
        if high_density.sum() == 0:
            return [(np.nan, np.nan, np.nan)]
        #print(f'High density ratio: {1.0*high_density.sum()/len(density)}')
        
        coords = np.round(self.swc_array[:, 2:5]).astype(np.int)[:, ::-1]   # to zyx order
        cmin = coords.min(axis=0)
        cmax = coords.max(axis=0)
        coords = coords - cmin
        bbox = cmax - cmin + 1
        # initialize an image3D with bbox
        img = np.zeros(bbox, dtype=np.uint8)
        dimg = np.zeros(bbox, dtype=np.uint16)
        coords_high_density = coords[high_density]
        
        img[tuple(zip(*coords_high_density))] = 1    # masking the points
        dimg[tuple(zip(*coords_high_density))] = density[high_density]
        # dilation or expansion
        k = 10
        img_dil = morphology.dilation(img, np.ones((k,k,k)))
        ccs, N = cc3d.connected_components(img_dil, return_N=True)
        # features
        features = []
        for i in range(N):
            cci_mask = ccs == i+1
            max_density = dimg[cci_mask].max()
            volume = cci_mask.sum()
            soma_pos = self.get_soma_pos()
            zz, yy, xx = np.nonzero(cci_mask)
            zc, yc, xc = zz.mean(), yy.mean(), xx.mean()
            center = np.array([xc, yc, zc])
            d2s = np.linalg.norm(soma_pos - center)
            features.append((d2s, volume, max_density))
            

        # for visualization
        visualize = False
        if visualize:
            vis = np.zeros(bbox, dtype=np.uint8)
            vis[img_dil == 1] = 45
            vis[tuple(zip(*coords))] = 128
            vis[tuple(zip(*coords_high_density))] = 255
            vis = vis.max(axis=0)
            vis = vis[::-1] #vertical flip
            skimage.io.imsave(f"./hubs/{self.fn}.png", skimage.transform.rescale(vis, 4, order=0))

        return features

# deprecated
def hub_clustering(rdict):
    all_features = []
    for fn, features in rdict.items():
        if np.isnan(features[0][0]):
            continue
        all_features.extend(features)
    # standardize
    fts = np.array(all_features)
    fts = (fts - fts.mean(axis=0)) / (fts.std(axis=0) + 1e-10)
    # DBSCAN clustering
    db = hdbscan.HDBSCAN().fit(fts)

    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.labels_ != -1] = True
    labels = db.labels_

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print(f"Estimated number of clusters: {n_clusters_:d}")
    print(f"Estimated number of noise points: {n_noise_:d}")

    plot = True
    if plot:
        # plotting
        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
        # map the features to 2D for better visualization
        reducer = umap.UMAP()
        embedding = reducer.fit_transform(fts)
        print(embedding.shape)

        for k, col in zip(unique_labels, colors):
            if k == -1:
                # black for noise
                col = [0, 0, 0, 1]

            class_member_mask = labels == k
            print(f'==> Class {k} has #samples={class_member_mask.sum()}')

            xy = embedding[class_member_mask & core_samples_mask]
            plt.plot(
                xy[:,0],
                xy[:,1],
                "o",
                c=tuple(col),
                markersize=1,
                label = f"class={k}"
            )

            xy = embedding[class_member_mask & ~core_samples_mask]
            plt.plot(
                xy[:,0],
                xy[:,1],
                "o",
                c=tuple(col),
                markersize=1
            )
        plt.title('todel.csv')
        plt.legend()

        figname = 'temp.png'
        plt.savefig(figname, dpi=300)


class PlotHubs(object):
    def __init__(self, rdict, cellfile, nb=10):
        self.rdict = rdict
        self.minima, self.maxima = self.get_feat_minmax()

        self.df_ct = pd.read_csv(cellfile, index_col=0)
        self.nb = nb

    
    def get_feat_minmax(self):
        features = []
        for fn, feats in self.rdict.items():
            if np.isnan(feats[0][0]):
                continue
            features.extend(feats)
        
        features = np.array(features)
        print(features.shape)
        minima = features.min(axis=0)
        maxima = np.percentile(features, 95, axis=0)
        return minima, maxima

    def vectorize_feature(self, fts):
        fts = np.array(fts)
        
        nf = len(fts)
        v = np.zeros((self.nb, nf))
        if np.isnan(fts[0]):
            v[0] = 1
            return v
        else:
            ib = np.minimum(np.floor(fts / self.maxima * self.nb).astype(int), self.nb-1)
            v[ib, np.arange(nf)] = 1
            return v

    def vectorize_features(self, features):
        for i, fts in enumerate(features):
            if i != 0:
                v += self.vectorize_feature(fts)
            else:
                v = self.vectorize_feature(fts)
        return v

    def plot_by_types(self, r2n_dict, regions, figname, show_area=False, area_colors=None, area_labels=None):
        df_hf = []
        names = []
        nums = []
        for region in regions:
            neurons = r2n_dict[region]
            print(f'[region: {region}]: {len(neurons)}')
            for i, neuron in enumerate(neurons.values):
                if neuron not in self.rdict: 
                    continue
                features = self.rdict[neuron]
                if i == 0:
                    v = self.vectorize_features(features)
                else:
                    v += self.vectorize_features(features)
            v /= (i+1)
            vsum = v.sum() / v.shape[1]
            nums.append(vsum)
            v /= vsum
            names.append(region)
            df_hf.append(v.transpose().reshape(-1))
            
        df_hf = np.array(df_hf)
        df_hf = pd.DataFrame(df_hf.transpose(), columns=names)
        print(f'Avg number of hubs in each category: {nums}')
        #nums = np.array(nums)
        #nums = nums / nums.max()
        #df_hf.loc[len(df_hf.index)] = nums

        # set index and rows
        indices = []
        row_colors = []
        ftnames = ['D2S', 'Volume', 'Density*']
        rcolors = ['g', 'b', 'm']
        rclabels = ['Distance-to-soma', 'Volume', 'Max density']
        for i, ftname in enumerate(ftnames):
            for pct in np.arange(10,110,10):
                indices.append(f'{ftname}-p{pct}')
                row_colors.append(rcolors[i])

        df_hf.rename(index=dict(zip(df_hf.index, indices)), inplace=True)

        if not show_area:
            area_colors = None
        cm = sns.clustermap(df_hf, cmap='Reds', yticklabels=1, xticklabels=1, row_cluster=False,
                            col_cluster=False, row_colors=row_colors, 
                            col_colors=area_colors, cbar_kws=dict(orientation='horizontal'))
        # row colors
        for i, rclabel, c in zip(np.arange(3), rclabels[::-1], rcolors):
            cm.ax_row_colors.axes.text(-0.4, (2*i+1)/6., rclabel, ha='center', va='center',
                    transform=cm.ax_row_colors.axes.transAxes, rotation=90., color=c)
        cm.ax_row_colors.axes.set_xticks(ticks=[0.5])
        cm.ax_row_colors.axes.set_xticklabels(['Feature'], rotation=90.)
       
        if show_area:
            # col colors
            cm.ax_col_colors.axes.set_yticks([0.5])
            cm.ax_col_colors.axes.set_yticklabels(['Brain area'])
            cm.ax_col_colors.axes.tick_params(left=False, right=True, labelleft=False, labelright=True)

        # colorbar
        cm.cax.set_position([0.85, 0.07, .1, .02])
        cm.cax.set_xlabel('Normalized feature')
        #cm.cax.set_yticks(np.arange(-2,3), np.arange(-2,3), fontsize=rc_fontsize)
        #cm.cax.tick_params(left=False, right=True, labelleft=False, labelright=True, direction='in')
        #cm.cax.yaxis.set_label_position("left")

        #plt.subplots_adjust(left=0.2, bottom=0.2)
        plt.savefig(figname, dpi=300)
        plt.close()

    def plot_by_stype(self, figname):
        r2n_dict, regions = select_neurons_by_stype(self.rdict, self.df_ct)
        rgbs = {
            'CTX': 'orange',
            'TH': 'pink',
            'STR': 'cyan'
        }
        area_colors = []
        area_labels = []
        for region in regions:
            area_colors.append(rgbs[stype2struct[region]])
            area_labels.append(stype2struct[region])
        self.plot_by_types(r2n_dict, regions, figname, show_area=True, 
                           area_colors=area_colors, area_labels=area_labels)
        
        
    def plot_by_sptype(self, figname):
        r2n_dict, regions = select_neurons_by_sptype(self.rdict, self.df_ct)
        self.plot_by_types(r2n_dict, regions, figname)

    def plot_by_sltype(self, figname):
        r2n_dict, regions = select_neurons_by_sltype(self.rdict, self.df_ct)
        self.plot_by_types(r2n_dict, regions, figname)



if __name__ == '__main__':
    swc_dir = '../recheck20230516/data/axon80_sort'
    cellfile = '../../common_lib/41586_2021_3941_MOESM4_ESM.csv'
    features_file = 'hub_features.pkl'
    precomputed = True
 
    if not precomputed:
        rdict = {}
        iswc = 0
        for swcfile in glob.glob(os.path.join(swc_dir, '*.swc')):
            fn = os.path.split(swcfile)[-1][:-9]
            ah = AxonalHubs(swcfile, fn=fn)
            features = ah.detect_hubs()
            rdict[fn] = features
            
            iswc += 1
            if iswc % 10 == 0:
                print(iswc)

        # dump to file
        with open(features_file, 'wb') as fp:
            pickle.dump(rdict, fp)
    else:
        with open(features_file, 'rb') as fp:
            rdict = pickle.load(fp)

    print(len(rdict))
    ph = PlotHubs(rdict, cellfile)
    #ph.plot_by_stype('temp_stype.png')
    #ph.plot_by_sptype('temp_sptype.png')
    #ph.plot_by_sltype('temp_sltype.png')



#!/usr/bin/env python

#================================================================
#   Copyright (C) 2023 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : generate_me_map.py
#   Author       : Yufeng Liu
#   Date         : 2023-02-09
#   Description  : 
#
#================================================================

import numpy as np
import pandas as pd
from skimage import exposure, filters, measure
from skimage import morphology
from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator, CloughTocher2DInterpolator
import matplotlib
import matplotlib.cm as cm
import cv2
import seaborn as sns
import matplotlib.pyplot as plt

from image_utils import get_mip_image, image_histeq
from file_io import load_image, save_image
from anatomy.anatomy_config import MASK_CCF25_FILE
from anatomy.anatomy_vis import get_brain_outline2d, get_section_boundary_with_outline

# features selected by mRMR
__MAP_FEATS__ = ('AverageContraction', 'HausdorffDimension', 'pca_vr3')

def process_features(mefile):
    df = pd.read_csv(mefile, index_col=0)
    df.drop(list(__MAP_FEATS__), axis=1, inplace=True)

    mapper = {}
    for mf in __MAP_FEATS__:
        mapper[f'{mf}_me'] = mf
    df.rename(columns=mapper, inplace=True)

    feat_names = [fn for fn in __MAP_FEATS__]

    return df, feat_names

def process_mip(img, mask2d, axis=0):
    mip = get_mip_image(img, axis).astype(float)
    bg_mask = mip.sum(axis=-1) == 0
    fg_mask = ~bg_mask
    # do interpolation for filling
    for i in range(mip.shape[2]):
        cur_mask = np.where(fg_mask)
        interp = NearestNDInterpolator(np.transpose(cur_mask), mip[:,:,i][cur_mask])
        #interp = LinearNDInterpolator(np.transpose(cur_mask), mip[:,:,i][cur_mask])
        mip[:,:,i] = interp(*np.indices(mip[:,:,i].shape))
    
    #nk = 3
    #nk2 = 2*nk + 1
    #mip = filters.median(mip, morphology.disk(nk).reshape((nk2,nk2,1)))
    
    # zeroing out non-fg regions
    dil_mask = morphology.dilation(fg_mask, morphology.disk(5))

    #mip[bg_mask] = 255
    mip[~dil_mask] = 255
    mip[mask2d & bg_mask] = 128
    mip = mip.astype(np.uint8)
    return mip

def calc_me_maps(mefile, outfile, show_region_boundary=True, histeq=True):
    mask = load_image(MASK_CCF25_FILE)  # z,y,x order!
    df, feat_names = process_features(mefile)

    c = len(feat_names)
    zdim, ydim, xdim = mask.shape
    zdim2, ydim2, xdim2 = zdim//2, ydim//2, xdim//2
    memap = np.zeros((zdim, ydim, xdim, c), dtype=np.uint8)
    xyz = np.round(df[['soma_x', 'soma_y', 'soma_z']].to_numpy()).astype(np.int32)
    # flip z-dimension, so that to aggregate the information to left or right hemisphere
    right_hemi_mask = xyz[:,2] < zdim2
    xyz[:,2][right_hemi_mask] = zdim - xyz[:,2][right_hemi_mask]

    # normalize to uint8
    fvalues = df[feat_names]
    fmin, fmax = fvalues.min(), fvalues.max()
    fvalues = ((fvalues - fmin) / (fmax - fmin) * 255).to_numpy()
    if histeq:
        for i in range(fvalues.shape[1]):
            fvalues[:,i] = image_histeq(fvalues[:,i])[0]
    
    debug = False
    if debug: #visualize the distribution of features
        g = sns.histplot(data=fvalues, kde=True)
        plt.savefig('fvalues_distr_histeq.png', dpi=300)
        plt.close('all')

    memap[xyz[:,2], xyz[:,1], xyz[:,0]] = fvalues
    
    # get the mask of mip brain
    sectionX = None
    mask2ds = []
    for axid in range(3):
        if show_region_boundary:
            mask2d = get_section_boundary_with_outline(mask, axis=axid, v=1, sectionX=sectionX, fuse=True)
        else:
            mask2d = get_brain_outline2d(mask, axis=axid, v=1)
        mask2ds.append(mask2d)

    # keep only values near the section plane
    thickX2 = 40
    prefix = f'{outfile}'
    for axid in range(3):
        print(f'--> Processing axis: {axid}')
        cur_memap = memap.copy()
        print(cur_memap.mean(), cur_memap.std())
        if axid == 0:
            cur_memap[:zdim2-thickX2] = 0
            cur_memap[zdim2+thickX2:] = 0
        elif axid == 1:
            cur_memap[:,:ydim2-thickX2] = 0
            cur_memap[:ydim2+thickX2:] = 0
        else:
            cur_memap[:,:,:xdim2-thickX2] = 0
            cur_memap[:,:,xdim2+thickX2:] = 0
        print(cur_memap.mean(), cur_memap.std())
        
        mip = process_mip(cur_memap, mask2ds[axid], axid)
        cv2.imwrite(f'{prefix}_mip{axid}.png', mip[:,:,::-1])
    


if __name__ == '__main__':
    mefile = './data/micro_env_features_nodes300-1500_withoutNorm.csv'
    mapfile = 'microenviron_map'

    calc_me_maps(mefile, outfile=mapfile)


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
import matplotlib
import matplotlib.cm as cm
import cv2

from image_utils import get_mip_image
from file_io import load_image, save_image
from anatomy.anatomy_config import MASK_CCF25_FILE

__MAP_FEATS__ = ('Length', 'Branches', 'AverageContraction', 
                 'AverageBifurcationAngleLocal', 'pc11', 'pc12', 'pc13',
                 'pca_vr1', 'pca_vr2', 'pca_vr3')

def process_features(mefile):
    df = pd.read_csv(mefile, index_col=0)
    df.drop(list(__MAP_FEATS__), axis=1, inplace=True)

    mapper = {}
    for mf in __MAP_FEATS__:
        mapper[f'{mf}_me'] = mf
    df.rename(columns=mapper, inplace=True)

    feat_names = [fn for fn in __MAP_FEATS__]
    df['Polarity'] = df['pca_vr1'].to_numpy() - df['pca_vr2'].to_numpy()
    feat_names.append('Polarity')

    remove_keys = ['pca_vr1', 'pca_vr2', 'pca_vr3', 'pc11', 'pc12', 'pc13']
    df.drop(remove_keys, axis=1, inplace=True)
    for rkey in remove_keys:
        feat_names.remove(rkey)

    return df, feat_names

def process_mip(img, mask2d, axis=0):
    mip = get_mip_image(img, axis)
    bg_mask = mip.sum(axis=-1) == 0
    mip[bg_mask] = 255
    mip[mask2d] = 0
    return mip

def detect_edges(mask2d):
    mask2d = mask2d.astype(np.float)
    gx, gy = np.gradient(mask2d)
    edges = (gy * gy + gx * gx) != 0
    return edges

def calc_me_maps(mefile, outfile):
    mask = load_image(MASK_CCF25_FILE)  # z,y,x order!
    df, feat_names = process_features(mefile)

    # get the mask of mip brain
    mask_bin = mask > 0
    mask2d0 = detect_edges(get_mip_image(mask_bin, 0))
    mask2d1 = detect_edges(get_mip_image(mask_bin, 1))
    mask2d2 = detect_edges(get_mip_image(mask_bin, 2))

    c = len(feat_names)
    memap = np.zeros((c, *mask.shape), dtype=np.float)
    xyz = np.round(df[['soma_x', 'soma_y', 'soma_z']].to_numpy()).astype(np.int32)
    # normalize to uint8
    fvalues = df[feat_names]
    fmin, fmax = fvalues.min(), fvalues.max()
    fvalues = ((fvalues - fmin) / (fmax - fmin)).to_numpy()
    memap[:, xyz[:,2], xyz[:,1], xyz[:,0]] = fvalues.transpose()
    fg_mask = memap > 0
    
    # write out for visualization
    #save_image(outfile, memap)
    norm = matplotlib.colors.Normalize(0, 1, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.seismic_r)
    for i, fn in enumerate(feat_names):
        print(f'==> Saving {fn} map: {memap[i].shape}')
        sx, sy, sz = memap[i].shape
        cur_img = np.zeros((sx, sy, sz, 3), dtype=np.uint8)
        #tmp = filters.gaussian(memap[i], 5)
        eq_hist = exposure.equalize_hist(memap[i][fg_mask[i]])
        rgb = (mapper.to_rgba(eq_hist)[:,:3] * 255).astype(np.uint8)
        cur_img[fg_mask[i]] = rgb
        
        prefix = f'{outfile}_{fn}'
        #save_image(f'{prefix}.tiff', cur_img)
        mip0 = process_mip(cur_img, mask2d0, 0)
        mip1 = process_mip(cur_img, mask2d1, 1)
        mip2 = process_mip(cur_img, mask2d2, 2)
        cv2.imwrite(f'{prefix}_mip0.png', mip0[:,:,::-1])
        cv2.imwrite(f'{prefix}_mip1.png', mip1[:,:,::-1])
        cv2.imwrite(f'{prefix}_mip2.png', mip2[:,:,::-1])
        #break
    
    
if __name__ == '__main__':
    mefile = './data/micro_env_features_nodes300-1500_withoutNorm.csv'
    mapfile = 'microenviron_map'

    calc_me_maps(mefile, outfile=mapfile)


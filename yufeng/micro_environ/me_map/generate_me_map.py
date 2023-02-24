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
from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator
import matplotlib
import matplotlib.cm as cm
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import cv2
import seaborn as sns
import matplotlib.pyplot as plt

from image_utils import get_mip_image, image_histeq
from file_io import load_image, save_image
from anatomy.anatomy_config import MASK_CCF25_FILE
from anatomy.anatomy_vis import get_brain_outline2d, get_section_boundary_with_outline, get_brain_mask2d, get_section_boundary

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

def process_mip(img, boundary_mask2d, outline_mask2d, brain_mask2d, axis=0, figname='temp.png', mode='composite'):
    mip = get_mip_image(img, axis)
    #if axis==1: cv2.imwrite('temp.png', mip); sys.exit()
    # redraw the image through different point style
    im = np.ones((mip.shape[0], mip.shape[1], 4), dtype=np.uint8) * 255
    im[~brain_mask2d] = 1
    
    fig, ax = plt.subplots()
    width, height = fig.get_size_inches() * fig.get_dpi()
    width = int(width)
    height = int(height)

    canvas = FigureCanvas(fig)
    im = ax.imshow(im)
    fig.patch.set_visible(False)
    ax.axis('off')
    
    bg_mask = mip.sum(axis=-1) == 0
    fg_mask = ~bg_mask
    fg_indices = np.where(fg_mask)
    if mode == 'composite':
        fg_values = mip[fg_indices] / 255.
        cmap = None
    else:
        fg_values = mip[fg_indices][:,0] / 255.
        cmap = 'coolwarm'
    
    ax.scatter(fg_indices[1], fg_indices[0], c=fg_values, s=5, edgecolors='none', cmap=cmap)
    # show boundary
    b_indices = np.where(boundary_mask2d)
    ax.scatter(b_indices[1], b_indices[0], s=0.5, c='black', alpha=0.5, edgecolors='none')
    o_indices = np.where(outline_mask2d)
    ax.scatter(o_indices[1], o_indices[0], s=1.0, c='orange', alpha=1.0, edgecolors='none')

    plt.savefig(figname, dpi=300)
    plt.close('all')

    #canvas.draw()       # draw the canvas, cache the renderer
    #img_buffer = canvas.tostring_rgb()
    #out = np.frombuffer(img_buffer, dtype=np.uint8).reshape(height, width, 3)
    #return out

def calc_me_maps(mefile, outfile, histeq=True, flip_to_left=True, mode='composite', findex=0):
    '''
    @param mefile:          file containing microenviron features
    @param outfile:         prefix of output file
    @param histeq:          Whether or not to use histeq to equalize the feature values
    @param flip_to_left:    whether map points at the right hemisphere to left hemisphere
    @param mode:            [composite]: show 3 features; otherwise separate feature
    @param findex:          index of feature to display
    '''

    if mode != 'composite':
        fname = __MAP_FEATS__[findex]
        prefix = f'{outfile}_{fname}'
    else:
        prefix = f'{outfile}'

    mask = load_image(MASK_CCF25_FILE)  # z,y,x order!
    df, feat_names = process_features(mefile)

    c = len(feat_names)
    zdim, ydim, xdim = mask.shape
    zdim2, ydim2, xdim2 = zdim//2, ydim//2, xdim//2
    memap = np.zeros((zdim, ydim, xdim, c), dtype=np.uint8)
    xyz = np.round(df[['soma_x', 'soma_y', 'soma_z']].to_numpy()).astype(np.int32)
    if flip_to_left:
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

    if mode == 'composite':
        memap[xyz[:,2], xyz[:,1], xyz[:,0]] = fvalues
    else:
        memap[xyz[:,2], xyz[:,1], xyz[:,0]] = fvalues[:,findex].reshape(-1,1)
    
    # get the mask of mip brain
    sectionX = None
    boundary_mask2ds = []
    outline_mask2ds = []
    brain_mask2ds = []
    for axid in range(3):
        boundary_mask2ds.append(get_section_boundary(mask, axis=axid, v=1, c=sectionX))
        outline_mask2ds.append(get_brain_outline2d(mask, axis=axid, v=1))
        brain_mask2ds.append(get_brain_mask2d(mask, axis=axid, v=1))

    # keep only values near the section plane
    thickX2 = 40
    for axid in range(3):
        print(f'--> Processing axis: {axid}')
        cur_memap = memap.copy()
        print(cur_memap.mean(), cur_memap.std())
        if thickX2 != -1:
            if axid == 0:
                cur_memap[:zdim2-thickX2] = 0
                cur_memap[zdim2+thickX2:] = 0
            elif axid == 1:
                cur_memap[:,:ydim2-thickX2] = 0
                cur_memap[:,ydim2+thickX2:] = 0
            else:
                cur_memap[:,:,:xdim2-thickX2] = 0
                cur_memap[:,:,xdim2+thickX2:] = 0
        print(cur_memap.mean(), cur_memap.std())
        
        figname = f'{prefix}_mip{axid}.png'
        mip = process_mip(cur_memap, boundary_mask2ds[axid], outline_mask2ds[axid], brain_mask2ds[axid], axid, figname, mode=mode)
        # load and remove the zero-alpha block
        img = cv2.imread(figname, cv2.IMREAD_UNCHANGED)
        wnz = np.nonzero(img[img.shape[0]//2,:,-1])[0]
        ws, we = wnz[0], wnz[-1]
        hnz = np.nonzero(img[:,img.shape[1]//2,-1])[0]
        hs, he = hnz[0], hnz[-1]
        img = img[hs:he+1, ws:we+1]
        if axid == 2:   # rotate 90
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        # set the alpha of non-brain region as 0
        img[img[:,:,-1] == 1] = 0
        cv2.imwrite(figname, img)
        


if __name__ == '__main__':
    mefile = './data/micro_env_features_nodes300-1500_withoutNorm.csv'
    mapfile = 'microenviron_map'
    flip_to_left = True
    mode = 'feature'
    findex = 0

    calc_me_maps(mefile, outfile=mapfile, flip_to_left=flip_to_left, mode=mode, findex=findex)


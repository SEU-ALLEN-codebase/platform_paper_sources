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
from scipy.optimize import curve_fit
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
from anatomy.anatomy_core import parse_ana_tree

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

def plot_section_outline(mask, axis=0, sectionX=None, ax=None, with_outline=True, outline_color='orange'):
    boundary_mask2d = get_section_boundary(mask, axis=axis, v=1, c=sectionX)
    sh, sw = boundary_mask2d.shape[:2]
    if ax is None:
        fig, ax = plt.subplots()
        brain_mask2d = get_brain_mask2d(mask, axis=axis, v=1)
        im = np.ones((sh, sw, 4), dtype=np.uint8) * 255
        im[~brain_mask2d] = 1

    # show boundary
    b_indices = np.where(boundary_mask2d)
    ax.scatter(b_indices[1], b_indices[0], s=0.5, c='black', alpha=0.5, edgecolors='none')
    # intra-brain regions
        
    if with_outline:
        outline_mask2d = get_brain_outline2d(mask, axis=axis, v=1)
        o_indices = np.where(outline_mask2d)
        ax.scatter(o_indices[1], o_indices[0], s=1.0, c=outline_color, alpha=1.0, edgecolors='none')
    
    if ax is None:
        return fig, ax
    else:
        return ax
    

def process_mip(mip, mask, sectionX=None, axis=0, figname='temp.png', mode='composite'):
    # get the mask
    brain_mask2d = get_brain_mask2d(mask, axis=axis, v=1)

    #if axis==1: cv2.imwrite('temp.png', mip); sys.exit()
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
    plot_section_outline(mask, axis=axis, sectionX=sectionX, ax=ax, with_outline=True, outline_color='orange')

    plt.savefig(figname, dpi=300)
    plt.close('all')

    #canvas.draw()       # draw the canvas, cache the renderer
    #img_buffer = canvas.tostring_rgb()
    #out = np.frombuffer(img_buffer, dtype=np.uint8).reshape(height, width, 3)
    #return out

def get_me_mips(mefile, shape3d, histeq, flip_to_left, mode, findex):
    df, feat_names = process_features(mefile)
    
    c = len(feat_names)
    zdim, ydim, xdim = shape3d
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
    
    # keep only values near the section plane
    thickX2 = 40
    mips = []
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
        
        mip = get_mip_image(cur_memap, axid)
        mips.append(mip)
    return mips

def generate_me_maps(mefile, outfile, histeq=True, flip_to_left=True, mode='composite', findex=0, fmt='svg'):
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
    shape3d = mask.shape
    mips = get_me_mips(mefile, shape3d, histeq, flip_to_left, mode, findex)
    for axid, mip in enumerate(mips):
        figname = f'{prefix}_mip{axid}.{fmt}'
        process_mip(mip, mask, axis=axid, figname=figname, mode=mode)
        if not figname.endswith('svg'):
            # load and remove the zero-alpha block
            img = cv2.imread(figname, cv2.IMREAD_UNCHANGED)
            wnz = np.nonzero(img[img.shape[0]//2,:,-1])[0]
            ws, we = wnz[0], wnz[-1]
            hnz = np.nonzero(img[:,img.shape[1]//2,-1])[0]
            hs, he = hnz[0], hnz[-1]
            img = img[hs:he+1, ws:we+1]
            if axid != 0:   # rotate 90
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            # set the alpha of non-brain region as 0
            img[img[:,:,-1] == 1] = 0
            cv2.imwrite(figname, img)
        
def plot_left_right_corr(mefile, outfile, histeq=True, mode='composite', findex=0):
    if mode != 'composite':
        fname = __MAP_FEATS__[findex]
        prefix = f'{outfile}_{fname}'
    else:
        prefix = f'{outfile}'

    def customized_func(x, a):
        return a*x

    def customized_line(xkey, ykey, color=None, label=None, **kwargs):
        # we should use `y = ax` to fit our data.
        ax = plt.gca()
        popt, pcov = curve_fit(customized_func, xkey, ykey)
        xmin = min(xkey.min(), ykey.min())
        xmax = max(xkey.max(), ykey.max())
        xdata = np.arange(xmin, xmax)
        ydata = popt[0] * xdata
        ax.plot(xdata, ydata, linewidth=3, color=color, label=label)
        # annotate
        ax.text(.03, .85, f'y={popt[0]:.2f}x\npcov={pcov[0][0]:.2g}', 
                fontsize=20, transform=ax.transAxes, color=color)

    
    mask = load_image(MASK_CCF25_FILE)  # z,y,x order!
    shape3d = mask.shape
    mips = get_me_mips(mefile, shape3d, False, False, mode, findex)
    xlabel = 'Value predicted \nfrom left-hemisphere'
    ylabel = 'Right-hemispheric value'
    data = []
    for axid, mip in enumerate(mips):
        if axid == 0: continue # no left/right difference
        s1, s2 = mip.shape[:2]
        sh = s1 // 2
        print(s1, s2, sh)
        fg_mask_left = mip.sum(axis=-1) != 0; fg_mask_left[:sh] = 0
        fg_mask_right = mip.sum(axis=-1) != 0; fg_mask_right[sh:] = 0
        vpreds = []
        vtrues = []
        lindices = np.where(fg_mask_left)
        rindices = np.where(fg_mask_right)
        rindices_m = (s1 - rindices[0], rindices[1])
        for i in range(mip.shape[2]):
            lvalues = mip[:,:,i][lindices]
            interp = LinearNDInterpolator(np.transpose(lindices), lvalues)
            vpreds.append(interp(*rindices_m))
            vtrues.append(mip[:,:,i][rindices])
        values = np.vstack((np.hstack(vpreds), np.hstack(vtrues)))
        fnames = []
        for mf in __MAP_FEATS__:
            for j in range(rindices[0].shape[0]):
                fnames.append(mf)
        cur_data = np.array([*values, fnames]).transpose()
        data.append(cur_data)

    size1, size2 = data[0].shape[0], data[1].shape[0]
    views = ['Axial' for i in range(size1)] + ['Coronal' for i in range(size2)]
    data = np.vstack(data)
    figname = f'{prefix}_left_right.png'
    df = pd.DataFrame(data, columns=[xlabel, ylabel, 'feature']).astype({xlabel: float, ylabel: float})
    df['view'] = views
    # remove nan values
    df = df[~df[xlabel].isna()]
    print(df.shape)

    sns.set_theme(style='white', font_scale=1.8)
    g = sns.lmplot(data=df, x=xlabel, y=ylabel, col='feature', 
                   row='view', fit_reg=False,
                   scatter_kws={'s':5}, 
                   facet_kws={'despine': False, 'margin_titles': True})
    g.map(customized_line, xlabel, ylabel, color='magenta')
    g.set(xlim=(0,255), ylim=(0,255), aspect='equal')
    for i, ax_list in enumerate(g.axes):
        for j, ax in enumerate(ax_list):
            ax.spines['left'].set_linewidth(2)
            ax.spines['right'].set_linewidth(2)
            ax.spines['top'].set_linewidth(2)
            ax.spines['bottom'].set_linewidth(2)
            ax.xaxis.set_tick_params(width=2, direction='in')
            ax.yaxis.set_tick_params(width=2, direction='in')
            if i == 0:
                ax.set_title(ax.get_title().split(' = ')[1])

    g.figure.subplots_adjust(wspace=-0.1, hspace=0)
    plt.savefig(figname, dpi=300)
    plt.close('all')
        
def colorize_atlas2d_cv2(outscale=3, annot=False, fmt='svg'):
    mask = load_image(MASK_CCF25_FILE)
    ana_dict = parse_ana_tree()
    for axid in range(3):
        boundaries = get_section_boundary(mask, axis=axid, c=None, v=1)
        c = mask.shape[axid] // 2
        section = np.take(mask, c, axid)
        out = np.ones((*section.shape, 3), dtype=np.uint8) * 255
        values = np.unique(section)
        print(f'Dimension of axis={axid} is: {section.shape}, with {len(values)-1} regions')

        if annot:
            centers = []
            rnames = []
            c2 = out.shape[0] // 2
            right_mask = boundaries.copy()
            right_mask.fill(False)
            right_mask[:c2] = True
            for v in values:
                if v == 0: continue
                rname = ana_dict[v]['acronym']
                
                # center of regions, 
                cur_mask = section == v
                out[:,:,:3][cur_mask] = ana_dict[v]['rgb_triplet']

                if rname in ['root', 'fiber tracts']:   # no texting is necessary
                    continue
                if axid != 0:   
                    cur_mask = cur_mask & right_mask #only left hemisphere
                cc = cv2.connectedComponents(cur_mask.astype(np.uint8))
                for i in range(cc[0] - 1):
                    cur_mask = cc[1] == (i+1)
                    if cur_mask.sum() < 50:
                        continue
                    indices = np.where(cur_mask)
                    xmean = (indices[0].min() + indices[0].max()) // 2
                    ymean = int(np.median(indices[1][indices[0] == xmean]))
                    centers.append((xmean, ymean))
                    rnames.append(rname)
        else:
            for v in values:
                if v == 0: continue
                cur_mask = section == v
                out[:,:,:3][cur_mask] = ana_dict[v]['rgb_triplet']
        # mixing with boudnary
        alpha = 0.5
        out[:,:,:3][boundaries] = (255 * alpha + out[boundaries][:,:3] * (1 - alpha)).astype(np.uint8)
        #out[:,:,3][boundaries] = int(alpha * 255)
        
        figname = f'atlas_axis{axid}.{fmt}'
        if outscale != 1:
            out = cv2.resize(out, (0,0), fx=outscale, fy=outscale, interpolation=cv2.INTER_CUBIC)
        # we would like to rotate the image, so that it can be better visualized
        if axid != 0:
            out = cv2.rotate(out, cv2.ROTATE_90_CLOCKWISE)

        so1, so2 = out.shape[:2]
        # annotation if required
        if annot:
            figname = f'atlas_axis{axid}_annot.{fmt}'
            for center, rn in zip(centers, rnames):
                sx, sy = center[1]*outscale, center[0]*outscale
                if axid != 0:
                    # rotate accordingly
                    new_center = (so2-sy, sx)
                else:
                    new_center = (sx, sy)
                cv2.putText(out, rn, new_center, cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,0,0), 1)

        if figname.endswith('svg'):
            # save to `svg` vectorized file, using plt
            fig, ax = plt.subplots()
            ax.imshow(out)
            fig.patch.set_visible(False)
            ax.axis('off')
            plt.savefig(figname, dpi=300)
            plt.close('all')
        else:
            cv2.imwrite(figname, out)


if __name__ == '__main__':
    mefile = './data/micro_env_features_nodes300-1500_withoutNorm.csv'
    mapfile = 'microenviron_map'
    flip_to_left = True
    mode = 'composite'
    findex = 0
    fmt = 'svg'

    #generate_me_maps(mefile, outfile=mapfile, flip_to_left=flip_to_left, mode=mode, findex=findex)
    #plot_left_right_corr(mefile, outfile=mapfile, histeq=True, mode='composite', findex=0)
    colorize_atlas2d_cv2(annot=False, fmt=fmt)


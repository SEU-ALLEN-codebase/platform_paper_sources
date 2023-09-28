import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import transforms

from file_io import load_image
from anatomy.anatomy_config import MASK_CCF25_FILE
from anatomy.anatomy_core import parse_ana_tree
from anatomy.anatomy_vis import get_brain_mask2d

def initialize_bkg_image(mask, axis=0, vbkg=196):
    mask2d = get_brain_mask2d(mask, axis=axis, v=1)
    bkg = np.ones((*mask2d.shape, 4), dtype=np.uint8) * 255
    bkg[:,:,-1] = 0
    bkg[mask2d] = vbkg
    return bkg

class Topography(object):
    def __init__(self, pre_dumped_tracts='../sd_matrix/main_tract_ptype.pkl', scale=25., vbkg=[204,204,255,255]):
        # load the dumped main tract data
        with open(pre_dumped_tracts, 'rb') as fp:
            self.tracts = pickle.load(fp)
            print(self.tracts.keys())
            print('\n')
        #import ipdb; ipdb.set_trace()
        
        self.mask = load_image(MASK_CCF25_FILE)
        self.bkgz = initialize_bkg_image(self.mask, axis=0, vbkg=vbkg)
        self.bkgy = initialize_bkg_image(self.mask, axis=1, vbkg=vbkg)
        self.bkgx = initialize_bkg_image(self.mask, axis=2, vbkg=vbkg)

        self.ana_dict = parse_ana_tree()
        self.scale = scale

    def get_proj_terminus(self, coords, return_soma=False):
        i = len(coords)
        sz, sy, sx = self.mask.shape
        while i:
            pos_t = coords[len(coords) - i]
            xyz = pos_t / self.scale
            z, y, x = np.round(xyz[::-1]).astype(int)
            if z >= sz or y >= sy or x >= sx or self.mask[z,y,x] == 0:
                i -= 1
            else:
                if return_soma:
                    return (z,y,x), np.round(coords[-1][::-1] / self.scale).astype(int)
                else:
                    return (z,y,x)
                
    def get_proj_regions(self, region):
        rnames = []
        for name, coords in self.tracts[region]:
            z,y,x = self.get_proj_terminus(coords)
            idx = self.mask[z, y, x]
            # get the region name
            rname = self.ana_dict[idx]['acronym']
            rnames.append(rname)

        return rnames

    def points_on_ccf(self, region):
        ts, ss = [], []
        for name, coords in self.tracts[region]:
            t,s = self.get_proj_terminus(coords, return_soma=True)
            ts.append(t)
            ss.append(s)
        ts = np.array(ts)
        ss = np.array(ss)
        # plot
        fig = plt.figure(figsize=(4.4,2))
        for i in range(1,3):

            if i == 0:
                bkg = self.bkgz
                ts2d, ss2d = ts[:,[1,2]], ss[:,[1,2]]
            elif i == 1:
                bkg = self.bkgy
                ts2d, ss2d = ts[:,[0,2]], ss[:,[0,2]]
                ax = plt.subplot(121)
            elif i == 2:
                bkg = self.bkgx
                ts2d, ss2d = ts[:,[0,1]], ss[:,[0,1]]
                ax = plt.subplot(122)
             
            #canvas = FigureCanvas(fig)
            img = ax.imshow(bkg)
            plt.scatter(ss2d[:,1], ss2d[:,0], marker='o', s=30, c='r')
            plt.scatter(ts2d[:,1], ts2d[:,0], marker='^', s=30, c='k')
            plt.tick_params(left = False, right = False, labelleft = False ,
                    labelbottom = False, bottom = False)
            # Iterating over all the axes in the figure
            # and make the Spines Visibility as False
            ax = plt.gca()
            for pos in ['right', 'top', 'bottom', 'left']:
                ax.spines[pos].set_visible(False)
        #plt.tight_layout()
        plt.subplots_adjust(left=0.0, bottom=0, right=1.0, top=1, wspace=-0.1)
        tr = transforms.Affine2D().rotate_deg(90)
        plt.savefig(f'{region}.png', dpi=450)
        plt.close()


            

if __name__ == '__main__':
    tg = Topography()
    for ptype in ['CTX_ET', 'CTX_IT']:
        for stype in ['MOp', 'MOs', 'SSp-bfd', 'SSp-m', 'SSp-n', 'SSs', 'SSp-ul', 'RSPv']:
            region = f'{ptype}-{stype}'
            #rnames = tg.get_proj_regions(region)
            #print(region, rnames)
            print(region)
            tg.points_on_ccf(region)
            #break



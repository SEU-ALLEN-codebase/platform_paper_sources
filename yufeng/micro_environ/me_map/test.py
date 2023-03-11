#!/usr/bin/env python

#================================================================
#   Copyright (C) 2023 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : test.py
#   Author       : Yufeng Liu
#   Date         : 2023-02-24
#   Description  : 
#
#================================================================
import numpy as np
import matplotlib.pyplot as plt

'''
# Generate pseudo-colorbar for feature-me-map
data = np.random.random((1000))
g = plt.scatter(x=np.arange(0,1,0.001), y=np.arange(0,1,0.001), c=data, cmap='coolwarm')
cbar = plt.colorbar(aspect=5, ticks=[], orientation='horizontal')
plt.savefig('todel.png', dpi=600)
plt.close('all')
'''

palette = {
    0: (102,255,102),
    1: (255,102,102),
    2: (255,255,102),
    3: (102,255,255),
    4: (102,102,255),
    5: (178,102,255)
}

n = 5
d0 = 5 * np.random.randn(n)
d1 = d0 + np.random.random(n) * 0.1 + 1
d2 = d0 + np.random.random(n) * 0.1 + 2
d3 = d0 + np.random.random(n) * 0.1 + 3
d4 = d0 + np.random.random(n) * 0.1 - 1
d5 = d0 + np.random.random(n) * 0.1 - 2

xp = np.arange(n)
marker_style = {
    'markeredgecolor': 'black'
}
plt.scatter(xp, d0 - 0.2, c='w', s=60, marker='s', label=f'Insufficient data', edgecolor='black')
for i, di in enumerate([d0,d1,d2,d3,d4,d5]):
    c = np.array(palette[i]).reshape((1,-1)) / 255
    print(c)
    plt.scatter(xp, di, c=c, s=60, marker='s', label=f'cluster{i}', edgecolor='black')
plt.legend(ncol=1, frameon=False, handletextpad=0.1, labelspacing=0.3)
plt.axis('off')
plt.savefig('pseudo_colorbar_classes.png', dpi=600)


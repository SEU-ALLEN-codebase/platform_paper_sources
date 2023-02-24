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

data = np.random.random((1000))
g = plt.scatter(x=np.arange(0,1,0.001), y=np.arange(0,1,0.001), c=data, cmap='coolwarm')
cbar = plt.colorbar(aspect=5, ticks=[], orientation='horizontal')
plt.savefig('todel.png', dpi=600)
plt.close('all')


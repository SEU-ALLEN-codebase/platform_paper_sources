#!/usr/bin/env python
# coding: utf-8

# In[30]:


import pandas as pd
import numpy as np
df_feature = pd.read_csv('../me_map_new20230510/data/lm_features_d22_15441.csv')
dimension = df_feature['HausdorffDimension'].values
df_feature = df_feature.iloc[np.nonzero(dimension==dimension)[0]]
# dimension = df_feature['AverageBifurcationAngleRemote'].values
# df_feature = df_feature.iloc[np.nonzero(dimension==dimension)[0]]
# dimension = df_feature['AverageBifurcationAngleLocal'].values
# df_feature = df_feature.iloc[np.nonzero(dimension==dimension)[0]]
dimension = df_feature['Nodes'].values
print(np.unique(dimension))
df_feature = df_feature.iloc[np.nonzero(dimension>=30)[0]]
dimension = df_feature['Nodes'].values
print(np.unique(dimension))
df_feature = df_feature.iloc[np.nonzero(dimension<=1500)[0]]
# remove regions with less than 10 neurons
regions, counts = np.unique(df_feature['region_id_r316'], return_counts=True)
keep_regions = regions[counts >= 10]
df_feature = df_feature[df_feature['region_id_r316'].isin(keep_regions)]
print(f'Number ater regional counts: {df_feature.shape[0]}')




from utils import get_level_id_list

region_316_list = df_feature['region_id_r316']
region_316_l13_list = get_level_id_list(region_316_list,4)
instruc = np.nonzero(np.array(region_316_l13_list)!=0)[0]
print(instruc)

df_plot_316 = df_feature.drop(columns=['soma_x','soma_y','soma_z','Unnamed: 0','dataset_name','brain_id','region_id_r671','region_name_r671','region_name_r316','brain_structure'])
df_plot_316 = df_plot_316.iloc[instruc]
df_plot_316 = df_plot_316.groupby('region_id_r316').agg('median')

std =  df_plot_316.std(axis=0)
df_plot_316 = pd.DataFrame(df_plot_316.values-np.array([df_plot_316.mean(axis=0)]).repeat(df_plot_316.shape[0],axis=0),
                      index=df_plot_316.index,columns=df_plot_316.columns)
df_plot_316 = df_plot_316.div(std,axis=1)
df_plot_316


# In[34]:


np.nonzero(df_plot_316.values!=df_plot_316.values)[0]


# In[35]:


from utils import get_level_id_list,get_region_name_list
import seaborn as sns
from matplotlib import pyplot as plt

df_plot = df_plot_316.copy()

# for noca in ['MB','MY','CBN','CTXsp','OLF','TH','PAL','HPF','HY']:
#     region_index_list = df_plot.index.tolist()
#     structure_index_list = get_level_id_list(region_index_list,13)
#     structure_list = get_region_name_list(structure_index_list,32)
#     df_plot = df_plot.iloc[np.nonzero(np.array(structure_list)!=noca)[0]]
region_index_list = df_plot.index.tolist()
structure_index_list = get_level_id_list(region_index_list,4)
structure_list = get_region_name_list(structure_index_list,32)
    
#structure_color_dict = {'CTX':'green','CNU':'blue','BS':'red','CB':'orange',}
structure_color_dict = {'CTX':'green','CNU':'blue','BS':'red',}
region_color_list = [structure_color_dict[s] for s in structure_list]
print(region_color_list) 

region_list = get_region_name_list(region_index_list,32)
df_plot.index = region_list  
df_plot



import matplotlib

font = {
    'weight': 'bold'
}
matplotlib.rc('font', **font)

# customize the padding between ticks and ticklabels
tickpadsize = 0.5
tickthickness = 0.3
ticklength = 2



#cols = df_plot.columns[[6,7,8,16,17,18,19,20,21]]
cols = df_plot.columns[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,]]
print(cols)
df = pd.DataFrame(df_plot[cols].values.T,columns=df_plot.index,index=cols)
df.to_csv("todel.csv")
   
g = sns.clustermap(df,
               cmap='BrBG',
               cbar_pos=(0.158,0.24,0.008,0.2),#bwr#coolwarm#Spectral#RdBu
               #cbar_kws={'frameon':False,'figsize':(2,2)},
               col_cluster=True,
               col_colors=region_color_list,
               row_cluster=True,
               figsize=(15,4),
#                xticklabels=1,
#                yticklabels=1,
               #square=True,
              )
g.ax_row_dendrogram.set_visible(False)
g.ax_cbar.set_yticks()
g.ax_cbar.set_yticklabels(g.ax_cbar.get_yticklabels(),fontsize=6)
g.ax_cbar.yaxis.set_tick_params(pad=tickpadsize, width=tickthickness, length=ticklength)

# g.ax_heatmap.tick_params = {"labelsize":20}
# g.ax_heatmap.update_params()

xlabels = df.columns.tolist()
g.ax_heatmap.set_xticks([i+0.5 for i in list(range(len(xlabels)))])
g.ax_heatmap.set_xticklabels([xlabels[i] for i in g.dendrogram_col.reordered_ind],fontsize=7)
g.ax_heatmap.xaxis.set_tick_params(pad=tickpadsize, width=tickthickness, length=ticklength)
for tick in g.ax_heatmap.get_xticklabels():        
    tick.set_rotation(90)
    

ylabels = [
        'Nodes', 
           'SomaSurface', 
           'Stems', 
           #
           'Bifurcations', 
           'Branches',
           'Tips',
      #
           'OverallWidth', 
           'OverallHeight', 
           'OverallDepth',
           'AvgDiameter',
       #
           'Length',
           'Surface',
           'Volume', 
       #
         'MaxEucDist',
       'MaxPathDist', 
       'MaxBranchOrder', 
           #
        'AvgContraction',
       'AvgFragmentation', 
        'AvgPDRatio',
           #
       #'Average\nBifurcation\nAngleLocal', 
       # 'Average\nBifurcation\nAngleRemote',
       #'Hausdorff\nDimension',
          ]
# Note that the ylabels are shuffled if row_cluster is turned on
g.ax_heatmap.set_yticks([i+0.5 for i in list(range(len(ylabels)))])
print(g.dendrogram_row.reordered_ind)
g.ax_heatmap.set_yticklabels([ylabels[i] for i in g.dendrogram_row.reordered_ind],fontsize=8)

g.ax_heatmap.yaxis.set_tick_params(pad=tickpadsize, width=tickthickness, length=ticklength)
for tick in g.ax_heatmap.get_yticklabels():        
    tick.set_verticalalignment('center')
    tick.set_horizontalalignment('left')
    


import matplotlib.patches as mpatches
legend_TN = [mpatches.Patch(color=c, label=l) for l,c in structure_color_dict.items()]
plt.legend(handles=legend_TN,
           handletextpad=0.5,#padding of color
           labelspacing=0,
           handleheight=2,#height of color box
           prop={'size':6},
           bbox_to_anchor=(2.6, 2.8),#legend box center
           frameon=False,#whether to plot frame
          )
#plt.subplots_adjust(bottom=0.2)

plt.savefig(f'clustermap_d15k.png',dpi=600)



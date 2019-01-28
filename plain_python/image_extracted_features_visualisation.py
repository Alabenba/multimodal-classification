
# coding: utf-8

# In[1]:


import os
import re
import numpy as np
import pandas as pd

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import plotly
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.plotly as py

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)


# In[2]:


custom_color_map=[
    "#9370DB", "#EE82EE", "#6495ED", "#DB7093", "#1E90FF", "#40E0D0", "#008B8B",
    "#2E8B57", "#00FF00", "#6B8E23", "#F0E68C", "#DAA520", "#FF8C00", "#FFA07A",
    "#FF7F50", "#DC143C", "#800000", "#D2691E", "#BC8F8F", "#5F9EA0", "#9ACD32",
    "#FFA500", "#FFD700", "#B22222", "#4B0082", "FF0000"
]
default_color="#FF0000"


# In[3]:


cwd=os.getcwd()
image_dir = os.path.join(cwd,'data/recipes_splitted/images')

df_extracted_features = pd.read_pickle(os.path.join(image_dir,"extracted_features.pkl"))
#print("Extraced features from images: \n", df_extracted_features)


# In[4]:


features=df_extracted_features['features'].values.tolist()
labels=df_extracted_features['labels'].values.tolist()
image_names=df_extracted_features['image_names'].values.tolist()

unique_labels=list(set(labels))
number_of_colors=len(unique_labels)
possible_color_indicies=np.linspace(0,number_of_colors-1,number_of_colors,dtype=int)
labels2color_indices=dict(zip(unique_labels,possible_color_indicies))

print(labels2color_indices)

color_indicies=[]
for l in labels:
    color_index=labels2color_indices[l]
    color_indicies.append(color_index)


# In[5]:


#PCA

pca = PCA(n_components=2)
pca_2D_embeddings = pca.fit_transform(features)

dict_PCA_2D = {
    'pca-1' : pca_2D_embeddings[:,0],
    'pca-2' : pca_2D_embeddings[:,1],
    'labels' : labels,
    'color_indicies': color_indicies,
    'image_names': image_names
}

df_PCA_2D=pd.DataFrame(dict_PCA_2D)
#print(df_2D_PCA)


# In[6]:


traces=[]

for n in range(0,number_of_colors):
    df_category=df_PCA_2D[(df_PCA_2D.color_indicies==n)]
    point_name=str(df_category['labels'][df_category.index[0]])
    trace = go.Scatter(
    x=df_category['pca-1'],
    y=df_category['pca-2'],
    mode='markers',
    marker=dict(
        colorscale='Viridis',
        opacity=0.7,
        line=dict(color='rgb(140, 140, 170)')
    ),
    name=point_name,
    text=df_category['image_names'],
    )

    traces.append(trace)

layout = go.Layout(
    title='PCA reduction: 2D features',
    hovermode='closest',
    margin=dict(
       l=30,
       r=30,
       b=30,
       t=30
    ),
      scene = dict(
        xaxis = dict(
            title='pca-1',
       ),
       yaxis = dict(
           title='pca-2',
       ),
    ),
 
)

fig = go.Figure(data=traces, layout=layout)
iplot(fig)


# In[9]:


#t-SNE 2D

tsne2D = TSNE(perplexity=50, n_components=2, verbose=2, init='pca', n_iter=2000,
              n_iter_without_progress=500, learning_rate=20,method='exact')
tsne_2D_embeddings = tsne2D.fit_transform(features)

dict_TSNE_2D = {
'pca-1' : tsne_2D_embeddings[:,0],
'pca-2' : tsne_2D_embeddings[:,1],
'labels' : labels,
'color_indicies': color_indicies,
'image_names': image_names
}

df_TSNE_2D=pd.DataFrame(dict_TSNE_2D)
#print(df_2D_PCA)


# In[10]:


traces=[]

for n in range(0,number_of_colors):
    df_category=df_TSNE_2D[(df_TSNE_2D.color_indicies==n)]
    point_name=str(df_category['labels'][df_category.index[0]])
    trace = go.Scatter(
    x=df_category['pca-1'],
    y=df_category['pca-2'],
    mode='markers',
    marker=dict(
        colorscale='Viridis',
        opacity=0.7,
        line=dict(color='rgb(140, 140, 170)')
    ),
    name=point_name,
    text=df_category['image_names'],
    )

    traces.append(trace)

layout = go.Layout(
    title='t-SNE reduction: 2D features',
    hovermode='closest',
    margin=dict(
       l=30,
       r=30,
       b=30,
       t=30
    ),
      scene = dict(
        xaxis = dict(
            title='pca-1',
       ),
       yaxis = dict(
           title='pca-2',
       ),
    ),
 
)

fig = go.Figure(data=traces, layout=layout)
iplot(fig)


# In[11]:


#t-SNE 3D

tsne_3D = TSNE(perplexity=20, n_components=3, verbose=2, init='pca',
               n_iter=4000, n_iter_without_progress=1000, learning_rate=25, method='exact')
low_dim_embs_3D = tsne_3D.fit_transform(features)


# In[12]:


dict_TSNE_3D = {'pca-1' : low_dim_embs_3D[:,0],
     'pca-2' : low_dim_embs_3D[:,1],
     'pca-3' : low_dim_embs_3D[:,2],
     'labels' : labels,
     'colors' : color_indicies,
     'image_names': image_names}

df_3D=pd.DataFrame(dict_TSNE_3D)


# In[13]:


traces=[]

for n in range(0,number_of_colors):
    df_category=df_3D[(df_3D.colors==n)]
    color=custom_color_map[n] if n < len(custom_color_map) else default_color
    point_name=str(df_category['labels'][df_category.index[0]])
    trace = go.Scatter3d(
    x=df_category['pca-1'],
    y=df_category['pca-2'],
    z=df_category['pca-3'],
    mode='markers',
    marker=dict(
        color=color,
        opacity=0.7,
        line=dict(color='rgb(140, 140, 170)')
    ),
    name=point_name,
    text=df_category['image_names'],
    )

    traces.append(trace)


data = traces

layout = go.Layout(
    title='t-SNE reduction: 3D features',
    margin=dict(
       l=30,
       r=30,
       b=30,
       t=30
    ),
      scene = dict(
        xaxis = dict(
            title='pca-1',
       ),
       yaxis = dict(
           title='pca-2',
       ),
        zaxis = dict(
            title='pca-3',
        ),
    ),
 
)


fig = go.Figure(data=data, layout=layout)
iplot(fig)


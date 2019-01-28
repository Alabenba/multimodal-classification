import os
import re
import pandas as pd
import numpy as np

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import plotly
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.plotly as py

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)


def calculate_means(features_filename='data/images/extracted_features.pkl', output_filename='data/images/mean_feature_vectors.pkl'):

    df_extracted_features = pd.read_pickle(features_filename)

    # In[3]:
    features=df_extracted_features.features.values.tolist()
    labels=df_extracted_features.labels.values.tolist()
    image_names=df_extracted_features.image_names.values.tolist()
    index=df_extracted_features.index.values.tolist()

    len_of_feature_vector=len(features[0])

    unique_labels=sorted(set(labels))
    number_of_colors=len(unique_labels)
    possible_color_indicies=np.linspace(0,number_of_colors-1,number_of_colors,dtype=int)
    labels2color_indices=dict(zip(unique_labels,possible_color_indicies))

    print(labels2color_indices)

    color_indicies=[]
    for l in labels:
        color_index=labels2color_indices[l]
        color_indicies.append(color_index)

    # In[4]:
    df_extracted_features

    # In[5]:
    labels_for_mean_vectors=[]
    names_for_mean_vectors=[]
    mean_feature_vectors_of_images_in_recipes=[]
    sum_of_vectors=[0]*len_of_feature_vector
    num_of_images_in_recipe=0
    last_image_name_prefix=0
    current_image_name_prefix=1
    i=0

    while i < len(index):
        image_name=image_names[i]
        image_name_splitted=re.split('-',image_name)
        image_name_splitted=re.split('\.',image_name_splitted[0])
        last_image_name_prefix=current_image_name_prefix
        current_image_name_prefix=image_name_splitted[0]
        if int(current_image_name_prefix) == int(last_image_name_prefix):
            num_of_images_in_recipe=num_of_images_in_recipe+1
            sum_of_vectors=list(map(lambda x,y:x+y,sum_of_vectors,features[i]))
            i=i+1
        else:
            labels_for_mean_vectors.append(labels[i-1])
            names_for_mean_vectors.append(last_image_name_prefix)
            mean_feature_vector=list(map(lambda x:x/num_of_images_in_recipe,sum_of_vectors)) 
            mean_feature_vectors_of_images_in_recipes.append(mean_feature_vector)
            num_of_images_in_recipe=0
            sum_of_vectors=[0]*len_of_feature_vector
            
            
    labels_for_mean_vectors.append(labels[i-1])
    names_for_mean_vectors.append(last_image_name_prefix)
    mean_feature_vector=list(map(lambda x:x/num_of_images_in_recipe,sum_of_vectors)) 
    mean_feature_vectors_of_images_in_recipes.append(mean_feature_vector)

    print(np.shape(mean_feature_vectors_of_images_in_recipes))

    # # In[6]:
    # #PCA

    # pca = PCA(n_components=2)
    # pca_2D_embeddings = pca.fit_transform(mean_feature_vectors_of_images_in_recipes)

    # dict_PCA_2D = {
    #     '1' : pca_2D_embeddings[:,0],
    #     '2' : pca_2D_embeddings[:,1],
    #     'mean_vector_labels' : labels_for_mean_vectors,
    #     'mean_vector_names': names_for_mean_vectors
    # }


    # df_PCA_2D=pd.DataFrame(dict_PCA_2D)
    # #print(df_2D_PCA)

    # # In[9]:
    # traces=[]

    # for l in unique_labels:
    #     df_category=df_PCA_2D[(df_PCA_2D.mean_vector_labels==l)]
    # #     df_category=df_PCA_2D[(df_TSNE_2D.mean_vector_labels==l)]
    #     point_name=str(df_category.mean_vector_labels[df_category.index[0]])
    #     trace = go.Scatter(
    #     x=df_category['1'],
    #     y=df_category['2'],
    #     mode='markers',
    #     marker=dict(
    #         colorscale='Viridis',
    #         opacity=0.7,
    #         line=dict(color='rgb(140, 140, 170)')
    #     ),
    #     name=point_name,
    #     text=df_category.mean_vector_names,
    #     )

    #     traces.append(trace)

    # layout = go.Layout(
    #     title='PCA reduction: 2D features',
    #     hovermode='closest',
    #     margin=dict(
    #     l=30,
    #     r=30,
    #     b=30,
    #     t=30
    #     ),
    #     scene = dict(
    #         xaxis = dict(
    #             title='pca-1',
    #     ),
    #     yaxis = dict(
    #         title='pca-2',
    #     ),
    #     ),
    
    # )

    # fig = go.Figure(data=traces, layout=layout)
    # iplot(fig)

    # # In[10]:
    # #t-SNE 2D

    # tsne2D = TSNE(perplexity=15, n_components=2, verbose=2, init='pca', n_iter=2000,
    #             n_iter_without_progress=500, learning_rate=20,method='exact')
    # tsne_2D_embeddings = tsne2D.fit_transform(mean_feature_vectors_of_images_in_recipes)

    # dict_TSNE_2D = {
    # '1' : tsne_2D_embeddings[:,0],
    # '2' : tsne_2D_embeddings[:,1],
    # 'mean_vector_labels' : labels_for_mean_vectors,
    # 'mean_vector_names': names_for_mean_vectors
    # }

    # df_TSNE_2D=pd.DataFrame(dict_TSNE_2D)
    # #print(df_2D_PCA)

    # # In[11]:
    # traces=[]

    # for l in unique_labels:
    #     df_category=df_TSNE_2D[(df_TSNE_2D.mean_vector_labels==l)]
    #     point_name=str(df_category.mean_vector_labels[df_category.index[0]])
    #     trace = go.Scatter(
    #     x=df_category['1'],
    #     y=df_category['2'],
    #     mode='markers',
    #     marker=dict(
    #         colorscale='Viridis',
    #         opacity=0.7,
    #         line=dict(color='rgb(140, 140, 170)')
    #     ),
    #     name=point_name,
    #     text=df_category.mean_vector_names,
    #     )

    #     traces.append(trace)

    # layout = go.Layout(
    #     title='t-SNE reduction: 2D features',
    #     hovermode='closest',
    #     margin=dict(
    #     l=30,
    #     r=30,
    #     b=30,
    #     t=30
    #     ),
    #     scene = dict(
    #         xaxis = dict(
    #             title='pca-1',
    #     ),
    #     yaxis = dict(
    #         title='pca-2',
    #     ),
    #     ),
    
    # )

    # fig = go.Figure(data=traces, layout=layout)
    # iplot(fig)

    # In[12]:
    dict_mean_feature_vectors = {
        'mean_feature_vectors': mean_feature_vectors_of_images_in_recipes,
        'mean_vector_labels' : labels_for_mean_vectors,
        'mean_vector_names': names_for_mean_vectors
    }

    df_mean_feature_vectors=pd.DataFrame(dict_mean_feature_vectors)


    print("Saving mean feature vectors to: ",
        output_filename)
    df_mean_feature_vectors.to_pickle(output_filename)

    print("Mean feature vectors pandas dataframe: \n")
    df_mean_feature_vectors


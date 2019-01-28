import os
import pandas as pd
import numpy as np
from tensorflow.python import pywrap_tensorflow
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import plotly
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.plotly as py


def embed(path_to_preprocessed_texts='data/texts/preprocessed_texts_for_doc2vec.pkl',
    path_to_saved_model='data/texts/models/doc2vec_recipes_checkpoint.ckpt',
    path_to_save_docs_embeddings_pkl='data/texts/docs_extracted_features.pkl'):

    # In[3]:
    def show_word_and_doc_embeddings():
        for tensor_name in var_to_shape_map:
            print("tensor_name: ", tensor_name) # get the variable name
            print(reader.get_tensor(tensor_name)) # get the variable
        
    reader = pywrap_tensorflow.NewCheckpointReader(path_to_saved_model)
    var_to_shape_map = reader.get_variable_to_shape_map() # 'var_to_shape_map' is a dictionary contains every tensor in the model
    show_word_and_doc_embeddings()

    doc_embeddings = reader.get_tensor('doc_embeddings')
    doc_embeddings = doc_embeddings.tolist() 
        
    df_preprocessed_texts = pd.read_pickle(path_to_preprocessed_texts)

    # In[5]:
    dict_doc_embeddings = {
    'text_names': df_preprocessed_texts.text_names,
    'labels' : df_preprocessed_texts.labels,
    'doc_embeddings' : doc_embeddings
    }
    

    df_doc_embeddings=pd.DataFrame(dict_doc_embeddings)

    # In[6]:
    doc_embeddings=df_doc_embeddings.doc_embeddings.values.tolist()
    labels=df_doc_embeddings.labels.values.tolist()
    text_names=df_doc_embeddings.text_names.values.tolist()

    unique_labels=sorted(set(labels))
    number_of_recipe_categories=len(unique_labels)
    possible_category_indices=np.linspace(0,number_of_recipe_categories-1,number_of_recipe_categories,dtype=int)
    labels2integers=dict(zip(unique_labels,possible_category_indices))

    print(labels2integers)

    category_indices=[]
    for l in labels:
        category_index=labels2integers[l]
        category_indices.append(category_index)

    # In[7]:
    #PCA

    pca = PCA(n_components=2)
    pca_2D_embeddings = pca.fit_transform(doc_embeddings)

    dict_PCA_2D = {
        'pca-1' : pca_2D_embeddings[:,0],
        'pca-2' : pca_2D_embeddings[:,1],
        'labels' : labels,
        'category_indices': category_indices,
        'text_names': text_names
    }

    df_PCA_2D=pd.DataFrame(dict_PCA_2D)

    # In[8]:
    traces=[]

    for n in range(0,number_of_recipe_categories):
        df_category=df_PCA_2D[(df_PCA_2D.category_indices==n)]
        point_name=str(df_category.labels[df_category.index[0]])
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
        text=df_category['text_names'],
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
        
    tsne2D = TSNE(perplexity=35, n_components=2, verbose=2, init='pca', n_iter=2000,
                    n_iter_without_progress=500, learning_rate=5,method='exact')
    tsne_2D_embeddings = tsne2D.fit_transform(doc_embeddings)
        
    dict_TSNE_2D = {
        'pca-1' : tsne_2D_embeddings[:,0],
        'pca-2' : tsne_2D_embeddings[:,1],
        'labels' : labels,
        'category_indices': category_indices,
        'text_names': text_names
    }

    df_TSNE_2D=pd.DataFrame(dict_TSNE_2D)

    # In[10]:
    traces=[]

    for n in range(0,number_of_recipe_categories):
        df_category=df_TSNE_2D[(df_TSNE_2D.category_indices==n)]
        point_name=str(df_category.labels[df_category.index[0]])
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
        text=df_category['text_names'],
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
    dict_docs_extracted_features = {
        'text_names': text_names,
        'labels' : labels,
        'features' : doc_embeddings   
    }

    df_docs_extracted_features=pd.DataFrame(dict_docs_extracted_features)


    print("Saving extracted features pandas dataframe to: ", path_to_save_docs_embeddings_pkl)
    df_docs_extracted_features.to_pickle(path_to_save_docs_embeddings_pkl)

    print("Extracted features pandas dataframe:")
    df_docs_extracted_features


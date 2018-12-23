# Multimodal classification of cooking recipes

## Text features extraction

### [Doc2vec extracted features:](http://nbviewer.jupyter.org/github/xkaple01/multimodal-classification/blob/text_feature_extraction/text_feature_extraction/doc_embeddings_visualisation.ipynb)
![](text_feature_extraction/graphs_and_visual_objects/doc2vec_v2_1_tsne.png)

### Nearest vectors to specified words:
![](text_feature_extraction/graphs_and_visual_objects/doc2vec_v2_1_nearest.png)

### [Text preprocessing pipeline for doc2vec using:](http://nbviewer.jupyter.org/github/xkaple01/multimodal-classification/blob/text_feature_extraction/text_feature_extraction/prepare_texts_for_doc2vec.ipynb)
1) remove numbers from text
2) lowercase
3) retain only nouns
4) lemmatization
5) stop words removing (minute,gram,pound,...)
6) delete rare nouns

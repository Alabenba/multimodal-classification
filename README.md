
# Multimodal classification

## Introduction

The classification task of the real-world multimedia data can be sufficiently complex for a single machine learning model. One of the possible ways to achive more precise results lies in the using of the multiple models to process multiple modalities (e.g. visual modality and textual modality) of the same datasource.  
The goal of the project is to show the performance of the multimodal classification approach on the common real-world dataset.  
The introduced multimodal classification method is based on using of the 3 neural networks:
- The first neural network extracts features from the visual modality
- The second neural network extracts features from the textual modality
- The third neural network decides which of 2 modalities is more informative one and then performs the classification with paying more attention to the more informative modality and less attention to the less informative modality  

## Dataset

Dataset consits of 316 cooking recipes collected from the internet portals. The photos of the cooked meal represent the visual modality (input to the first neural network) and the text of recipe (ingredients and cooking method) is the textual modality (input to the second neural network).  
Collected dataset is complex enough: dataset consists of 27 food categories. Some of categories were choosen such that the repices are sufficiently similar (for example, sashimi and sushi have some common ingredients; cooked meals looks similarly).  Some recipes are long and detailed, contain multiple photos of the meal, while another recipes are very short.   

##  Image feature extraction

### Neural network architecture:
CNN consiting of the 13 convilutional layers, 5 max pulling layers and 2 dense layers extracts feature vector of length 202 from each image. In case that the recipe has multiple images, neural network extracts feature vector from each image, then the mean value of all obtained vectors is computed.  

![](image_feature_extraction/graphs_and_visual_objects/neural_net_architecture.png)

###  Extracted features visualization:



### - [t-SNE 3D](https://plot.ly/~xkaple01/185)
![](image_feature_extraction/graphs_and_visual_objects/27_classes_dataset_filtered_tsne.gif)





## Text features extraction
Feature extraction from the recipe text is performed in unsupervised manner via doc2vec neural network architecture adopted for our task. Obtained feature vectors of the recipes belonging to the same category are more similar than the feature vectors of recipes belonging to the different categories. The length of the feature vector is 27 (equal to the number of food categories)  

### Doc2vec architecture: Tensorflow implementation  
Both of classical doc2vec architectures (DM and DBOW) were implemented and tested, then the resulting neural network was obtained by such doc2vec modification that brings the best performance on our dataset.  
Training phase architecture:    
![](text_feature_extraction/graphs_and_visual_objects/doc2vec_train_27cl.png)  
Inference phase architecture:    
![](text_feature_extraction/graphs_and_visual_objects/doc2vec_27cl_test.png)  
Feature vectors obtained during the training and the inference phases:
![](text_feature_extraction/graphs_and_visual_objects/doc2vec_test_ex_1.png)  



### Text preprocessing pipeline for doc2vec using:
1) removing the numbers from text:  
the text of 2 recipes can be the same except that the first recipe can be designed for 2 people, while the second - for 4 people; the amount of ingridients have not to be the discriminative feature for the given category of the cooking recipes
2) lowercasing the text
3) retaining only the nouns
4) performing lemmatization
5) removing of the stop words (gram, pound, ...):  
the text of 2 recipes can be the same except that the first recipe describes the needed amount of ingridients in grams, while another - in pounds 
6) deleting the rare nouns:  
rare is the noun which occurs less than 3 times in the all recipes belonging to the same category  

//### Nearest vectors to specified words:
//![](text_feature_extraction/graphs_and_visual_objects/doc2vec_v2_1_nearest.png)  

//### Doc2vec extracted features
//![](text_feature_extraction/graphs_and_visual_objects/doc2vec_v2_1_tsne.png)







## Multimodal classification - gaited multimodal unit:
Consider the following scenarios:  
- the text of the recipe is very detailed, but the photo of the meal is taken at the wrong angle:  
in this case the recipe can be successfully classified based on the textual modality
- the recipe is too exotic, but the meal on the photo doesn't differs too much from the other meals of this category:
in this case the recipe can be successfully classified based on the visual modality  

The task of the gaited multimodal unit is to estimate how informative is the visual modality and how informative is the textual modality of the given recipe. More informative modality is more important for the final classification performed by the GMU.  

GMU schema:    
![](multimodal_classification/graphs_and_visual_objects/gmu_cropped.png)  


Modified GMU:  
![](multimodal_classification/graphs_and_visual_objects/gaited_multimodal_unit_graph.png)  
In case that the lengths of image feature vector and text feature vector are different enough, the following modification is suiatable: additional dense layer allows to reduce the length of the image feature vector (202) such that the size of both modalities is the same (12). After the dimensionality reduction, both vectors of the same length are inputted to the classical GMU.  


### Results of multimodal classification  
Gaited multimodal unit was tested on 13 randomly choosen recipes (32 were used for training). 
Test dataset:  
- 3 recipes of sashimi (label 0)
- 3 recipes of steak (label 1)
- 4 recipes of sushi (labes 2)
- 3 recipes of tiramisu (label 3)  

![](multimodal_classification/graphs_and_visual_objects/result.png)   

GMU classified all 13 recipes correctly:
- sashimi category was classified by paying more attention to the visual modality
- steak category was differed from the other categories exclusively based on the textual modality
- sushi category was classified based on the visual modality
- tiramisu category differs from others by the textual modality 
  
The explanation of the results might be the following:  
- sushi and sashimi looks very different from the other meals, so these categories can be easily differed from the other categories based on the photos
- the photos of the steak and the tiramisu might not be sufficiently discriminative (especially the close-up photos: both meals can look as the brown substance), so it is more suitable to classify mentoined categories based on the recipe text
- in case that the classification based almost on the only single modality is hard, GMU can give almost the same importance to the textual and to the visual modality: test sample number 1 (recipe of sashimi) was correctly classified by giving 0.42 of attention to the textual modality and 0.58 to the visual modality   


# Accuracy 100%

import os
import re
import h5py
import numpy as np
import pandas as pd

import keras
from keras.utils.io_utils import HDF5Matrix
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, MaxPool2D, Flatten, Dense, Dropout, Activation, Input
from keras.optimizers import SGD
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator

def extract_features(image_dir='data/images', models_filename='data/v8_vgg16_model_1.h5'):
      
      image_size = (224, 224)
      batch_size = 1
      epochs = 80

      # In[4]:
      image_data_generator = ImageDataGenerator(
            rescale = 1./255,
            horizontal_flip = False,
            fill_mode = "nearest",
            zoom_range = 0,
            width_shift_range = 0,
            height_shift_range=0,
            rotation_range=0)


      generator_flow = image_data_generator.flow_from_directory(
            image_dir,
            target_size = (image_size[0], image_size[1]),
            batch_size = batch_size, 
            class_mode = "categorical",
            shuffle=False)


      num_of_classes = len(generator_flow.class_indices)
      total_number_of_images=generator_flow.n

      # In[5]:
      images_label_and_name=generator_flow.filenames
      labels=[]
      image_names=[]

      for i in images_label_and_name:
            splitted_label_and_image_name=re.split('\/',i)
            labels.append(splitted_label_and_image_name[0])
            image_names.append(splitted_label_and_image_name[1])
      
      keys=generator_flow.class_indices.keys()
      food_categories_sorted=sorted(keys,key=str)

      # In[6]:
      model = VGG16(weights=None, include_top=False, input_shape=(image_size[0], image_size[1], 3))

      
      x = model.output
      x = Flatten()(x)
      x = Dense(101*2, activation="relu", name="dense_1")(x)
      x = Dense(101*2, activation="relu", name="dense_2")(x)
      predictions = Dense(101, activation="softmax", name="dense_3")(x)
      model_final = Model(input=model.input, output=predictions)
      model_final.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
      model_final.load_weights(models_filename)

      # In[7]:
      layer_name = 'dense_2'
      intermediate_output_model = Model(inputs=model_final.input,
                                    outputs=model_final.get_layer(layer_name).output)

      intermediate_output_model.summary()

      # In[8]:
      #Extracting features from images
      features=[]

      for n in range(total_number_of_images):
            batch = generator_flow.next()
            image = batch[0][0]
            extracted_features_from_image = intermediate_output_model.predict(np.asarray([image]))
            features.extend(extracted_features_from_image)
      
      print("Network extracts",len(features[1]), "dimensionality feature vector from each of",
            len(labels), "images")

      # In[9]:
      dict_images_extracted_features = {
            'image_names': image_names,
            'labels' : labels,
            'features' : features   
      }

      df_images_extracted_features=pd.DataFrame(dict_images_extracted_features)


      print("Saving extracted features pandas dataframe to: ",
            os.path.join(image_dir,"extracted_features.pkl"))
      df_images_extracted_features.to_pickle(os.path.join(image_dir,"extracted_features.pkl"))

      print("Extracted features pandas dataframe: \n",df_images_extracted_features)

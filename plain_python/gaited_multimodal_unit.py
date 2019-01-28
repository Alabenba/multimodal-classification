import os
import pandas as pd
import tensorflow as tf
import numpy as np
# import matplotlib.pyplot as plt
import shutil


def classify(visual_m_path='data/images/mean_feature_vectors.pkl',
    textual_m_path='data/texts/docs_extracted_features.pkl'):

    df_visual_m = pd.read_pickle(visual_m_path)
    df_textual_m = pd.read_pickle(textual_m_path)

    number_of_recipes=len(df_visual_m)
    unique_labels=sorted(set(df_visual_m.mean_vector_labels.values))
    number_of_classes=len(unique_labels)
    possible_class_indices=list(range(0,number_of_classes))
    labels2class_indices=dict(zip(unique_labels,possible_class_indices))
    print(labels2class_indices)

    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    # In[8]:
    test_samples=[3,5,6,
                11,17,18,
                23,26,33,34,
                38,39,44
                ]
    all_samples=set(range(0,number_of_recipes))
    train_samples=list(all_samples.difference(test_samples))
    print("train samples: ", train_samples)
    print("test samples: ", test_samples)

    # In[9]:
    df_visual_m_train=df_visual_m.iloc[train_samples]
    df_visual_m_test=df_visual_m.iloc[test_samples]
    df_textual_m_train=df_textual_m.iloc[train_samples]
    df_textual_m_test=df_textual_m.iloc[test_samples]

    visual_m_train_inputs=list(df_visual_m_train.mean_feature_vectors.values)
    visual_m_test_inputs=list(df_visual_m_test.mean_feature_vectors.values)

    textual_m_train_inputs=list(df_textual_m_train.features.values)
    textual_m_test_inputs=list(df_textual_m_test.features.values)

    train_correct_class_ids=[labels2class_indices[l] for l in df_visual_m_train.mean_vector_labels]
    test_correct_class_ids=[labels2class_indices[l] for l in df_visual_m_test.mean_vector_labels]

    number_of_training_samples=len(visual_m_train_inputs)
    number_of_test_samples=len(visual_m_test_inputs)
    len_of_visual_features_vec=len(visual_m_train_inputs[0])
    len_of_textual_features_vec=len(textual_m_train_inputs[0])

    print(test_correct_class_ids)
    print(np.shape(visual_m_train_inputs))
    print(np.shape(textual_m_train_inputs))

    # In[10]:
    batch_size=4
    learning_rate=0.01
    hidden_state_dim = 6
    z_dim=1
    number_of_training_iterations=500
    print_valid_every=10
    # num_repeat_training=10
    num_repeat_training=1

    # In[11]:
    def create_training_batch():
        inputs_visual=[]
        inputs_textual=[]
        correct_classes=[]
        for i in range(batch_size):
            train_sample_index=np.random.choice(range(0,number_of_training_samples),1)[0]
            inputs_visual.append(visual_m_train_inputs[train_sample_index])
            inputs_textual.append(textual_m_train_inputs[train_sample_index])
            correct_classes.append(train_correct_class_ids[train_sample_index])
        return np.array(inputs_visual),np.array(inputs_textual),np.array(correct_classes)

    inputs_visual,inputs_textual,correct_classes=create_training_batch()
    print(np.array(correct_classes))
    print(np.shape(inputs_visual))
    print(np.shape(inputs_textual))
    print(np.shape(correct_classes))

    # In[12]:
    visual = tf.placeholder(tf.float32, shape=[None,len_of_visual_features_vec])
    textual = tf.placeholder(tf.float32, shape=[None,len_of_textual_features_vec])
    target = tf.placeholder(tf.int32, shape=[None])

    v_reduced = tf.layers.dense(visual,
                        2*hidden_state_dim,
                        activation=tf.nn.relu)
    h_v = tf.layers.dense(v_reduced,
                        hidden_state_dim,
                        activation=tf.nn.tanh)
    h_t = tf.layers.dense(textual,
                        hidden_state_dim,
                        activation=tf.nn.tanh)
    z = tf.layers.dense(tf.concat([v_reduced,textual], axis=1),
                        z_dim,
                        activation=tf.nn.sigmoid)
    h = z * h_v + (1 - z) * h_t


    logits = tf.layers.dense(h, number_of_classes)
    scores = tf.nn.sigmoid(logits)

    multi_class_labels=tf.one_hot(target, depth=number_of_classes)
    loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(multi_class_labels=multi_class_labels,
                                        logits=logits))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss)

    correct_prediction=tf.equal(tf.argmax(logits, axis=1), tf.argmax(multi_class_labels,axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver(None)

    # In[13]:
    def show_validation_result(accuracy_res, loss_res, scores_res,z_res):
        print("Validation: Loss: ", loss_res," Accuracy: ", accuracy_res)
        df_scores=pd.DataFrame(data={'Class scores':list(np.around(scores_res,4)), 'Correct labels':test_correct_class_ids,
                                'Trust to visual modality': list(np.around(z_res,4)),
                                'Trust to textual modality': list(np.around(1-z_res,4))})
        print(df_scores)  
        

    def train(num_of_run):
        with tf.Session() as session:        
            session.run(tf.global_variables_initializer())
        
            print("Start model training")
        
            for train_iter in range(number_of_training_iterations):
                inputs_visual,inputs_textual,correct_classes=create_training_batch()
                _, l = session.run([train_op, loss], {visual: inputs_visual,
                                            textual: inputs_textual,
                                            target: correct_classes})
                print("Training: loss: ", l)
            
                if (train_iter+1)%print_valid_every==0:
                    accuracy_res, loss_res, scores_res,z_res = session.run([accuracy, loss, scores, z],
                                                                            {visual: visual_m_test_inputs,
                                                                            textual: textual_m_test_inputs,
                                                                            target: test_correct_class_ids})        
                    
                    show_validation_result(accuracy_res, loss_res, scores_res,z_res)
                    
                    if accuracy_res==1.0: 
                        return 0
            return -1

    # In[14]:
    for num_of_run in range(num_repeat_training):            
        if train(num_of_run)==0:
            break


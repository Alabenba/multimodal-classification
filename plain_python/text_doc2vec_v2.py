import os
import numpy as np
import pandas as pd
import collections
import tensorflow as tf
import pickle

def doc2vec(models_folder_name='data/texts/models', path_to_preprocessed_texts='data/texts/preprocessed_texts_for_doc2vec.pkl'):
    # In[2]:
    # models_folder_name = os.path.join(os.getcwd(),'models')
    # path_to_preprocessed_texts = os.path.join(os.getcwd(),
    #                                         'texts','preprocessed_texts_for_doc2vec.pkl') 

    df_preprocessed_texts = pd.read_pickle(path_to_preprocessed_texts)

    preprocessed_texts = df_preprocessed_texts.preprocessed_texts.values.tolist()
    labels = df_preprocessed_texts['labels'].values.tolist()

    unique_labels=sorted(set(labels))
    number_categories=len(unique_labels)
    categories_indices=np.linspace(0,number_categories-1,number_categories,dtype=int)
    labels2integers=dict(zip(unique_labels,categories_indices))

    print(labels2integers)

    # In[3]:
    batch_size = 2
    generations = 100000
    model_learning_rate = 0.001

    embedding_size = 24   #word embedding size
    doc_embedding_size = 12  #document embedding size
    concatenated_size = embedding_size + doc_embedding_size

    save_embeddings_every = 5000
    print_valid_every = 5000
    print_loss_every = 50

    # In[4]:
    def build_dictionary(preprocessed_texts):
        words=[w for words_in_recipe in preprocessed_texts for w in words_in_recipe]
        count = []
        count.extend(collections.Counter(words))
        count=sorted(count)
        word_dict = {}
        for word in count:
            word_dict[word] = len(word_dict)
        
        return (word_dict)

    #replace each word in texts with integer value
    def text_to_numbers(preprocessed_texts, word_dict):
        data = []
        for prepr_text in preprocessed_texts:
            text_data = []
            for word in prepr_text:
                if word in word_dict:
                    word_ix = word_dict[word]
                else:
                    word_ix = 0
                text_data.append(word_ix)
            data.append(text_data)
        return (data)


    def create_batch_data(text_with_words_conv_to_numbers, batch_size=batch_size):
        batch_data = []
        label_data = []
        
        rand_text_ix = int(np.random.choice(len(text_with_words_conv_to_numbers), size=1))
        rand_text = text_with_words_conv_to_numbers[rand_text_ix]
        word_to_predict_label=np.random.choice(list(set(rand_text)), size=1,replace=False)
        
        while len(batch_data) < batch_size:
            item_in_batch=[]        
            
            label_words=np.random.choice(rand_text, size=1,replace=False)

            item_in_batch.extend(word_to_predict_label)
            item_in_batch.append(rand_text_ix)     
            label_data.extend(label_words)
            batch_data.append(item_in_batch)

            
        batch_data = np.array(batch_data)
        label_data = np.transpose(np.array(label_data))

        return (batch_data, label_data)

    # In[5]:
    word_dictionary=build_dictionary(preprocessed_texts)
    vocabulary_size=len(word_dictionary)
    print(word_dictionary)
    print(vocabulary_size)

    word_dictionary_rev = dict(zip(word_dictionary.values(), word_dictionary.keys()))

    # In[6]:
    text_data = text_to_numbers(preprocessed_texts, word_dictionary)
    print(text_data)

    # In[7]:
    valid_words = ['tuna', 'rice', 'sushi', 'roll', 'sashimi','steak','grill', 'sauce', 'cream']

    valid_examples = [word_dictionary[x] for x in valid_words]
    print(valid_examples)

    # In[8]:
    batch_data, label_data = create_batch_data(preprocessed_texts)
    print(batch_data)
    print(label_data)
    print(np.shape(label_data))

    # In[9]:
    sess = tf.Session()

    # In[10]:
    print('Creating Model')

    embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    doc_embeddings = tf.Variable(tf.random_uniform([len(preprocessed_texts), doc_embedding_size], -1.0, 1.0))

    decoder_weights = tf.Variable(tf.truncated_normal([vocabulary_size, concatenated_size],
                                                stddev=1.0 / np.sqrt(concatenated_size)))
    decoder_biases = tf.Variable(tf.zeros([vocabulary_size]))


    x_inputs = tf.placeholder(tf.int32, shape=[None, 2]) #1 for word index and 1 for doc index
    y_target = tf.placeholder(tf.int32, shape=[batch_size])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    embed= tf.nn.embedding_lookup(embeddings, x_inputs[:, 0])
        
    doc_indices = tf.slice(x_inputs, [0,1],[batch_size,1])
    doc_embed = tf.nn.embedding_lookup(doc_embeddings,doc_indices)
    final_embed = tf.concat([embed, tf.squeeze(doc_embed)],1)

    logits = tf.matmul(final_embed, tf.transpose(decoder_weights)) + decoder_biases


    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=y_target))
    optimizer = tf.train.AdamOptimizer(learning_rate=model_learning_rate)
    train_step = optimizer.minimize(loss)

    #cosine similarity between words
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)


    saver = tf.train.Saver({"embeddings": embeddings, "doc_embeddings": doc_embeddings})

    init = tf.initialize_all_variables()
    sess.run(init)


    print('Starting Training')

    loss_vec = []
    loss_x_vec = []
    for i in range(generations):
        batch_inputs, batch_labels = create_batch_data(text_data)
        feed_dict = {x_inputs : batch_inputs, y_target : batch_labels}

        #run the train step
        sess.run(train_step, feed_dict=feed_dict)

        #return the loss
        if (i+1) % print_loss_every == 0:
            loss_val = sess.run(loss, feed_dict=feed_dict)
            loss_vec.append(loss_val)
            loss_x_vec.append(i+1)
            print('Loss at step {} : {}'.format(i+1, loss_val))
            
        #validation: print some random words and top 5 related words
        if (i+1) % print_valid_every == 0:
            sim = sess.run(similarity, feed_dict=feed_dict)
            for j in range(len(valid_words)):
                valid_word = word_dictionary_rev[valid_examples[j]]
                top_k = 5 # number of nearest neighbors
                nearest = (-sim[j, :]).argsort()[1:top_k+1]
                log_str = "Nearest to {}:".format(valid_word)
                for k in range(top_k):
                    close_word = word_dictionary_rev[nearest[k]]
                    log_str = '{} {},'.format(log_str, close_word)
                print(log_str)
                
        #save dictionary + embeddings
        if (i+1) % save_embeddings_every == 0:
            #save vocabulary dictionary
            os.makedirs(models_folder_name, exist_ok=True)
            with open(os.path.join(models_folder_name,'doc2vec_recipes_dict_words_integers.pkl'), 'wb') as f:
                pickle.dump(word_dictionary, f)
            
            #save embeddings
            model_checkpoint_path = os.path.join(os.getcwd(),models_folder_name,'doc2vec_recipes_checkpoint.ckpt')
            save_path = saver.save(sess, model_checkpoint_path)
            print('Model saved in file: {}'.format(save_path))


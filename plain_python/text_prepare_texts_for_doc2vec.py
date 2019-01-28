import nltk
from nltk import WordNetLemmatizer, FreqDist
import pandas as pd
import os
import re
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')

def prepare_texts(path_to_raw_texts='data/texts/raw_texts', preprocessed_texts_file='data/texts/preprocessed_texts_for_doc2vec.pkl'):

    stopwords=['inch','pound','pint','tbsp','tablespoon','tsp','teaspoon',
            'g','gram','kg','f','l','ll','ml','t','s','cm','mm',
            'sec','m','minute','min','h','hour','%','eel','pre','lb','oz',
            '–','-','c',';','...','x','*','+','!','?','<','>',
            'ºf', 'ºc',
            'i','ii','iii','iv','v','vi','vii','viii',
            'ix','xi','xii','xiii','viiii','xiv','®','e','re',
            '’','’’','/','‘','‘‘',':','°',
            'spoon','dish','fork','knife','cup','cover','pan','jug','plate',
            'half','quarter',
            'first','second','third',
            'step','ingridients','method',
            'temperature','degree','doneness','room','flavor','heat','bowl','result','ve','one','size','chef','ease','hand',
            'transfer','edge','glass','plastic','repeat','use','side','line','content','kitchen',
            'video','underside','photo','motion','place','top','cooking','thick','closest',
            'need','centre','way','grade','tip','lock','shape','length','width','lengthwise',
            'log','lengthways', 'crosswise','front','start','finish','cloth','film','board',
            'ingredient','cook','center','taste','pair','segment','chip','presentation','gras',
            'ounce','end','time','package','serving','serve']
    #nltk can't lemmatize word 'ingridients', so it is in plural in the list of stopwords
    word_importance_low_bound=3

    # In[19]:
    def read_recipe_texts(path_to_raw_texts):
        return nltk.corpus.reader.plaintext.PlaintextCorpusReader(path_to_raw_texts,'.*\.txt')

    def basic_text_preprocessing(corpus):
        labels=[]
        text_names=[]
        preprocessed_texts=[]
        all_words_in_food_categories={}
        
        abspaths=corpus.abspaths()
        lemmatizer = WordNetLemmatizer()

        for path_of_raw_recipe in abspaths:
        
            splitted_path=re.split('\/',path_of_raw_recipe)
            length_of_path=len(splitted_path)
            food_category=splitted_path[length_of_path-2]
            labels.append(food_category)
            text_name=splitted_path[length_of_path-1]
            text_names.append(text_name)
        
            recipe_text=corpus.words(path_of_raw_recipe)
        
            text_without_numbers=[]
            for word in recipe_text:
                text_without_numbers.append(re.sub(r'\d+', '', word)) #sushi2 -> sushi
            
            lowercase_text=' '.join(text_without_numbers).lower()
            tokens = nltk.word_tokenize(lowercase_text)
            tags = nltk.pos_tag(tokens)
            nouns = [word for word,pos in tags if (pos == 'NN'  or  pos == 'NNS')]
        
            l_nouns=[]
            for n in nouns:
                l_nouns.append(lemmatizer.lemmatize(n))
            
            filtered_nouns = [l_noun for l_noun in l_nouns if l_noun not in stopwords]
        
            all_words_in_food_categories.setdefault(food_category,[])
            all_words_in_food_categories[food_category].extend(filtered_nouns)
            
            preprocessed_texts.append(filtered_nouns)  
            
        return labels,text_names,preprocessed_texts,all_words_in_food_categories

    def find_most_common_word_in_food_categories(all_words_in_food_categories):
        dict_category_common_words={}
        for category,all_words_in_food_category in all_words_in_food_categories.items():
            words_frequency_distribution=FreqDist(all_words_in_food_category)
            common_words=[word for word, num_of_occur in words_frequency_distribution.items() if
                        num_of_occur > word_importance_low_bound]
            dict_category_common_words[category]=common_words
        return dict_category_common_words

    def filter_rare_words_in_recipes(labels,preprocessed_texts,common_words_in_food_categories):
        important_words_in_texts=[]
        num_of_important_words_in_texts=[]
        
        for ix in range(len(preprocessed_texts)):
            food_category=labels[ix]
            basically_preprocessed_text=preprocessed_texts[ix]
            common_words_for_food_category=common_words_in_food_categories[food_category]
            important_words=[word for word in basically_preprocessed_text if word in common_words_for_food_category]
            important_words_in_texts.append(important_words)
            num_of_important_words_in_texts.append(len(important_words))
            
        return important_words_in_texts, num_of_important_words_in_texts

    # In[20]:
    corpus = read_recipe_texts(path_to_raw_texts)

    labels,text_names,preprocessed_texts,all_words_in_food_categories=basic_text_preprocessing(corpus)

    common_words_in_food_categories=find_most_common_word_in_food_categories(all_words_in_food_categories)
        
    important_words_in_texts,num_of_important_words_in_texts=filter_rare_words_in_recipes(labels,preprocessed_texts,common_words_in_food_categories)

    dict_preprocessed_texts = {
        'text_names': text_names,
        'labels' : labels,
        'preprocessed_texts' : important_words_in_texts,
        'number_of_important_words' : num_of_important_words_in_texts
    }
        
    df_preprocessed_texts=pd.DataFrame(dict_preprocessed_texts)  

    print("Saving preprocessed texts pandas dataframe to: ", preprocessed_texts_file)
    df_preprocessed_texts.to_pickle(preprocessed_texts_file)

    print("Preprocessed texts pandas dataframe: \n")
    df_preprocessed_texts


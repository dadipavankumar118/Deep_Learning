import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm

import re
import emoji
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline




st.title('Movie Sub-titles Search Engine')


df = pd.read_json("bert_vectors_data.json")
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('paraphrase-MiniLM-L3-v2')




def emoji_remove(x):
    
    return x.apply(lambda x : emoji.demojize(x))

def decontration(x):
    
    x = x.apply(lambda x:re.sub(r"aren't", 'are not', x))
    x = x.apply(lambda x:re.sub(r"won't", 'will not', x))
    x = x.apply(lambda x:re.sub(r"doesn't", 'does not', x))
    x = x.apply(lambda x:re.sub(r"n\'t", " not", x))
    x = x.apply(lambda x:re.sub(r"\'s", " is", x))
    x = x.apply(lambda x:re.sub(r"\'d", " would", x))
    x = x.apply(lambda x:re.sub(r"\'ll", " will", x))
    x = x.apply(lambda x:re.sub(r"\'t", " not", x))
    x = x.apply(lambda x:re.sub(r"\'ve", " have", x))
    x = x.apply(lambda x:re.sub(r"\'m", " am", x))

    return x

def lowercase(x):
    
    return x.str.lower()

def html_tags(x):
    
    return x.apply(lambda x:re.sub("<.+?>"," ",x))

def urls(x):
    
    return x.apply(lambda x:re.sub("https[s]?://.+? +"," ",x))

def unwanted_characters(x):
    
    return x.apply(lambda x:re.sub("[^a-z\s0-9]"," ",x))

def lemmatization(x):
    
    list_stp = stopwords.words("english")
    wl = WordNetLemmatizer()

    def lemmatize_text(text):
        
        words = word_tokenize(text)
        lemmatized_words = [wl.lemmatize(word, pos="v") for word in words if word not in list_stp]

        return " ".join(lemmatized_words)

    return x.apply(lemmatize_text)


# Take user's search query
user_query = st.text_input("Enter your search query: ")

preprocess = pickle.load(open('Pre-Processing Model.pkl', 'rb'))

cleaned_qurey = preprocess.transform(pd.DataFrame([user_query],columns= ['text']))

tqdm.pandas()

Query = cleaned_qurey.progress_apply(model.encode)


df['text_vector_bert'] = df['text_vector_bert'].progress_apply(lambda x: [float(i) for i in x])

cos_sim = util.cos_sim(Query, df['text_vector_bert'])

similarity_data = cos_sim.argsort()[:][::].tolist()

similarity_list = []

for i in similarity_data:
    similarity_list.extend(i)

top_sub = df.iloc[similarity_list[::-1]]

top_sub = top_sub[: 10]


if st.button('Search'):
    st.write(top_sub[['name']])

    st.subheader("Top Subtitle")
    top_sub_top = top_sub.reset_index()
    st.write(top_sub_top.loc[0,'chunks_text'])
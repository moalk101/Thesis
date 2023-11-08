# %%
import gensim
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer
#from gensim.models import Word2Vec
import pandas as pd
import numpy as np
import gensim.downloader
# import re
#import matplotlib.pyplot as plt
# from matplotlib.pyplot import figure
# from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine
# from sklearn.ensemble import IsolationForest
# import random
import os
import ast

# %%
training_set = gensim.downloader.load('word2vec-google-news-300')


# %%
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# %%


def read_data(d,df):
     df.loc[len(df)]= [d["address"],d["html"],d["hash_code"],[],[]]
     df.at[len(df)-1,"html"]=df["html"][len(df)-1].lower()


# %%


# %%
def clean_tokens(tokens,df):
    tokens=[token for token in tokens if token.isalpha()]
    stop_words=set(stopwords.words("english"))
    tokens=[token for token in tokens if token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens=[lemmatizer.lemmatize(token,pos="v") for token in tokens]

    return list(set(tokens))

# %%
#preprocessed_tokens=preprocess(tokenized)


# %%
def add_embedding_indices(df):
    #df["embedding_index"]=[[] for i in range(0,len(df))]
    i=len(df)
    
    #for text in df["html"]:
    idxs=[]
    for word in df["html"][i-1]:
        if training_set.__contains__(word):
            idxs.append(training_set.key_to_index[word])
    df.at[i-1,"embedding_index"]=idxs    
    

# %%
def compute_mean_vectors(df):
    vectors= df["embedding_index"][len(df)-1]
    df.at[len(df)-1,"mean_vector"] = list(np.mean([training_set[index]
                                                    for index in vectors],
                                                      axis=0))






def prepare(d):
    df = pd.DataFrame(columns=["address","html","hash_code","embedding_index","mean_vector"])
    if not os.path.isfile("saved_documents/modified_data.csv"):
        df.to_csv("saved_documents/modified_data.csv",index=False)
    df=pd.read_csv("saved_documents/modified_data.csv")

    df["mean_vector"]=df["mean_vector"].apply(lambda x: ast.literal_eval(x))


    hashes=df["hash_code"].tolist()
    if d["hash_code"] in hashes:
        return
    
    read_data(d,df)
    df.at[len(df)-1,"html"]=clean_tokens(word_tokenize(df["html"][len(df)-1]),df)
    add_embedding_indices(df)
    compute_mean_vectors(df)
    df.to_csv("saved_documents/modified_data.csv",index=False) 


# !!! the following version of the prepare method was modified to suit the testing purposes

# def prepare(d,df):
#     # df = pd.DataFrame(columns=["address","html","hash_code","embedding_index","mean_vector"])
#     # if not os.path.isfile("modified_data_from_wikipedia.csv"):
#     #     df.to_csv("modified_data_from_wikipedia.csv",index=False)
#     # df=pd.read_csv("modified_data_from_wikipedia.csv")

#     # df["mean_vector"]=df["mean_vector"].apply(lambda x: ast.literal_eval(x))


#     hashes=df["hash_code"].tolist()
#     if d["hash_code"] in hashes:
#         return
    
#     read_data(d,df)
#     df.at[len(df)-1,"html"]=clean_tokens(word_tokenize(df["html"][len(df)-1]),df)
#     add_embedding_indices(df)
#     compute_mean_vectors(df)
#     # df.to_csv("modified_data_from_wikipedia.csv",index=False) 
      

def mean_vector(input):
    keys=[x for x in input.split()
           if training_set.__contains__(x)]
    meanVec=np.mean(np.array(list(
                                map(lambda x :
                                    training_set.get_vector(x),
                                    keys))),axis=0)
    return meanVec

# %%

#returns the most similar websites according the query 
def search(input,nresults):
    #indices_to_read=[0,2226,2227,2228,2229,2230]
    dataframe=pd.read_csv("saved_documents/modified_data.csv")
    dataframe["mean_vector"]=dataframe["mean_vector"].apply(lambda x: ast.literal_eval(x))
    sims=[(i,1-cosine(mean_vector(input),dataframe["mean_vector"][i]))
           for i in range(len(dataframe))]
    sims.sort(key=lambda x:x[1],reverse=True)
    links=[(dataframe["address"][i[0]],i[1]) for i in sims[0:nresults]]
    return links
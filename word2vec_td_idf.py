#!/usr/bin/env python
# coding: utf-8

# In[28]:


Author: Ilia Kabanov


# In[ ]:


Task: to predict wether product review is positive 


# In[1]:


# Import packages
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import string

from sklearn.metrics import make_scorer, roc_auc_score
import json
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from skopt.space import Integer, Real, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize
from scipy.sparse import csr_matrix, csc_matrix
from tqdm import tqdm
from gensim.models import Word2Vec
import re
from pymystem3 import Mystem
from pymorphy2 import MorphAnalyzer
from copy import deepcopy
from sklearn.svm import SVC

import gc
import os
import sys
import time
import itertools
from tqdm.notebook import tqdm
import pickle
import json
import joblib
import collections
import requests 
from urllib.parse import urlencode 


import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings

warnings.filterwarnings("ignore")


# In[3]:


# Read input data
train = pd.read_csv('train.csv', sep=',', engine='python')
test = pd.read_csv('test.csv', sep=',', engine='python' )
print(train.head())


# In[4]:


print(test.head())


# In[5]:


# Drop lines in X_train with empty 'sentiment' and lower all the words
train = train[train['sentiment'].notnull()]
train.reset_index(inplace=True)
train.drop(['index'], axis=1, inplace=True)
train.iloc[:, 0] = train.iloc[:, 0].apply(lambda x: x.lower())

# Drop lines with no id in X_train, and lower all the words 
test = test[test['id'].notnull()]
test = test[test['id'].apply(lambda x: x.isnumeric())]
test.reset_index(inplace=True)
test.drop(['index'], axis=1, inplace=True)
test.iloc[:, 1] = test.iloc[:, 1].apply(lambda x: x.lower())
test_ids = test['id']
X_test = test['review']

X_train = train.iloc[:, 0]
y_train = train['sentiment']


# In[ ]:


Then let's use lemmatization procedure with both X_train and X_test


# In[6]:


# Lemmatization
morph = MorphAnalyzer()
train_lemmatized_descriptions = []
for i, description in tqdm(enumerate(X_train.values), total=len(X_train)):
    try:
        lemmatized_description = [
            morph.parse(token)[0].normal_form for token in
            re.findall(r'\w+', description)
        ]
        train_lemmatized_descriptions.append(lemmatized_description)
    except Exception as e:
        print(f'Не удалось распарсить description с индексом={i}:')
        print("descrition:")
        print(description, end='\n\n')
        print(e, end='\n\n')
        train_lemmatized_descriptions.append([])


# In[7]:


morph = MorphAnalyzer()
test_lemmatized_descriptions = []
for i, description in tqdm(enumerate(test['review'].values), total=len(test['review'])):
    try:
        lemmatized_description = [
            morph.parse(token)[0].normal_form for token in
            re.findall(r'\w+', description)
        ]
        test_lemmatized_descriptions.append(lemmatized_description)
    except Exception as e:
        print(f'Не удалось распарсить description с индексом={i}:')
        print("descrition:")
        print(description, end='\n\n')
        print(e, end='\n\n')
        train_lemmatized_descriptions.append([])


# In[ ]:


Transform list with lemmatized words into dataset with previous form 


# In[33]:


X_train = pd.DataFrame(train_lemmatized_descriptions).fillna(value=' ', inplace=False)
X_train = X_train.apply(' '.join, axis=1)


# In[34]:


X_test = pd.DataFrame(test_lemmatized_descriptions).fillna(value=' ', inplace=False)
X_test = X_test.apply(' '.join, axis=1)


# In[ ]:


Lets have a look at our lemmatized train and test datasets


# In[35]:


X_train.head(5)


# In[36]:


X_test.head(5)


# In[ ]:


We got exactly what we need, our dataset are ready for embedding.
Let's count tf_idf of every word in text_corpus


# In[37]:


# Implementing td-idf and getting our X_train, y_train
russian_stop_words = list(
    pd.read_csv('https://raw.githubusercontent.com/stopwords-iso/stopwords-ru/master/stopwords-ru.txt', header=None)[0]
)
text_corpus = pd.concat([X_train, X_test], axis=0).reset_index().drop(['index'], axis=1, inplace=False).iloc[:, 0]
tf_idf = TfidfVectorizer(stop_words=russian_stop_words,
                             ngram_range=(1, 2),
                             min_df=5,
                             max_df=0.99
                            )


# In[38]:


get_ipython().run_cell_magic('time', '', 'tf_idf.fit([\' \'.join(str(word) for word in sentence) for sentence in train_lemmatized_descriptions])\nX_tr = tf_idf.transform(\n    [\' \'.join(str(word) for word in sentence) for sentence in train_lemmatized_descriptions]\n)\nX_te = tf_idf.transform(\n    [\' \'.join(str(word) for word in sentence) for sentence in test_lemmatized_descriptions]\n)\nprint(f"X_tr: {X_tr.shape[0]:,} x {X_tr.shape[1]:,}")\nprint(f"X_te: {X_te.shape[0]:,} x {X_te.shape[1]:,}")\n')


# In[39]:


# Creating Word2Vec vocabulary, based on our lemmatized X_train
text_corpus = train_lemmatized_descriptions
d = 50 # embedding dimension
w2v_model = Word2Vec(sentences=text_corpus,
                     min_count=10,
                     window=10,
                     vector_size=d)


# In[40]:


# Translate words from the dataset into word2vec vectors
copy_X_train = deepcopy(X_train)
X_train = pd.DataFrame(train_lemmatized_descriptions).fillna(value=0, inplace=False)
X_train = X_train.applymap(lambda x: w2v_model.wv[x] if x in w2v_model.wv else 0)


# In[43]:


X_train.head(2)


# In[ ]:


In every cell we have 50-dimensional vector of an embedded word. Let's check: the first vector in the first line should be
the word 'категорически'


# In[45]:


(X_train.iloc[0, 0] == w2v_model.wv['категорически']).all()


# In[ ]:


Good, everything is allright
Lets take embedding of a sentnence as weighted mean of every word in the sentence.
I am going to use idf of a word divided by sum of idf's in this sentence. It allows us to give more weight to the rarest words.
They are what makes each review unique and defines it.


# In[46]:


# Create mapping: (token, idf вес)
word2idf = {}
mean_idf_weight, n = 0, 0
for word, idx in tqdm(tf_idf.vocabulary_.items()):
    word2idf[word] = tf_idf.idf_[idx]
    mean_idf_weight += tf_idf.idf_[idx]
    n += 1
mean_idf_weight /= n

X_tr_emb = []
emb_size = len(w2v_model.wv['не'])
for text in tqdm(train_lemmatized_descriptions):
    res = np.zeros(emb_size)
    denominator = 1e-20
    for token in text:
        idf_weight = word2idf.get(token, mean_idf_weight)
        denominator += idf_weight
        try:
            res += w2v_model.wv[token] * idf_weight
        except:
            res += np.zeros(emb_size)
    res /= denominator
    X_tr_emb.append(list(res))
X_tr_emb = np.array(X_tr_emb)


# In[31]:


X_te_emb = []
emb_size = len(w2v_model.wv['не'])
for text in tqdm(test_lemmatized_descriptions):
    res = np.zeros(emb_size)
    denominator = 1e-20
    for token in text:
        idf_weight = word2idf.get(token, mean_idf_weight)
        denominator += idf_weight
        try:
            res += w2v_model.wv[token] * idf_weight
        except:
            res += np.zeros(emb_size)
    res /= denominator
    X_te_emb.append(list(res))
X_te_emb = np.array(X_te_emb)


# In[50]:


print(X_tr_emb[0, :])
print(np.shape(X_tr_emb))


# In[ ]:


Here we got the embedding for the first review.
Nice, now we have fully embedded dataset, we can run the ml model. We are going to use logistic regression as we need to get
the best score according to ROS-AUC score and we believe that it describes our data structure rather well. Before training we
need to optimize the hyperparameter C and l1_ratio through cross-validation.


# In[52]:


# define search space
search_space = list()
search_space.append(Real(0.1, 10, name='C'))
search_space.append(Real(10**(-3), 1, name='l1_ratio'))


# define the function needed to optimize
@use_named_args(search_space)
def cross_val_mean(**param):
    log_reg = LogisticRegression(**param, n_jobs=-1)
    acc = -np.mean(cross_val_score(estimator=log_reg, X=X_tr, y=y_train, scoring=make_scorer(roc_auc_score), cv=5))
    return acc


# perform optimization
result = gp_minimize(
    func=cross_val_mean,
    dimensions=search_space,
    n_calls=40,
    random_state=42,
    verbose=True
)

print('Best AUC-ROC: %.3f' % (-result.fun))
print('Best Parameters: %s' % (result.x))


# In[ ]:


We got our best hyperparameters, now we can fit the model and give predictions for test dataset.


# In[53]:


# Fit our best model on the whole train dataframe
C, l1_ratio = result.x
log_reg = LogisticRegression(C=C, l1_ratio=l1_ratio, n_jobs=-1)

log_reg.fit(X=X_tr_emb, y=y_train)
pred_test = log_reg.predict_proba(X=X_te_emb)[:, 1]
submission = pd.DataFrame()
submission["id"] = test_ids
submission["sentiment"] = pred_test
submission.to_csv("submission.csv", index=False)


# In[ ]:





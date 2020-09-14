
# word embeddings are the vector representations of a word in the context.
# similar words have similar embeddings, in the sence that they have similar cosine score 
#

# %%
import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import re

twitter_df = pd.read_csv('twitter_train.csv')
twitter_df = twitter_df.fillna('0')

twitter_df_test = pd.read_csv('twitter_test.csv')
twitter_df_test = twitter_df_test.fillna('0')

twitter_df = twitter_df.drop('location', axis=1)
target = twitter_df.target
# twitter_df = twitter_df.drop('target', axis=1)

twitter_df_test = twitter_df_test.drop('location', axis=1)
# target = twitter_df_test.target

# CBOW is a word embedding technique to predict middle word in the context.
# hence capturing some semantic information. 
#%%
from nltk import TweetTokenizer
wt = TweetTokenizer()
stop_words = nltk.corpus.stopwords.words('english')


def normalize_corpus(df_1 = twitter_df, text_col = 'text'):
    
    # refining the text by removing the special characters,
    # lower casing all the words, removing white spaces
    
    # size less than 3
    # RNN, LSTM
    # remove all non words.
    # can remove names as they each form a different vocabulary, and 2 common names dosent mean anything.
    # stemming, lemmatization
    
    df = df_1.copy()
    df.dropna(inplace=True)
    
    url_re = r'(https?://\S+|www\.\S+)'  #r'(https?:\/\/\S+)$?'
    english_re = r'([a-zA-Z]\w+)'
    extended_stop_words_re = stop_words + ['&amp;','rt','th','co', 're','ve','kim','daca','p.m.']
    single_letters_re = r'.'
    
    df['preprocessed_'+ text_col] = df[text_col].str.lower() # lower casing the text.
    
    df['preprocessed_'+ text_col] = df['preprocessed_'+ text_col].apply(lambda row: ' '.join([word for word in row.split()
                                                                                             if (re.match(english_re, word))
                                                                                             and (not word in extended_stop_words_re)
                                                                                             and (not word in single_letters_re)]))
    
    # df['preprocessed_'+text] = re.sub(english, '', df['preprocessed_'+text])
    df['preprocessed_'+ text_col] = df['preprocessed_'+ text_col].apply(lambda row: re.sub(url_re, '', row)) # removing urls.
    
    # tokenize document
    df['tokenised_' + text_col] = df['preprocessed_'+ text_col].apply(lambda row: wt.tokenize(row))
    # df['tokenised_' + text_col].apply(re.sub(single_letters_re, '', row))
    return df

norm_df = normalize_corpus(twitter_df)
norm_df_test = normalize_corpus(twitter_df_test)

# %%
# importing the requires libraries to generate word embeddings.
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

tokenised_doc = norm_df.tokenised_text
target = norm_df.target

# convert tokenized document into gensim formatted tagged data
tagged_data = [TaggedDocument(d , [i]) for i, d in zip(target, tokenised_doc)]

# tagged_data = [TaggedDocument(d , [i]) for (i, d) in norm_df[["keyword", "tokenised_text"]]]

tokenised_doc_test = norm_df_test.tokenised_text
keywords = norm_df_test.keyword

tagged_data_test = [TaggedDocument(d, [i]) for i, d in zip(keywords, tokenised_doc_test)]

# tagged_dada_1 = norm_df.apply(lambda r: TaggedDocument(words=r['tokenised_text'], tags = r['keyword']))

# initialising doc2vec model weights
'''
model = Doc2Vec(dm = 1, documents = tagged_data, vector_size = 200, min_count = 5, epochs = 50)

# model.wv.vocab
# model.corpus_total_words

# the model weights are already initialised.
# Paragraph Vector Distributed memory acts as a memory that remembers what is missing from the 
# current context.

for epoch in range(30):
    model.train(tagged_data, total_examples = len(tagged_data), epochs=5)
    model.alpha -=0.002
    model.min_alpha = model.alpha

# save trained model
model.save('train_doc2vec.model')
'''
# load saved model
model = Doc2Vec.load("train_doc2vec.model")

# tagged_data[1000].tags[0]

vector = model.infer_vector(['leagues', 'ball', 'olympic', 'level', 'body', 'bagging', 'like', 'career', 'nothing'])

vector.shape
# %%
def vectors_Doc2vec(model, tagged_docs):
    sents = tagged_docs
    tags, vectors = zip(*[(doc.tags[0], model.infer_vector(doc.words)) for doc in sents])
    return tags, vectors

targets, vectors = vectors_Doc2vec(model, tagged_data)
targets_test, vectors_test = vectors_Doc2vec(model, tagged_data_test)

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import pickle
import csv

def use_Logistic_reg(params, use_GridSearchCV = False, C = 0.01, n_jobs = -1, cv = 5, save_model = False):
    # this function returns the optimum Logistic Regression Model according to the preferences.

    clf = LogisticRegression(C=C, n_jobs=n_jobs)
    if use_GridSearchCV:
        model = GridSearchCV(clf, param_grid=params, cv = cv)
        model.fit(vectors, target)
    else:
        model = clf.fit(vectors, target)

    filename = "Logistic_Reg_clf.pickle"
    if save_model:
        pickle.dump(model, open(filename, 'wb'))

        # load the model from the disk
        model = pickle.load(open(filename, 'rb'))

    return model

id_1 = twitter_df_test.id
def save_pred(model, id_ = id_1, name_ = "name_1.csv", vectors_ = vectors_test):
    predict = model.predict(vectors_)
    
    # checking if the predictions are correct format vector
    assert len(predict) == 3263

    # the file is saved in the current folder
    with open(name_, 'w', newline='\n') as f:
        writer = csv.writer(f)
        
        writer.writerow(['id', 'target'])
        for id_, target in zip(id_, predict):
            writer.writerow([id_, target])

# Fitting the vectors over Logistic Regression Model 
# calling the function over s
# params = {'C':[0.001, 0.01, 0.1]}
# model_lr = use_Logistic_reg(params = params, use_GridSearchCV=True)
# y_pred = model_lr.predict(vectors)

from sklearn.metrics import accuracy_score
# print('Logistic Regression Accuracy Score is : ',accuracy_score(y_pred, target))
# save_pred(model=model_lr, id_= id_1, name_="LR_twitter.csv")

from sklearn.ensemble import RandomForestClassifier

# %%

def use_Rand_forest(params, use_GridSearchCV = False, n_jobs = -1, cv = 5, save_model = False):
    # this function fits a RandomForest Model on the word vectors.

    clf = RandomForestClassifier(n_jobs=n_jobs)
    if use_GridSearchCV:
        model = GridSearchCV(clf, param_grid=parameters, cv = cv, scoring='accuracy')
        model.fit(vectors, target)
    else:
        model = clf.fit(vectors, target)

    filename = "RandomForestClf.pickle"
    if save_model:
        pickle.dump(model, open(filename, 'wb'))

        # load the model from the disk
        model = pickle.load(open(filename, 'rb'))

    return model

parameters = {'max_depth': [100, 150],    # max depth of the tree
    'max_features': [7, 9, 11],           # number of features to consider when looking for best split
    'n_estimators': [900],                # Number of trees in the forest
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [2, 5, 10],
    'criterion': ['gini', 'entropy']}     # Quality of the split

model_rf = use_Rand_forest(params=parameters, use_GridSearchCV=False)

print('the Random Forest Accuracy score is : ',accuracy_score(model_rf.predict(vectors), target))

# the test set does not containt the labels.
# save the predictions in as csv format required by the kaggle competitions.

# saving the Random Forest Model using the function save_pred
save_pred(model=model_rf, id_= id_1, name_="Rf_twitter.csv")

# the testing accuracy was 68.188%, which clearly means that the Random Forest algorithm overfits.
# %%

import catboost as cb
cat_1 = cb.CatBoostClassifier(iterations=1000, eval_metric='F1')

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(vectors, targets, stratify = targets)

model_2= cat_1.fit(X_train, y_train, eval_set=(X_test,y_test), plot=True)

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error

k = 5 # number of Kfolds

list_train_index = []
list_valid_index = []

skf = StratifiedKFold(n_splits=k, shuffle = True)
for train_index, valid_index in skf.split(X = vectors, y=targets):
    list_train_index.append(train_index)
    list_valid_index.append(valid_index)

list_train_index

cat = cb.CatBoostClassifier(iterations=300, eval_metric='F1')


for i in range(k):
    X_t = np.asarray(vectors)[list_train_index[i],:]
    X_v = np.asarray(vectors)[list_valid_index[i],:]
    
    y_t = np.asarray(targets)[list_train_index[i]]
    y_v = np.asarray(targets)[list_valid_index[i]]
    
    cat.fit(X = X_t, y = y_t, eval_set=(X_v, y_v));
    print(accuracy_score(y_t, cat.predict(X_t)))

pred_cat = cat.predict(np.asarray(vectors_test))

# %%

# running the XGBoost classifier on data
from xgboost import XGBClassifier

def use_XGBoost(params, use_GridSearchCV = False, C = 0.01, n_jobs = -1, cv = 5, save_model = False):
    # this function fits a RandomForest Model on the word vectors.

    clf = XGBClassifier(nthread = -1)
    if use_GridSearchCV:
        model = GridSearchCV(clf, param_grid=params, cv = cv, scoring='accuracy')
        model.fit(np.array(vectors), np.array(target))
    else:
        model = clf.fit(np.array(vectors), np.array(target))

    filename = "XGB.pickle"
    if save_model:
        pickle.dump(model, open(filename, 'wb'))

        # load the model from the disk
        model = pickle.load(open(filename, 'rb'))

    return model

params_xgb = {'n_estimators' : [4, 10, 20, 50, 100, 200],
               'gamma':np.linspace(.01, 1, 10, endpoint=True), 
               'learning_rate' : np.linspace(.01, 1, 10, endpoint=True),
               'reg_lambda': np.linspace(0.01, 10, 20, endpoint=True),
               'max_depth' : np.linspace(1, 32, 32, endpoint=True, dtype=int)
                 }

model_xg = use_XGBoost(params_xgb, use_GridSearchCV=True, save_model=True)
print('the Accuracy score using XGBoost is : ',accuracy_score(model_xg.predict(np.array(vectors)), np.array(target)))

save_pred(model=model_xg, id_= id_1, name_="xgb_twitter.csv", vectors_=np.array(vectors_test))
# %%


# implementing RNN and LSTM
# %%
import pandas as pd
import numpy as np
import nltk
import sklearn
import matplotlib.pyplot as plt
import re
import tqdm

twitter_df = pd.read_csv('twitter_train.csv')
twitter_df = twitter_df.fillna('0')

twitter_df_test = pd.read_csv('twitter_test.csv')
twitter_df_test = twitter_df_test.fillna('0')

twitter_df = twitter_df.drop('location', axis = 1)

import json
"""with open('contractions.json', "w") as f:
    json.dump(contractions, f)"""
with open('abbrevations.json') as f:
    abbrevation = json.load(f)

# importing required libraries for RNN

import tensorflow as tf;
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
# nltk.download('wordnet')

print("Tensorflow version", tf.__version__)

# preprocessing include lemmatizing, stemming, stopwords removal
# tokenizing, pad_sequences
# %%
lemmatizer = WordNetLemmatizer()
stop_words = stopwords.words('english')
stemmer = PorterStemmer()
from nltk import TweetTokenizer
wt = TweetTokenizer()

# Cleaning means to extract the useful information from the text data and removing those
# data that does not contribute to the LSTM and RNN learnings.

# The clean_text function lower cases the input text, removes the tags and mentions
# expands the contractions, can deal with emojis, non alphabets.
# It also removes the stop words and Lemmatizes the word to its root word.

def text_clean(text, abbrevations=abbrevation, stemmer=False):
    
    # lower casing
    text = text.lower()
    
    for word in text.split():                               # use the abbrevations dictionary to replace
        if word in abbrevations.keys():
            text = text.replace(word, abbrevations[word])
    
    
    # removing URL, tags, non-alphabets, new-lines
    text = re.sub(r'(https?://\S+|www\.\S+)', '', text)     # removing http links
    text = re.sub(r'@([a-zA-Z0-9:_]+)\s', '', text)         # removing the hash tags and mentions
    
    text = re.sub('[^\w\s]', ' ', text)                     # remove anything except words and spaces.
    text = re.sub('\d', ' ', text)                          # removing digits
    # text = re.sub('\b[a-z]{1,2}\b', ' ', text)              # all words with 3 or less characters
    text = re.sub('\n', ' ', text)                          # new line phrase
    
    # Lemmatising and removing Stop_words
    
    extended_stop_words_re = stop_words + ['&amp;','rt','th','co', 're','ve','kim','daca','p.m.','retweet', 'ir']
    if not stemmer:
        text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if (not word in extended_stop_words_re)])
    else:
        text = ' '.join([stemmer.stem(word) for word in text.split() if (not word in extended_stop_words_re)])

    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    
    text = emoji_pattern.sub(r'', text)

    return text


# further cleaning is needed.
# tekenization of the cleaned text.

cleaned1 = lambda text: text_clean(text)
cleaned2 = lambda text: re.sub('\s[a-z]{,3}\s', ' ', text)
cleaned3 = lambda row: wt.tokenize(row)                         # tokenizing the row

# escented characters
# expanding contractions with the words in the contractions dictionary
# stemming, lemmatizing
# negated words can be used to get the sentiment
# extra spaces shouls also be removed
# %%
# Cleaning the input text with help of utility functions
twitter_df['cleaned_text'] = twitter_df['text'].apply(cleaned1)
twitter_df['cleaned_text_'] = twitter_df['cleaned_text'].apply(cleaned2)
twitter_df['tokenized_text'] = twitter_df['cleaned_text_'].apply(cleaned3)
twitter_df_test['cleaned_text'] = twitter_df_test['text'].apply(cleaned1)
twitter_df_test['cleaned_text_'] = twitter_df_test['cleaned_text'].apply(cleaned2)
twitter_df_test['tokenized_text'] = twitter_df_test['cleaned_text_'].apply(cleaned3)

# twitter_df[['id','text','cleaned_text','target']].iloc[50:110]

length = []
length = twitter_df.cleaned_text.apply(lambda x: len(x.split()))
np.max(length)


train_text = twitter_df.cleaned_text
train_label = twitter_df.target

test_text = twitter_df_test.cleaned_text

oov_tok = '<oov>'
padding_type = 'post'
trun_type = 'post'
max_length = 25


# %%

tokenizer = Tokenizer(num_words = 10000, oov_token = oov_tok)
tokenizer.fit_on_texts(train_text)

word_index = tokenizer.word_index

train_sequences = tokenizer.texts_to_sequences(train_text)
print("train sequences sample", train_sequences[10])

test_sequences = tokenizer.texts_to_sequences(test_text)
print("test sequences sample", test_sequences[1])

train_padded = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trun_type)
test_padded = pad_sequences(test_sequences, maxlen=max_length, padding=padding_type, truncating=trun_type)

vocab_size = len(word_index) + 1
#
label_tokenizer = Tokenizer()
# %%

from gensim.models.word2vec import Word2Vec


embedding_index = {}

# for text in twitter_df:
#     embedding = model.infer_vector(text)



# %%
# utility function to return the string of a list of numbers.
# this will be the input to the tokenizer methods
def return_str(text):
    return text.apply(lambda x: str(x))

label_tokenizer.fit_on_texts(return_str(twitter_df.target))

train_label_seq = np.array(label_tokenizer.texts_to_sequences(return_str(train_label)))

print(train_label_seq.shape)

train_label_seq = train_label.values.reshape(-1,1)

# trying to explore the original tweet and tweet after padding
reverse_word_index = dict([(index, word) for word, index in word_index.items()])

def decode_article(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

print(decode_article(train_padded[10]))
print('-----------------')
print(train_text.iloc[10])

# %%

from gensim.models.word2vec import Word2Vec
vector_size = 100

wv = Word2Vec(sentences=twitter_df.tokenized_text,size = vector_size, iter = 50)

from collections import Counter


embedding_matrix = np.zeros((vocab_size, vector_size))
# word_index.pop("''")
for word, index in word_index.items():
    try:
        embedding_vector = wv[word]
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector
    except:
        pass

# %%

embedding_dim = 100

# building the Model Architecture
from tensorflow.keras import layers # importing Dense, Bidirectional, LSTM, Dropout, Embedding
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, SGD, Nadam

# Word Embeddings are the inputs to the neural networks.
def create_embeddings(vocab_size, embedding_dim,):
    return layers.Embedding(vocab_size, embedding_dim, embeddings_initializer='GlorotNormal', weights = [embedding_matrix])

# Utility function to get fully connected dense Layers
def create_dense(in_dim = embedding_dim, activ_in = 'relu'):
    return layers.Dense(in_dim, activation = activ_in, use_bias = True)

# Create LSTM layers
def get_LSTM(embedding_dim, Bi_directional = True, dropout=0.2):
    if Bi_directional:
        layer = layers.Bidirectional(layers.LSTM(embedding_dim, recurrent_dropout = 0.3, dropout=dropout))
    else:
        layer = LSTM(embedding_dim, recurrent_dropout = 0.3, dropout=dropout)
    return layer

# Dropout can be represented as one of the core layers.
# They handle overfitting of the neural networks by allowing all the nodes to learn the weights 
# Lot of fine tuning can be done to the Dropout layer.
def drop_out(dropout_rate = 0.2):
    dp = layers.Dropout(dropout_rate)
    return dp

def get_seq_model(in_dim = embedding_dim, 
                    out_class=2,
                    optimizers_ = 'sgd', 
                    learning_rate = 0.000083,
                    activ_out = 'softmax'):
    
    # frees up GPU memory everytime the code is run fresh
    tf.keras.backend.clear_session()

    model = tf.keras.Sequential([
        
        # adding embedding layer, expecting input vocab size of 5000
        create_embeddings(vocab_size, embedding_dim),
        
        # adding Dropout layer
        drop_out(0.3),

        # Bi-Directional LSTM
        get_LSTM(embedding_dim, dropout = 0.3),

        # Fully connected dense layers
        create_dense(embedding_dim),
        
        # adding Dropout layer
        drop_out(0.3),

        # Fully connected dense layers
        create_dense(embedding_dim),
        
        # adding Dropout layer
        drop_out(0.3),
        
        # the final output layer
        layers.Dense(out_class, activation = 'softmax', use_bias = True)
        ])

    # Model fitting and defining callbacks
    if optimizers_.lower() == "adam":
        opt = Adam(lr=learning_rate)
    elif optimizers_.lower() == "sgd":
        opt = SGD(lr=learning_rate)
    else:
        opt = Nadam(lr=learning_rate)

    model.compile(loss='sparse_categorical_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])

    return model
# %%
# creating the model
num_epochs=30
model = get_seq_model(learning_rate=0.01)

# printing model summary to console
print(model.summary())

# callbacks stop the traiing after predefined patience
early_stopping = EarlyStopping(monitor='val_accuracy', patience=2)

# training the model
result = model.fit(train_padded, train_label_seq, 
                        epochs=num_epochs, 
                        # callbacks = [early_stopping],
                        verbose=2, validation_split=0.2
                        )

def plot_graphs(history, string, save = False):
    plt.plot(result.history[string])
    plt.plot(result.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    title = "Bi-LSTM " + string
    if save:
        plt.savefig(title)
    plt.show()

plot_graphs(result, "accuracy")
plot_graphs(result, "loss")

# %%
import csv
id_1 = twitter_df_test.id 

def save_pred(model, id_ = id_1, name_ = "name_1.csv", vectors_ = test_padded):
    predict = model.predict_classes(vectors_)
    
    # checking if the predictions are correct format vector
    assert len(predict) == 3263

    # the file is saved in the current folder
    with open(name_, 'w', newline='\n') as f:
        writer = csv.writer(f)
        
        writer.writerow(['id', 'target'])
        for id_, target in zip(id_, predict):
            writer.writerow([id_, target])

  
save_pred(model, id_=id_1,name_='rnn_bilstm_1.csv')
# %%

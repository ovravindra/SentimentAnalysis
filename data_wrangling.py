# AIM : to construct a modular code blocks to perform standard text wrangling operations and also to 
# produce handy plots and other functionality.

# we can borrow the word vectors generated using the CountVectoriser and TfidfVectoriser to
# train the binary classification Logistic Regression model and check the accuracy.
# starting the Data wrangling using pandas and numpy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style = 'whitegrid', color_codes = True)
import os
import re

# loading the text data
twitter_df = pd.read_csv("twitter_train.csv")

twitter_df.head(50)

"""
input variables
1. id : is the ID of the tweet
2. Keyword : is the main word that is used to extract the tweet. both the positive and negative sentiments revolve around this words
3. Location: is the geo-location of the person tweeting.
4. Text: is the main body of the tweet. Any tweet can have only 280 characters. i.e., for example any one of the following list is a character :- [a,1,@,A... etc]
5. Target: finally the sentiment of the tweet, manually added. This makes the data labeled data, and hence, the problem a classification problem.
"""
twitter_df.info()

twitter_df.shape

twitter_df = twitter_df.dropna(how = 'all', axis = 0)

twitter_df.shape

twitter_df = twitter_df.fillna('0')

print(twitter_df['target'].value_counts())
sns.countplot(x = 'target', data = twitter_df, palette='hls')

df = twitter_df.groupby(['keyword', 'target'])['text'].count().reset_index()

"""
further implementations to be done
1. number of characters in a tweet.
"""
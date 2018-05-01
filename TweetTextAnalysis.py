
# coding: utf-8

# In[1]:


from textblob import TextBlob
import nltk
from nltk.corpus import stopwords


# In[2]:


import pandas as pd
import numpy as np


# In[3]:


tweet_text = pd.read_csv(r'./tweetsText.csv')


# In[4]:


tweet_text.columns


# In[5]:


tweet_text.head(10).text


# ### Tweet Language:

# TextBlob will determine the language of text, but requires that the analyzed text be at least 3 characters. For example, tweet below is causing an error.

# In[65]:


len(tweet_text.iloc[756,1])


# In[69]:


tweet_text.head(31).apply(lambda x: TextBlob(x['text']).detect_language(),axis=1)


# In[68]:


tweet_text.iloc[30]


# defining a function to preserve the short tweets, and avoid the error due to string length.

# In[75]:


def getLang(text_sample):
    if len(text_sample) < 3:
        return np.nan
    else:
        return TextBlob(text_sample).detect_language()


# There seems to be a timeout issue when processing large amounts of tweets. May be caused by API limits? Testing with increasing numbers here.

# In[102]:


tweet_lang = tweet_text[:1000].apply(lambda x: getLang(x['text']),axis=1)


# Language processing seems to be inconsistent.

# In[103]:


tweet_lang.groupby(tweet_lang).count()


# In[98]:


tweet_words = tweet_text.text.str.lower().str.split(r'\s+',expand=True).stack().value_counts()


# In[6]:


stop_words = set(stopwords.words('english')) | set(stopwords.words('spanish'))


# In[8]:


stop_list = list(stop_words)


# In[99]:


tweet_words[tweet_words.index.str.len() > 3][:200]


# In[31]:


tweet_words[~(tweet_words.index.isin(stop_list))].head(20)


# In[34]:


tweet_words[~(tweet_words.index.isin(stop_list)) & (tweet_words.index.str.len() > 3)].head(20)


# #### Twitter Sentiment Analysis Testing

# In[10]:


tweet_text.head(10).apply(lambda x: TextBlob(x['text']).sentiment.polarity,axis=1)


# In[11]:


tweet_text.head(10).apply(lambda x: TextBlob(x['text']).sentiment.subjectivity,axis=1)


# Experimenting with a large sentiment analysis dataset. Attempting to use the Twitter Sentiment Analysis Dataset Corpus obtained from http://thinknook.com/twitter-sentiment-analysis-training-corpus-dataset-2012-09-22/

# Twitter Corpus has some extraneous quotation marks that affect parsing.

# In[12]:


# import re

# new_file = []

# re_string = '^(\d+,\d+,\w+,)(.+)$'
# g = open('.\sentiment_corrected.csv','w')
# g.seek(0)
# with open('.\Sentiment Analysis Dataset.csv','r') as f:
#     lines = f.readlines()
       
# for line in lines:
#     line = line.replace('"',"'")
#     line = re.sub(re_string,r'\1"\2"',line)
#     g.writelines(line)

# f.close()
# g.close()


# In[13]:


twitter_corpus = pd.read_csv(r'./sentiment_corrected.csv')


# In[14]:


twitter_corpus.head()


# Corpus text is in alphabetical order. Using this to experiment with 60/20/20 split for train/test/val

# In[15]:


train_sample = np.split(twitter_corpus.sample(frac=1),[int(.6*len(twitter_corpus)),int(.8*len(twitter_corpus))])


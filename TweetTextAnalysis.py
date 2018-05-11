
# coding: utf-8

# In[2]:


from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer


# In[3]:


import pandas as pd
import numpy as np


# In[4]:


tweet_text = pd.read_csv(r'./tweetsText.csv')


# In[5]:


tweet_text.columns


# In[6]:


tweet_text.head(10).text


# In[13]:


tweet_text.info()


# ### Working with TfIdf - Term frequency/Inverse document frequency

# In[8]:


stopWords = set(stopwords.words('english')) | set(stopwords.words('spanish'))


# In[9]:


tweet_vector = TfidfVectorizer(analyzer='word',stop_words=stopWords).fit_transform(tweet_text['text'])


# In[10]:


tweet_vector


# In[11]:


tweet_vector[0:10000].toarray()


# In[14]:


from sklearn.metrics.pairwise import linear_kernel
cosine_similarities = linear_kernel(tweet_vector[0:5], tweet_vector).flatten()


# In[15]:


cosine_similarities.argsort()


# ### Tweet Language:

# TextBlob will determine the language of text, but requires that the analyzed text be at least 3 characters. For example, tweet below is causing an error.

# In[16]:


len(tweet_text.iloc[756,1])


# In[17]:


tweet_text.head(31).apply(lambda x: TextBlob(x['text']).detect_language(),axis=1)


# In[18]:


tweet_text.iloc[30]


# defining a function to preserve the short tweets, and avoid the error due to string length.

# In[19]:


def getLang(text_sample):
    if len(text_sample) < 3:
        return np.nan
    else:
        return TextBlob(text_sample).detect_language()


# There seems to be a timeout issue when processing large amounts of tweets. May be caused by API limits? Testing with increasing numbers here.

# In[ ]:


tweet_text['Lang'] = tweet_text[:10].apply(lambda x: getLang(x['text']),axis=1)


# Trying to circumvent the API limitations with an iterator. (Using the `stop_point` variable to check on progress, and start over later.)

# In[95]:


tweet_text.to_csv(r'./tweets_with_lang.csv')


# In[ ]:


stop_point = 36605
for i in range(stop_point,tweet_text.shape[0]):
    tweet_text.iloc[i,2] = getLang(tweet_text.iloc[i,1])


# In[96]:


tweet_text[tweet_text['Lang'].notnull()].iloc[-1]


# In[94]:


tweet_text.iloc[36605]


# In[91]:


tweet_text[tweet_text['Lang'].notnull()]['Lang'].groupby(tweet_text['Lang']).count()


# Language processing seems to be inconsistent.

# In[ ]:


tweet_lang.groupby(tweet_lang).count()


# In[ ]:


tweet_words = tweet_text.text.str.lower().str.split(r'\s+',expand=True).stack().value_counts()


# In[ ]:


stop_words = set(stopwords.words('english')) | set(stopwords.words('spanish'))


# In[ ]:


stop_list = list(stop_words)


# In[ ]:


tweet_words[tweet_words.index.str.len() > 3][:200]


# In[ ]:


tweet_words[~(tweet_words.index.isin(stop_list))].head(20)


# In[ ]:


tweet_words[~(tweet_words.index.isin(stop_list)) & (tweet_words.index.str.len() > 3)].head(20)


# #### Twitter Sentiment Analysis Testing

# In[ ]:


tweet_text.head(10).apply(lambda x: TextBlob(x['text']).sentiment.polarity,axis=1)


# In[ ]:


tweet_text.head(10).apply(lambda x: TextBlob(x['text']).sentiment.subjectivity,axis=1)


# Experimenting with a large sentiment analysis dataset. Attempting to use the Twitter Sentiment Analysis Dataset Corpus obtained from http://thinknook.com/twitter-sentiment-analysis-training-corpus-dataset-2012-09-22/

# Twitter Corpus has some extraneous quotation marks that affect parsing.

# In[ ]:


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


# In[ ]:


twitter_corpus = pd.read_csv(r'./sentiment_corrected.csv')


# In[ ]:


twitter_corpus.head()


# Corpus text is in alphabetical order. Using this to experiment with 60/20/20 split for train/test/val

# In[ ]:


train_sample = np.split(twitter_corpus.sample(frac=1),[int(.6*len(twitter_corpus)),int(.8*len(twitter_corpus))])


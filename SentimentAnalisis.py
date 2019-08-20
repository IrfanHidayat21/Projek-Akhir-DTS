# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 08:20:58 2019

@author: ROG-GL553VD
"""
import tweepy
import re
import seaborn as sns
import numpy as np
import pandas as pd
import json
from googletrans import Translator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from collections import Counter

analyser = SentimentIntensityAnalyzer()
translator = Translator()

ACCESS_TOKEN = '1152949123659624448-xSTKfdNXyGUSUKIxa64GNicBaMtiAH'
ACCESS_SECRET = 'zmxMMn9WU2HV5fYb6EkLW09YNSsIE3GAp4fuKCbGrEswV'
CONSUMER_KEY = 'jADe8twJaq509mPHMl6S0Ouav'
CONSUMER_SECRET = 'sSvraQz68poyejc18MYfZ6ElEPCqQo9wdKt9GiR2aAD7B1e6Mj'

auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)
api = tweepy.API(auth)


def list_tweets(user_id, count, prt=False):
    tweets = api.user_timeline(
        "@" + user_id, count=count, tweet_mode='extended')
    tw = []
    for t in tweets:
        tw.append(t.full_text)
        if prt:
            print(t.full_text)
            print()
    return tw

def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)        
    return input_txt

def _removeNonAscii(s): 
    return "".join(i for i in s if ord(i)<128)

def clean_text(text):
    text = re.sub(r"\n", " ", text)
    # remove ascii
    text = _removeNonAscii(text)
    # to lowecase
    text = text.lower()
    return text

def clean_lst(lst):
    lst_new = []
    for r in lst:
        lst_new.append(clean_text(r))
    return lst_new

def clean_tweets(lst):    
    
    # remove twitter Return handles (RT @xxx:)
    lst = np.vectorize(remove_pattern)(lst, "rt @[\w]*:")
    # remove twitter handles (@xxx)
    lst = np.vectorize(remove_pattern)(lst, "@[\w]*")
    # remove URL links (httpxxx)
    lst = np.vectorize(remove_pattern)(lst, "https?://[A-Za-z0-9./]*")
    # remove punctuation 
    lst = np.core.defchararray.replace(lst, "[^\w\s]+", "")
    # remove special characters, numbers, punctuations (except for #)
    lst = np.core.defchararray.replace(lst, "[^a-zA-Z#]", " ")
    return lst

def sentiment_analyzer_scores(text,eng=False):
    translator = Translator()
    if eng:
        try:
            text = translator.translate(text).text
        except Exception as e:
            print(str(e))
            
    score = analyser.polarity_scores(text)
    lb = score['compound']
    if lb >= 0.05:
        return 1
    elif (lb > -0.05) and (lb < 0.05):
        return 0
    else:
        return -1

def anl_tweets(lst, title='Tweets Sentiment', engl=False ):
    sents = []
    for tw in lst:
        try:
            st = sentiment_analyzer_scores(tw, engl)
            sents.append(st)
        except:
            sents.append(0)
    ax = sns.distplot(
        sents,
        kde=False,
        bins=3)
    ax.set(xlabel='Negative                Neutral                 Positive',
           ylabel='#Tweets',
          title="Tweets of @"+title)
    return sents

user_id = 'detikcom'
count = 200

dt_dirgahayu = {"raw": pd.Series(list_tweets(user_id, count, True))}
tw_dirgahayu = pd.DataFrame(dt_dirgahayu)
tw_dirgahayu['raw'][3]

tw_dirgahayu['clean_text'] = clean_lst(tw_dirgahayu['raw'])
tw_dirgahayu['clean_text'][1]

tw_dirgahayu['clean_vector'] = clean_tweets(tw_dirgahayu['clean_text'])
tw_dirgahayu['clean_vector'][1]

sentiment_analyzer_scores(tw_dirgahayu['clean_text'][3],True)

tw_dirgahayu['sentiment'] = pd.Series(anl_tweets(tw_dirgahayu['clean_vector'], user_id, True))

#Word Cloud + Sentiment Analysis
stop_words = []
f = open('data\stopwords.txt', 'r')
for l in f.readlines():
    stop_words.append(l.replace('\n', ''))

f = open('data\stopwords_indonesia.txt', 'r')
for l in f.readlines():
    stop_words.append(l.replace('\n', ''))
    
additional_stop_words = ['t', 'will']
stop_words += additional_stop_words

def word_cloud(wd_list):
    from skimage import io

    coloring = io.imread('data/hut742.png')
    stopwords = stop_words + list(STOPWORDS)
    all_words = ' '.join([text for text in wd_list])
    wcloud6 = WordCloud(
        background_color='white',
        stopwords=stopwords,
        width=200,
        height=200,
        random_state=21,
        max_words=200,
        max_font_size=150,
        font_path='data/Atma-Bold.ttf', 
        mask=coloring).generate(all_words)
    
    
    plt.figure(figsize=(10, 9))
    plt.axis('off')
    plt.imshow(wcloud6, interpolation="bilinear");

word_cloud(tw_dirgahayu['clean_vector'])
word_cloud(tw_dirgahayu['clean_vector'][tw_dirgahayu['sentiment'] == 1])
word_cloud(tw_dirgahayu['clean_vector'][tw_dirgahayu['sentiment'] == -1])

from flask import Flask, render_template
from nltk.tokenize import word_tokenize
import json

app = Flask(__name__)

@app.route('/word_cloud', methods=['GET'])
def word_cloud():
    try:    
        words = []
        for tweet in tw_dirgahayu['clean_vector']:
          tokens = word_tokenize(tweet)
          words.extend(tokens)
        
        words = [word for word in words if word not in stop_words and len(word)>2]
        words_freq = Counter(words)
        words_json = [{'text': word, 'weight': count} for word, count in words_freq.items()]
        return json.dumps(words_json)
    except Exception as e:
        return '[]'
    
@app.route('/sentiment_chart', methods=['GET'])
def sentiment_chart():
    sentimen_freq = Counter(tw_dirgahayu['sentiment'])
    sentimen_freq['negative'] = sentimen_freq[-1]
    del sentimen_freq[-1]
    sentimen_freq['netral'] = sentimen_freq[0]
    del sentimen_freq[0]
    sentimen_freq['positive'] = sentimen_freq[1]
    del sentimen_freq[1]
    sentiment_json = {'result':[{'label': sentiment, 'weight': count} for sentiment, count in sentimen_freq.items()]}
    return json.dumps(sentiment_json)
    

@app.route('/')
def home_page():
    return render_template('index.html')
if __name__ == '__main__':
    app.run(port=8080)
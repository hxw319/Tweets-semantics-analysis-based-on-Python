#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 18:28:40 2018

@author: luxie
"""

import tweepy,re

consumer_key = 'WWqjiBIzzakgnkthoYaTqKZDo'
consumer_secret = 'scCnmwC4fLVy8lbR4A5EvnYAd9TQHGKs5312b0tUK68olbwuux'
access_token = '783392289485795328-77Ffpy21yNZGR5iFJtu0byb6VAdivXL'
access_token_secret = 'JLAaS5QMh7uT9uMSAUVfNv4JqXwQLTxqUW6xs5Oh4aD6z'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret) 
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

# emoji_pattern = re.compile(
#     u"(\ud83d[\ude00-\ude4f])|"  # emoticons
#     u"(\ud83c[\udf00-\uffff])|"  # symbols & pictographs (1 of 2)
#     u"(\ud83d[\u0000-\uddff])|"  # symbols & pictographs (2 of 2)
#     u"(\ud83d[\ude80-\udeff])|"  # transport & map symbols
#     u"(\ud83c[\udde0-\uddff])"  # flags (iOS)
#     "+", flags=re.UNICODE)

def filter_emoji(desstr,restr=''):
    '''
    过滤表情
    '''
    try:
        co = re.compile(u'[\U00010000-\U0010ffff]')
    except re.error:
        co = re.compile(u'[\uD800-\uDBFF][\uDC00-\uDFFF]')
    return co.sub(restr, desstr)



username = 'CongressmanRaja'
tweetCount = 60
results = api.user_timeline(id = username, count = tweetCount, tweet_mode='extended')
num = 1
for tweet in results:
    tweet.full_text=tweet.full_text.strip()
    link_index=tweet.full_text.find('https')
    tweet.full_text=tweet.full_text[:link_index].strip()
    tweet.full_text=re.sub(r'[#@]\w*\s?','',tweet.full_text).strip()
    tweet.full_text=re.sub(r'&amp','',tweet.full_text).strip()
    tweet.full_text=re.sub(r'RT : : |RT : |\(\)','',tweet.full_text).strip()
    tweet.full_text=filter_emoji(tweet.full_text)
    # tweet.full_text = re.sub(ur'(\.\’s)\s?','',tweet.full_text).strip()
    # tweet.full_text = re.sub(ur'\:\s','',tweet.full_text)
    print str(num)+'. '+tweet.full_text
    num += 1
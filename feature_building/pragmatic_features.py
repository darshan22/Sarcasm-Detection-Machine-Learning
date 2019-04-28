"""
This script defines the functions to extract the pragmatic features from a tweet.
"""

import codecs
import numpy as np
import re
import emoji
from nltk.tokenize import word_tokenize

#open the dataset as a file object
sarcasm_file = codecs.open('../Data/sarcasm_clean.txt', encoding = 'utf-8')


def count_emoji(tweet):
    #regex pattern for capturing the emoji
    emoji_regex = re.compile(r'[\U0001f600-\U0001f650]')
    return len(emoji_regex.findall(tweet))

def count_capitalized_words(tweet_tokens):
    count = 0
    for token in tweet_tokens:
        if token.isupper and len(token)>1:
            count += 1
    return count

def count_user_mentions(tweet):
    #regex pattern for finding user_mentions
    user_mentions_regex = re.compile(r'@\w+\s?|NAME')
    return len(user_mentions_regex.findall(tweet))

def count_hashtags(tweet):
    #regex pattern for finding hashtags
    hashtags_regex = re.compile(r'#\w+\s?')
    return len(hashtags_regex.findall(tweet))

def count_slang_laughterexp(tweet):
    pat1=re.compile(r'(\blols?z?o?\b)+?',re.I)
    pat2=re.compile(r'(\brofl\b)+?',re.I)
    pat3=re.compile(r'(\blmao\b)+?',re.I)
    return (len(pat1.findall(tweet)) + len(pat2.findall(tweet)) + len(pat3.findall(tweet)))

def count_punctuation(tweet):
    count = tweet.count('!')+tweet.count('?')
    for x in range(len(tweet)-2):
        if tweet[x]=='.' and tweet[x+2]=='.':
            count += 1
    return count
    

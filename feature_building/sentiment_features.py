"""The function defined below generates the sentiment based features for a given tweet.
It generates features such as sentiment incongruities, count of positive, neutral, and 
negative words, lexical polarity of the tweet, largest positive/negative sub-sequence.
"""

import codecs
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np

def sentiment_based_features(tweet):
    #split the tweet into tokens and initialize the count to zero.
    tweet_tokens = tweet.split()
    polarity_scores = np.zeros(len(tweet_tokens))
    positive_words = 0
    negative_words = 0
    neutral_words = 0
    polarity_flip = 0
    overall_polarity = 0
    #create an object of sentimentIntensityAnalyzer class.
    analyzer = SentimentIntensityAnalyzer()
    
    #for loop to calculate the polarity of each word in the tweet.
    for index, token in enumerate(tweet_tokens):	
        polarity = analyzer.polarity_scores(token)
        if polarity['compound']>0.0:
            positive_words += 1
            polarity_scores[index] = 1
        elif polarity['compound']<0.0:
            negative_words += 1
            polarity_scores[index] = -1
        else:
            neutral_words += 1
    
    #find the flip in polarities.
    for i in range(len(polarity_scores)-1):
        if polarity_scores[i]==1 and polarity_scores[i+1]==-1:
            polarity_flip += 1
        elif polarity_scores[i]==-1 and polarity_scores[i+1]==1:
            polarity_flip += 1
    
    sentence_polarity = analyzer.polarity_scores(tweet)
        
    return positive_words, negative_words, neutral_words, round(sentence_polarity['compound'],2), polarity_flip
    

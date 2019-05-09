"""
This script generate all the lexical, pragmatic and sentiment-based features for 
sarcasm and non-sarcasm dataset.It then creates the labels and combines them to form
one csv file which will then be further used as a dataset for model-building.
"""

import sentiment_features as sf
import pragmatic_features as pf
import codecs
import pandas as pd
import numpy as np
import nltk

#read the files
sarcasm_file = codecs.open('../Data/sarcasm_clean.txt', encoding='utf-8')
nonsarcasm_file = codecs.open('../Data/nonsarcasm_clean.txt', encoding='utf-8')

#initialize lists to store the generated features for sarcastic tweets.
emoji_count = []
capitalized_words = []
user_mentions = []
number_hashtags = []
number_slang_exp = []
number_punctuations = []
pos_words = []
neg_words = []
neutral_words = []
tweet_polarity = []
polarity_flip = []

#generating features for sarcastic tweets
print("Generating features for sarcastic tweets")

for tweet in sarcasm_file:
    tokens = nltk.word_tokenize(tweet)
    emoji_count.append(pf.count_emoji(tweet))
    capitalized_words.append(pf.count_capitalized_words(tokens))
    user_mentions.append(pf.count_user_mentions(tweet))
    number_hashtags.append(pf.count_hashtags(tweet))
    number_slang_exp.append(pf.count_slang_laughterexp(tweet))
    number_punctuations.append(pf.count_punctuation(tweet))
    pos, neg, neu, pol, flip = sf.sentiment_based_features(tweet)
    pos_words.append(pos)
    neg_words.append(neg)
    neutral_words.append(neu)
    tweet_polarity.append(pol)
    polarity_flip.append(flip)

#initialize empty lists for non-sarcastic tweets
ns_emoji_count = []
ns_capitalized_words = []
ns_user_mentions = []
ns_number_hashtags = []
ns_number_slang_exp = []
ns_number_punctuations = []
ns_pos_words = []
ns_neg_words = []
ns_neutral_words = []
ns_tweet_polarity = []
ns_polarity_flip = []

#generate features for non-sarcastic tweets
print("Generating features for non-sarcastic tweets")

for tweet in nonsarcasm_file:
    tokens = nltk.word_tokenize(tweet)
    ns_emoji_count.append(pf.count_emoji(tweet))
    ns_capitalized_words.append(pf.count_capitalized_words(tokens))
    ns_user_mentions.append(pf.count_user_mentions(tweet))
    ns_number_hashtags.append(pf.count_hashtags(tweet))
    ns_number_slang_exp.append(pf.count_slang_laughterexp(tweet))
    ns_number_punctuations.append(pf.count_punctuation(tweet))
    pos, neg, neu, pol, flip = sf.sentiment_based_features(tweet)
    ns_pos_words.append(pos)
    ns_neg_words.append(neg)
    ns_neutral_words.append(neu)
    ns_tweet_polarity.append(pol)
    ns_polarity_flip.append(flip)

#zip the features together
sarcasm_features = list(zip(emoji_count, capitalized_words, user_mentions, number_hashtags, number_slang_exp,
                      number_punctuations, pos_words, neg_words, neutral_words, tweet_polarity, polarity_flip))

nonsarcasm_features = list(zip(ns_emoji_count, ns_capitalized_words, ns_user_mentions, ns_number_hashtags, ns_number_slang_exp,
                         ns_number_punctuations, ns_pos_words, ns_neg_words, ns_neutral_words, ns_tweet_polarity, ns_polarity_flip))

#read the data into a dataframe, concatenate the data and write to csv
columns = ['Emoji', 'Capital Words', 'User Mentions', 'Hashtags', 'Slang laughter Exp', 'Punctuations', 
          '+ve Words', '-ve words', 'neutral wors', 'Polarity', 'Polarity flip']
final_sarcasm = pd.DataFrame(sarcasm_features, columns=columns)
final_nonsarcasm = pd.DataFrame(nonsarcasm_features, columns=columns)

label_sarcasm = [1] * final_sarcasm.shape[0]
label_nonsarcasm = [0] * final_nonsarcasm.shape[0]

final_sarcasm["label"] = label_sarcasm
final_nonsarcasm["label"] = label_nonsarcasm

dataframes = [final_sarcasm, final_nonsarcasm]
dataset = pd.concat(dataframes, ignore_index=True)

#write the data to a csv file.
dataset.to_csv("../Data/final_dataset.csv", index=False)

print("DONE!")

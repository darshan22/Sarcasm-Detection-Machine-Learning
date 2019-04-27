"""
This snippet is used to clean the tweets.
It removes the #sarcasm and #sarcastic keywords from the tweets.
It also removes the tweets containg hyperlinks
"""

import csv
import re
import codecs
import numpy as np

#function definition. Takes a file object as an input and returns a numpy array of clean tweets.
def clean_tweets(file_object):
    clean_tweets = []
    #regex to remove #sarcasm and #sarcastic from sarcastic tweets
    remove_sarcasm = re.compile(re.escape('#sarcasm'), re.IGNORECASE)
    remove_sarcastic = re.compile(re.escape('#sarcastic'), re.IGNORECASE)
    remove_hyperlink = re.compile(re.escape('hyperlink'), re.IGNORECASE)
    for tweet in file_object:
        temp = tweet
        if len(temp) > 0 and 'http' not in temp:
            temp = remove_sarcasm.sub('',temp)
            temp = remove_sarcastic.sub('',temp)
            temp = remove_hyperlink.sub('', temp)
            #remove useless spaces in a tweet
            temp = ' '.join(temp.split())
            if len(temp.split()) > 3:
                clean_tweets.append(temp)
    #create a numpy array of the clean tweets
    clean_tweets = np.array(clean_tweets)
    
    return clean_tweets
            
# load the sarcasm and nonsarcasm dataset
print("Reading the data")
sarcasm_file = codecs.open('../Data/sarcasm_data.txt', encoding='utf-8')
non_sarcasm_file = codecs.open('../Data/nonsarcasm_data.txt', encoding='utf-8')

sarcasm_clean = clean_tweets(sarcasm_file)
nonsarcasm_clean = clean_tweets(non_sarcasm_file)

#write to a text file
print("Writing to a text file")
np.savetxt('../Data/sarcasm_clean.txt', sarcasm_clean, fmt='%s')
np.savetxt('../Data/nonsarcasm_clean.txt', nonsarcasm_clean, fmt='%s')

print("Done!")

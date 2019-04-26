import re
import csv
import pandas as pd

data1 = pd.read_csv('../Data/raw_data/Train_v1.txt', sep='\t', header=None, names = ['Index', 'Label', 'Tweet'])
data2 = pd.read_csv('../Data/raw_data/Test_v1.txt', sep='\t', header=None, names = ['Index', 'Label', 'Tweet'])
data1.drop('Index', axis = 1, inplace = True)
data2.drop('Index', axis = 1, inplace = True)

sarcasm_data = []
nonsarcasm_data = []
with open('../Data/raw_data/sarcasm_tweets.txt', 'r') as sarcasm_file:
    sarcasm_data.append(sarcasm_file.readlines())
with open('../Data/raw_data/nonsarcasm_tweets.txt', 'r') as nonsarcasm_file:
    nonsarcasm_data.append(nonsarcasm_file.readlines())
sarcasm_dataframe = pd.DataFrame(sarcasm_data).T
nonsarcasm_dataframe = pd.DataFrame(nonsarcasm_data).T
sarcasm_dataframe.columns = ['Tweet']
nonsarcasm_dataframe.columns = ['Tweet']

dataframes = [data1, data2]
final_data = pd.concat(dataframes)
sarcasm_tweets = final_data[final_data['Label'] == 1]
nonsarcasm_tweets = final_data[final_data['Label'] == 0]

sarcasm_tweets.is_copy = False
nonsarcasm_tweets.is_copy = False
sarcasm_tweets.drop('Label', axis=1, inplace=True)
nonsarcasm_tweets.drop('Label', axis=1, inplace=True)

sarcasm_tweets.reset_index(inplace=True)
sarcasm_tweets.drop('index', axis = 1, inplace=True)
nonsarcasm_tweets.reset_index(inplace=True)
nonsarcasm_tweets.drop('index', axis = 1, inplace=True)

sarcasm_dataframes_list = [sarcasm_tweets, sarcasm_dataframe]
nonsarcasm_dataframes_list = [nonsarcasm_tweets, nonsarcasm_dataframe]
final_sarcasm = pd.concat(sarcasm_dataframes_list)
final_nonsarcasm = pd.concat(nonsarcasm_dataframes_list)

final_sarcasm.to_csv('../Data/sarcasm_data.txt', sep='\t', index=False, header=False, quoting=csv.QUOTE_NONE, escapechar='\\')
final_nonsarcasm.to_csv('../Data/nonsarcasm_data.txt', sep='\t', index=False, header=False, quoting=csv.QUOTE_NONE, escapechar='\\')


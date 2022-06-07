import sys
import pandas as pd
import numpy as np
import re, string, unicodedata
import csv
import matplotlib.pyplot as plt
from scipy import special
from scipy.stats import lognorm
import collections
from wordcloud import WordCloud
import nltk
from nltk import word_tokenize, sent_tokenize, FreqDist
from nltk.stem import LancasterStemmer, WordNetLemmatizer
nltk.download
nltk.download('wordnet')
nltk.download('punkt')
from nltk.tokenize import TweetTokenizer
import spacy
import preprocessor as p

def detect_language(X):
    """ remove other language using langdetect """
    from langdetect import detect
    try:
        lang = detect(X)
        return(lang)
    except:
        return("other")

def load_data():
    # data = pd.read_csv('./D1.txt', names=columns, sep='\t', lineterminator='\n', error_bad_lines=False, quoting=csv.QUOTE_NONE)
    data = pd.read_csv('./D2.txt', names=columns, sep='\t', lineterminator='\n', error_bad_lines=False, quoting=csv.QUOTE_NONE)

    return data
#writing csv output files
def write_data():
    # df.to_csv (r'./export_D1.csv', index = False, header= True)
    # pd.DataFrame(all_words_unique_list).to_csv("./D1_unique.csv", index = False, header= True)
    # pd.DataFrame(word_count_dict.most_common(100)).to_csv("./D1_Top100.csv", index = False, header= ['word','count'])

    df.to_csv (r'./export_D2.csv', index = False, header= True)
    pd.DataFrame(all_words_unique_list).to_csv("./D2_unique.csv", index = False, header= True)
    pd.DataFrame(word_count_dict.most_common(100)).to_csv("./D2_Top100.csv", index = False, header= ['word','count'])
    pd.DataFrame(word_count_dict.most_common(500)).to_csv("./D2_Top500.csv", index = False, header= ['word','count'])

#compare the common between the two list
# def mostcommon():
#     columns = ['word' , 'count']
#     data1 = pd.read_csv('./D1_Top100.csv', names=columns, sep='\t', lineterminator='\n', error_bad_lines=False, quoting=csv.QUOTE_NONE)
#     data2 = pd.read_csv('./D2_Top100.csv', names=columns, sep='\t', lineterminator='\n', error_bad_lines=False, quoting=csv.QUOTE_NONE)
#     data1.head()
#     data2.head()
#     print(data1)
#     word_list1 = list(data1.values)
#     word_list2 = list(data2.values)
#     df = pd.DataFrame(word_list1)

    # for i,col in enumerate (word_list1):
    #     data1[col] = data1.String.map( lambda x:x.split(','[i].strip()))
   

    # data1['word_list1'] = df['word_list1'].apply(lambda x: tokenization(x))
    # data2['word_list2'] = df['word_list2'].apply(lambda x: tokenization(x))
    # data1.head()
    # data2.head()
    # print(data1.head(5))
    # word_list1 = list(data1['word_list1'].explode())
    # word_list2 = list(data2['word_list2'].explode())
    # word_count_dict1 = collections.Counter(word_list1)
    # word_count_dict2 = collections.Counter(word_list2)
    # print(word_count_dict1.most_common(10));
    # print(word_count_dict2.most_common(10));



#load tweet txt file
columns = ['id' , 'text']
df = load_data()
df.head()


#remove punctuation, special characters, etc. in tweets other character from another language 
#this assignment doesn't require to remove stop word
def remove_punct(text):
    # remove URLs
    text_clean = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', '', text) 
    # remove usernames
    text_clean = re.sub('@[^\s]+', '', text_clean) 
    # remove the # in #hashtag
    text_clean = re.sub(r'#([^\s]+)', r'\1', text_clean) 
    #special characters, etc.
    text_clean = ''.join([c for c in text_clean if ord(c) < 128])
    # remove punctuations and convert characters to lower case
    text_clean = "".join([char.lower() for char in text_clean if char not in string.punctuation]) 
    # substitute multiple whitespace with single whitespace
    # Also, removes leading and trailing whitespaces
    text_clean = re.sub('\s+', ' ', text_clean).strip()
    # remove numbers, but not number that within a word
    text_clean = re.sub(r'\b[0-9]+\b\s*', '', text_clean)
    return text_clean

# df['Tweet_punct'] = df['text'].apply(lambda x: remove_punct(x))
df['Tweet_punct'] = df['text'].apply(lambda x: remove_punct(x))
df.head(10)

def tokenization(text):
    text = re.split('\W+', text)
    text = list(filter(None,text))  # remove empty string without the tokenized tweet
    return text

#tokenize words in tweets

df['word_list'] = df['Tweet_punct'].apply(lambda x: tokenization(x.lower()))
df.head()
#finding word frequency from a list

#create a unique list of words:
all_words_unique_list = np.asarray((df['word_list'].explode()).unique())
print("Unique word:"+str(len(all_words_unique_list)))

#world list from DF
word_list = list(df['word_list'].explode())
word_list = filter(None, word_list)
# word_list = [_ for _ in word_list if '' not in _]
#create a dict of word frequency in tweets using collection.Counter
word_count_dict = collections.Counter(word_list)
#creating Normalized frequency count

normalized_count = {}
for k, v in word_count_dict.items():
    normalized_count[k] = v/len(df['word_list'])


#create a dict of word frequency in tweets using nltk
nltk_count = nltk.FreqDist(word_list)

print(nltk_count)
# #Plot bar with values from dict and label with keys
# plt.bar(range(len(word_count_dict)), word_count_dict.values(), align='center')
# plt.xticks(range(len(word_count_dict)), word_count_dict.keys())
# #Rotate labels by 90 degrees so you can see them
# locs, labels = plt.xticks()
# plt.setp(labels, rotation=90)

# plt.show()



# data = list(word_count_dict)
# params = (0.40951093774076597, 5.1802137214177684, 60.158303995566413)
# shape, loc, scale = params[0], params[1], params[2]
# prob = 1-lognorm.cdf(388,shape,loc=params[1], scale=params[2])
# count, bins, ignored = plt.hist(data,1000)

# mu = np.mean(np.log(data))
# sigma = np.std(np.log(data))
# x = np.linspace(min(bins),max(bins),10000)
# pdf = (np.exp(-(np.log(x)-mu)**2 / (2 * sigma**2)) / (x * sigma * np.sqrt(2*np.pi)))
# plt.xscale('log')
# plt.yscale('log')
# plt.plot(x,pdf,color='r',linewidth= 2)



print("Normalized for covid19 is: " + str(normalized_count['covid19']))
write_data()



# print("**********")
#
# print(nltk_count.most_common(10))
#
#
# #simple word cloud
# wordcloud = WordCloud().generate_from_frequencies(nltk_count)
# plt.figure(figsize=(10, 7))
# plt.imshow(wordcloud, interpolation="bilinear")
# plt.axis('off')
# plt.show()







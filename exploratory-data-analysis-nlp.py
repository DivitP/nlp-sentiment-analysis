#%load_ext autoreload
#%autoreload 2
import numpy as np
from collections import Counter
from importlib.machinery import SourceFileLoader
from os.path import join
import warnings
warnings.filterwarnings("ignore")

import nltk
nltk.download('punkt')

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import *
from nltk.corpus import stopwords
nltk.download('stopwords' ,quiet=True)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import multilabel_confusion_matrix
from sklearn import metrics
import gdown
import string
from wordcloud import WordCloud, STOPWORDS

import matplotlib.pyplot as plt
pd.set_option('max_colwidth', 100)
from sklearn.feature_extraction.text import CountVectorizer

nltk.download('wordnet')
nltk.download('punkt_tab')

def get_finance_train():
  df_train = pd.read_csv("finance_train.csv")
  return df_train
def get_finance_test():
  df_test = pd.read_csv("finance_test.csv")
  return df_test

def plot_word_cloud(text):
  text = text.Sentence.values
  wordcloud = WordCloud(
      width = 3000,
      height = 2000,
      background_color = 'black',
      stopwords = STOPWORDS).generate(str(text))
  fig = plt.figure(
      figsize = (10, 7),
      facecolor = 'k',
      edgecolor = 'k')
  plt.imshow(wordcloud, interpolation = 'bilinear')
  plt.axis('off')
  plt.tight_layout(pad=0)
  plt.show()

def preprocess_data(df):
  sentences = df.Sentence.values
  labels = df.Label.values
  tokenized_sentences = [word_tokenize(word) for word in sentences]
  filtered_sentences = [remove_stopwords(word) for word in tokenized_sentences]
  return filtered_sentences, labels

def plot_confusion_matrix(y_true,y_predicted):
  cm = metrics.confusion_matrix(y_true, y_predicted)
  print ("Plotting the Confusion Matrix")
  labels = ["Negative","Neutral","Positive"]
  df_cm = pd.DataFrame(cm,index =labels,columns = labels)
  fig = plt.figure(figsize=(14,12))
  res = sns.heatmap(df_cm, annot=True,cmap='Blues', fmt='g')
  plt.yticks([0.5,1.5,2.5], labels,va='center')
  plt.title('Confusion Matrix - TestData')
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.show()
  plt.close()

LABEL_MAP = {0 : "negative", 1 : "neutral", 2 : "positive"}
STOPWORD_SET = set(stopwords.words('english'))

df_train = get_finance_train()
df_train.head()

length = len(df_train)
print(length)
print(df_train['Sentence'].head(length))


df_test = get_finance_test()
df_test.head(20)

length = len(df_test)
print(length)
df_test.shape

LABEL_MAP = {
    '0': "negative",
    '1': "neutral",
    '2': "positive"
}

fig = plt.figure(figsize=(10,6))
plt.title('Training Set Label Distribution')
plt.xlabel('Sentiment Label')
plt.ylabel('Count')

df_train.groupby('Label').Sentence.count().plot.bar(ylim=0)
plt.show()

negative_data = df_train[df_train['Label'] == 0]
plot_word_cloud(negative_data)

positive_data = df_train[df_train['Label'] == 2]
plot_word_cloud(positive_data)

neutral_data = df_train[df_train['Label'] == 1]
plot_word_cloud(neutral_data)


def remove_stopwords(tokenized_sentence):
  filtered_sentence = []
  for word in tokenized_sentence:
    if word not in STOPWORD_SET:
      filtered_sentence.append(word)

  return filtered_sentence


train_sentences, train_labels = preprocess_data(df_train)
for i in range(5):
  print(train_sentences[i])


test_sentences, test_labels = preprocess_data(df_test)
for i in range(5):
  print(test_sentences[i])



all_sentences = ["Google has made very good progress in the AI sector.", "Google Stock is not at an all time high", "Google has lots of competitors in the AI race."]

vectorizer = CountVectorizer()

vectorizer.fit(all_sentences) #tokenize
bag_of_words_matrix = vectorizer.transform(all_sentences).toarray() #encode the sentences --> vectors

bag_of_words_matrix.shape
print(bag_of_words_matrix)
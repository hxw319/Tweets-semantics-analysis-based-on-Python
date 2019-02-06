import numpy as np
import pandas as pd
import csv,re,nltk
from collections import Counter
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import roc_curve,auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split

def reformat():
  dataset = pd.read_csv('data.csv', sep='^', quoting=csv.QUOTE_NONE, encoding='utf-8', header=0)
  count = Counter(line[0] for line in dataset['party'])
  print(count)
  plt.bar(count.keys(), count.values())
  plt.title('Number of democratic and republican tweets')
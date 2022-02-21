import json
import re

import pandas as pd
import seaborn as sns
from numerofwords import WordUtility
from preprocesstext import TextProcessor
from visualization import Visualization
from tokenvector import VectorizationToken
from tfidf import Tfidf
from customstopwords import StopWords
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt


import nltk
from nltk.corpus import stopwords
#nltk.download('stopwords')



with open('data.json', 'r') as fp:
    data = json.load(fp)
df = pd.DataFrame.from_dict(data["root"])

#Wordcount function
wordutility = WordUtility()
wordutility.count_words(df)

#Preprocessing of text
textprocessor=TextProcessor()
corpus=textprocessor.process(df)

#Visulaization of text
#visual=Visualization()
#visual.visulalize_text(corpus)

#visual.visualize_unigrams(corpus)

#vectorization and tokenization
vec = VectorizationToken()
vec.findWordcountVector(corpus)

#TF-IDF
tfidf = Tfidf()
tfidf.sort_tfidf_vectors(corpus)














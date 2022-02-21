from customstopwords import StopWords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


class VectorizationToken:
    def findWordcountVector(self, corpus):
        sw = StopWords()
        stop_words = sw.customizeStopWords()
        cv = CountVectorizer(max_df=0.8, stop_words=stop_words, max_features=10000, ngram_range=(1, 3))
        X = cv.fit_transform(corpus)
        #     print(list(cv.vocabulary_.keys())[:10])


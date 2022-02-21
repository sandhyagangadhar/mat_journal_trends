from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from preprocesstext import TextProcessor
from customstopwords import StopWords
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

import matplotlib.pyplot as plt


class Visualization:
    def visulalize_text(self, corpus):
        sw = StopWords()
        wordcloud = WordCloud(background_color='white',
                              stopwords=sw.customizeStopWords(),
                              max_words=100,
                              max_font_size=50,
                              random_state=42
                              ).generate(str(corpus))
        print(wordcloud)
        fig = plt.figure(1)
        plt.imshow(wordcloud)
        plt.axis('off')
        #  plt.show()
        fig.savefig("word1.png", dpi=900)

    # MOST frequently occuring words
    def get_top_n_words(self, corpus, n=None):
        vec = CountVectorizer().fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in
                      vec.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1],
                            reverse=True)
        return words_freq[:n]

    # Convert most freq words to dataframe for plotting bar plot
    def visualize_unigrams(self, corpus):
        top_words = self.get_top_n_words(corpus, n=2)
        top_df = pd.DataFrame(top_words)
        top_df.columns = ["Word", "Freq"]
        sns.set(rc={'figure.figsize': (8, 4)})
        g = sns.barplot(x="Word", y="Freq", data=top_df)
        g.set_xticklabels(g.get_xticklabels(), rotation=30)
    # plt.show()

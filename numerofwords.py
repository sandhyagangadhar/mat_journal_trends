#counting the words of each title from the dataframe
import pandas as pd
import nltk

class WordUtility:
    def count_words(self, df):
        wordcount_list = []
        for i in range(len(df)):
            res = len(df['title'][i].split())
            wordcount_list.append(res)

        df.insert(2, 'wordcount', wordcount_list)
        df.drop(df.columns[1], axis=1, inplace=True)
        freq = pd.Series(''.join(df['title']).split()).value_counts()[:3]
        freq1 = pd.Series(''.join(df['title']).split()).value_counts()[:-3]









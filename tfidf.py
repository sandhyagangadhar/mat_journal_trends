from sklearn.feature_extraction.text import TfidfTransformer
from customstopwords import StopWords
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import coo_matrix


class Tfidf():
    def get_morefequent_keywords(self, corpus):
        sw = StopWords()
        stop_words = sw.customizeStopWords()
        cv = CountVectorizer(max_df=0.8, stop_words=stop_words, max_features=10000, ngram_range=(1, 3))
        X = cv.fit_transform(corpus)
        tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
        tfidf_transformer.fit(X)
        feature_names = cv.get_feature_names_out()
        doc = corpus[2]
        tf_idf_vector = tfidf_transformer.transform(cv.transform([doc]))
        return tf_idf_vector, feature_names

    def sort_coo(self, coo_matrix):
        tuples = zip(coo_matrix.col, coo_matrix.data)
        return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

    def extract_topn_from_vector(slef,feature_names, sorted_items, topn=10):
        sorted_items = sorted_items[:topn]
        score_values = []
        feature_values = []
        for idx, score in sorted_items:
            score_values.append(round(score, 3))
            feature_values.append(feature_names[idx])
        results = {}
        for idx in range(len(feature_values)):
            results[feature_values[idx]] = score_values[idx]
        return results

    def sort_tfidf_vectors(self, corpus):
        tf_idf_vector, feature_names = self.get_morefequent_keywords(corpus)
        sorted_items = self.sort_coo(tf_idf_vector.tocoo())
        keywords = self.extract_topn_from_vector(feature_names, sorted_items, topn=10)
        # print("\nKeywords:")
        for k in keywords:
            print(k, keywords[k])

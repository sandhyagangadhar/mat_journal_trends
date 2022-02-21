import nltk
from nltk.corpus import stopwords
#nltk.download('stopwords')



class StopWords:
    def customizeStopWords(self):
        stop_words = set(stopwords.words("english"))
        new_words = ["using", "show", "result", "large", "also"]
        stop_words = stop_words.union(new_words)
        return(stop_words)

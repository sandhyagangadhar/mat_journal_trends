import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from customstopwords import StopWords

#nltk.download('wordnet')
#nltk.download('omw-1.4')

class TextProcessor:
    def process(self, df):
        corpus = []
        sw=StopWords()
        for i in range(0, 3):
            # Remove punctuations
            text = re.sub('[^a-zA-Z]', ' ', df['title'][i])
            # Convert to lowercase
            text = text.lower()
            # remove tags
            text = re.sub("&lt;/?.*?&gt;", " &lt;&gt; ", text)
            # remove special characters and digits
            text = re.sub("(\\d|\\W)+", " ", text)
            ##Convert to list from string
            text = text.split()
            ##Stemming
            ps = PorterStemmer()
            # Lemmatisation
            lem = WordNetLemmatizer()
            text = [lem.lemmatize(word) for word in text if not word in
                                                                sw.customizeStopWords()]
            text = " ".join(text)
            corpus.append(text)
        return(corpus)


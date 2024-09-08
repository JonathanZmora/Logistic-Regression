import nltk
import numpy as np
from collections import Counter
from nltk.corpus import stopwords


class BagOfWords:

    def __init__(self, n_features=1000):
        nltk.download("stopwords", quiet=True)
        self.stop_words = stopwords.words('english')
        self.n_features = n_features

    @staticmethod
    def remove_punctuation(text):
        """ Removes any character that is not a letter or a whitespace from the given text """
        result = ''.join(filter(lambda x: x.isalpha() or x.isspace(), text))
        return result

    def extract_words(self, text):
        """ Returns a list of all the words in the given text which are not stop words """
        words = self.remove_punctuation(text).split()
        cleaned_words = [w.lower() for w in words if w not in self.stop_words]
        return cleaned_words

    def create_vocabulary(self, text_list):
        """
        Returns a vocabulary containing the different words in all the texts in the given text list.
        The vocabulary returned is a Counter object in which the keys are the different words and the count
        represents the number of appearances of each words in all the texts in the given text list,
        and it contains the 'n_features' most common words.
        """
        vocabulary = Counter()
        for text in text_list:
            words = self.extract_words(text)
            vocabulary.update(words)
        most_common_vocab = vocabulary.most_common(self.n_features)
        vocab_with_indices = {word: index for index, (word, count) in enumerate(most_common_vocab)}
        return vocab_with_indices

    def generate_vectors(self, text_list):
        """
        Uses the bag of words algorithm to convert each text in the text list to a vector of numbers
        and returns a dataset containing all the generated vectors.
        The returned dataset can now be used to train and test a classifier.
        """
        vectors = []
        vocabulary = self.create_vocabulary(text_list)
        vocab_size = len(vocabulary)
        for text in text_list:
            vector = np.zeros(vocab_size)
            words = self.extract_words(text)
            for word in words:
                if word in vocabulary:
                    vector[vocabulary[word]] += 1
            vectors.append(vector)
        return np.array(vectors)

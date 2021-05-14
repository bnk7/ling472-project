# !/usr/bin/python3


# TODO: Implement a Laplace-smoothed trigram model :)
class LanguageModel:

    def __init__(self):
        pass

    def read_data(self, corpus): # Arshana
        # start and stop tokens
            # two start token on the first sentence
            # two stop on the last
            # rest one
            # sentence on each line
            # or double dashes
        # three dataframes: unigram index, bigram as index and each individual word as a column
        # trigram index and three columns for words
        pass

    def train_unk(self): # Brynna
        pass

    def smoothing(self): # Anna
        pass

    def probability(self): # Arshana
        pass

    def calculate_MLE(self): # Brynna
        pass

    def print_ngram(self): # Anna
        pass

    def train(self, train_corpus):
        print('I am an unimplemented TRIGRAM train() method.')  # delete this!

    def score(self, test_corpus):
        print('I am an unimplemented TRIGRAM score() method.')  # delete this!

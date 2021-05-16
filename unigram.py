import pandas as pd
import re
# !/usr/bin/python3

"""
read in data: one function
    split whitespace, or regex with no punctuation
    store in pandas, dataframe or series
    word as index

<UNK>: when word appears once


Smoothing to calculate the probabilities, add column to dataframe


print n-gram, MLE, add column to dataframe

score:

store the sum of probabilities for perplexity, but print probabilities as we go along

<unk> unseen words

probability for each sentence

Perplexity, sum each log probability for each sentence

"""



# TODO: Implement a Laplace-smoothed unigram model :)
class LanguageModel:

    def __init__(self):
        self.unigram = pd.DataFrame(columns=["count"])

    def read_data(self, corpus): # Anna
        f = open(corpus, 'r')
        pattern = r'[^a-zA-Z0-9\s]'
        entire_file = f.read()
        entire_file = re.sub(pattern=pattern, repl='', string=entire_file)
        entire_file = entire_file.split()
        for word in entire_file:
            if word in self.unigram.index:
                self.unigram.loc[word, "count"] += 1
            else:
                row = pd.Series(data={"count": 1}, name=word)
                self.unigram = self.unigram.append(row, ignore_index=False)
        print(self.unigram)
        f.close()


    def train_unk(self): # Arshana
        pass

    def train_prob(self): # Brynna
        #smoothing here too
        pass

    def print_ngram(self): #Arshana
        pass

    def train(self, train_corpus):
        self.read_data(train_corpus)
        print('I am an unimplemented UNIGRAM train() method.')  # delete this!

    def score_unk(self, sent):
        pass

    def score_prob(self, sent):
        pass

    def calc_perplex(self, sum, count):
        pass

    def score(self, test_corpus):
        # move through sentences, pass one at a time
        # use read_data
        # sum probability variable
        # count sentences variable
        print('I am an unimplemented UNIGRAM score() method.')  # delete this!

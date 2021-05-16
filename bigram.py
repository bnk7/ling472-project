import pandas as pd
import re
# !/usr/bin/python3


# TODO: Implement a Laplace-smoothed bigram model :)
class LanguageModel:

    def __init__(self):
        self.unigram = pd.DataFrame(columns=["count"])
        self.bigram = pd.DataFrame(columns=["w1", "w2", "count"])
        
        

    def read_data(self, corpus): #Brynna
        # start and stop tokens
            # sentence on each line
            # or double dashes
        # two dataframes: unigram index, bigram as index and each individual word as a column
        pass

    def train_unk(self): # Anna
        # unigram
        is_one = self.unigram["count"] == 1
        unk_count = self.unigram[is_one].size
        unked_words = self.unigram[is_one].index
        unked_df = self.unigram[self.unigram["count"] != 1]
        row = pd.Series(data={"count": unk_count}, name="<UNK>")
        self.unigram = unked_df.append(row, ignore_index=False)
        
        # bigram
        unked_bigram = self.bigram.replace(unked_words, "<UNK>")
        unked_bigram = unked_bigram.groupby(['w1', 'w2']).sum()
        unked_bigram2 = pd.DataFrame()
        for tup in unked_bigram.index:
            w1, w2 = tup
            row = pd.Series(data=[w1, w2, unked_bigram.loc[w1, w2]["count"]], name=w1 + " " + w2)
            unked_bigram2 = unked_bigram2.append(row, ignore_index=False)
        unked_bigram2.columns = ["w1", "w2", "count"]
        self.bigram = unked_bigram2
        print(self.bigram)
        

    def smoothing(self): # Arshana
        pass

    def train_prob(self): # Anna
        pass

    def print_ngram(self): # Brynna
        pass

    def train(self, train_corpus):
        print('I am an unimplemented BIGRAM train() method.')  # delete this!

    def score(self, test_corpus):
        print('I am an unimplemented BIGRAM score() method.')  # delete this!

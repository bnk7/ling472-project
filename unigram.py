# !/usr/bin/python3

import pandas as pd
import re
from math import log2
from pathlib import Path

class LanguageModel:

    def __init__(self):
        self.df = pd.DataFrame(columns=["cnt"])

    def read_data(self, corpus):
        """
        reads in corpus data to fill in the unigram df with the counts for each word
        index: word
        cnt: count of word
        """
        f = open(corpus, 'r')
        pattern = r'[^a-zA-Z0-9\s]'
        entire_file = f.read()
        entire_file = re.sub(pattern=pattern, repl='', string=entire_file)
        entire_file = entire_file.split()
        for word in entire_file:
            if word in self.df.index:
                self.df.loc[word, "cnt"] += 1
            else:
                row = pd.Series(data={"cnt": 1}, name=word)
                self.df = self.df.append(row, ignore_index=False)
        f.close()

    def train_unk(self):
        num_unk = self.df.loc[self.df['cnt'] == 1].size
        # remove UNKed words
        self.df.drop(self.df[self.df['cnt'] == 1].index, inplace=True)
        # create UNK row
        row = pd.Series(data={'cnt': num_unk}, name="<UNK>")
        # append row
        self.df = self.df.append(row, ignore_index=False)

    # returns the smoothed probability of a single word
    def get_train_prob(self, cnt):
        # P = (count of unigram + 1)/(sum of all counts + size of vocab)
        num_tokens = self.df.cnt.sum()
        num_types = self.df.shape[0]
        denominator = num_tokens + num_types
        prob = log2((cnt + 1)/denominator)
        return prob

    # applies Laplace smoothing and adds an "MLE" column
    def train_prob(self):
        self.df['MLE'] = self.df.cnt.apply(self.get_train_prob)

    def print_ngram(self):
        # fix alphabetical
        self.df.index.name = "index"
        self.df.sort_values(by=['MLE', "index"], ascending = [False, True], inplace = True)

        for index, row in self.df.iterrows():
            print(index, round(row['MLE'], 3))

    def train(self, train_corpus):
        """
        pass in the training corpus, prints each unigram with its logged MLE
        creates unigram_df.csv for efficiency after training the model once
        """
        filename = "unigram_df.csv"
        if Path(filename).exists():
            self.df = pd.read_csv(filename, index_col=0)
        else:
            self.read_data(train_corpus)
            self.train_unk()
            self.train_prob()
            self.df.to_csv(filename)
        self.print_ngram()


    def score_unk(self, sent):
        """
        takes in a string sentence with no punctuation
        returns the unked string
        """
        unked_sent = ""
        for word in sent.split():
            if word in self.df.index:
                unked_sent += word + " "
            else:
                unked_sent += "<UNK> "
        return unked_sent

    # returns the probability of a sentence
    def score_prob(self, sent):
        # start with 0 because we're adding
        prob = 0
        sent_list = sent.split()
        for word in sent_list:
            if word in self.df.index:
                MLE = self.df.loc[word, 'MLE']
            else:
                # P = (1)/(sum of all counts + size of vocab)
                num_tokens = self.df.cnt.sum()
                num_types = self.df.shape[0]
                denom = num_tokens + num_types
                MLE = log2(1.0/denom)
            # adding logs is equivalent to multiplying typical numbers
            prob += MLE
        return prob

    def calc_perplex(self, sum, count):
        H = -sum / count
        return round(2 ** H, 3)

    def score(self, test_corpus):
        """
        Reads in the test corpus and prints each line with its logged probability
        Prints out the perplexity of the system at the end. All rounded to the
        third decimal place.
        """
        total_prob = 0
        num_words = 0
        pattern = r'[^a-zA-Z0-9\s]'
        # read in data
        f = open(test_corpus, 'r')
        lines = f.readlines()
        f.close()
        # per sentence w/ prob
        for line in lines:
            clean_line = re.sub(pattern=pattern, repl='', string=line)
            num_words += len(clean_line.split())
            unked_line = self.score_unk(clean_line)
            prob = self.score_prob(unked_line)
            total_prob += prob
            print(line.strip() + " " + str(prob))
        # System's perplexity
        perplex = self.calc_perplex(total_prob, num_words)
        print("Perplexity = " + str(perplex))

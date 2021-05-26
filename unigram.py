# !/usr/bin/python3

import pandas as pd
import re
from math import log2
from pathlib import Path

"""
1. save probabilties into csv and read them in train if they exist
2. Generate sentences for trigram
    randomize top 3 probabilities
    randomize for the first trigram with start token </s> <s> word or <s> <s> word
    print(1 if trigram runs well, 5 if bad)
3. Error analysis on 5 sentences each
4. Write-up

"""

class LanguageModel:

    def __init__(self):
        self.df = pd.DataFrame(columns=["cnt"])

    def read_data(self, corpus): # Anna
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

    def train_unk(self): # Arshana
        num_unk = self.df.loc[self.df['cnt'] == 1].size
        # remove UNKed words
        self.df.drop(self.df[self.df['cnt'] == 1].index, inplace=True)
        # create UNK row
        row = pd.Series(data={'cnt': num_unk}, name="<UNK>")
        # append row
        self.df = self.df.append(row, ignore_index=False)

    # returns the smoothed probability of a single word
    def get_train_prob(self, cnt): # Brynna
        # P = (count of unigram + 1)/(sum of all counts + size of vocab)
        num_tokens = self.df.cnt.sum()
        num_types = self.df.shape[0]
        denominator = num_tokens + num_types
        prob = log2((cnt + 1)/denominator)
        return prob

    # applies Laplace smoothing and adds a "probability" column
    def train_prob(self): # Brynna
        self.df['MLE'] = self.df.cnt.apply(self.get_train_prob)

    def print_ngram(self): #Arshana
        # fix alphabetical
        self.df.index.name = "index"
        self.df.sort_values(by=['MLE', "index"], ascending = [False, True], inplace = True)

        for index, row in self.df.iterrows():
            print(index, round(row['MLE'], 3))

    def train(self, train_corpus):
        filename = "unigram_df.csv"
        if Path(filename).exists():
            self.df = pd.read_csv(filename, index_col=0)
        else:
            self.read_data(train_corpus)
            self.train_unk()
            self.train_prob()
            self.df.to_csv(filename)
        self.print_ngram()


    def score_unk(self, sent): # Anna
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
    def score_prob(self, sent): # Brynna
        # start with 0 because we're adding
        prob = 0
        sent_list = sent.split()
        for word in sent_list:
            MLE = self.df.loc[word, 'MLE']
            # adding logs is equivalent to multiplying normal numbers
            prob += MLE
        return prob

    def calc_perplex(self, sum, count): # Arshana
        #untested
        H = -sum / count
        return round(2 ** H, 3)

    def score(self, test_corpus): # Anna
        """
        Reads in the test corpus and prints each line with its logged probability
        Prints out the perplexity of the system at the end. All rounded to the
        third decimal place.
        """
        # I have not tested any of this code yet.
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

        # total_prob = 0
        # break test_corpus -> entire_file
            # print (line1)
            # score_unk (line1) -> return unked sent
                # add start and stop tokens
            # score_prob(sent) -> return prob1
            # total_prob += prob1
        # calc_perplex(total_prob, num_words)

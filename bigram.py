# !/usr/bin/python3

from math import log2
import pandas as pd
import re
from pathlib import Path

class LanguageModel:

    def __init__(self):
        self.unigram_df = pd.DataFrame(columns=["cnt"])
        self.bigram_df = pd.DataFrame(columns=["w1", "w2", "cnt"])

    def read_data(self, corpus):
        f = open(corpus, 'r')
        lines = f.readlines()
        entire_file = ""
        for line in lines:
            line = line.strip()
            # get rid of most punctuation
            line = re.sub(pattern=r"[^a-zA-Z0-9\s]", repl="", string=line)
            # add beginning and end of sentence tokens
            entire_file += "<s> " + line + " </s> "
        f.close()

        # adapt unigram code to make unigram df
        entire_file = entire_file.split()
        for word in entire_file:
            if word in self.unigram_df.index:
                self.unigram_df.loc[word, "cnt"] += 1
            else:
                row = pd.Series(data={"cnt": 1}, name=word)
                self.unigram_df = self.unigram_df.append(row, ignore_index=False)

        # make bigram df with bigram as index and each individual word as a column
        for i in range(len(entire_file)-1):
            first_word = entire_file[i]
            second_word = entire_file[i+1]
            bigram = first_word + " " + second_word
            if bigram in self.bigram_df.index:
                self.bigram_df.loc[bigram, "cnt"] += 1
            else:
                row = pd.Series(data={"cnt": 1, "w1": first_word, "w2": second_word}, name=bigram)
                self.bigram_df = self.bigram_df.append(row, ignore_index=False)

    def train_unk(self):
        """
        Unks the unigram df. Using the unked word list from the unigram,
        also unks the bigram df.
        """
        # UNIGRAM
        is_one = self.unigram_df["cnt"] == 1
        # count of words to unk
        unk_count = len(self.unigram_df[is_one].index)
        # all unked words
        unked_words = self.unigram_df[is_one].index
        # dataframe with none of the unked words
        unked_df = self.unigram_df[self.unigram_df["cnt"] != 1]
        # create row for unk
        row = pd.Series(data={"cnt": unk_count}, name="<UNK>")
        # append row and save to class variable
        self.unigram_df = unked_df.append(row, ignore_index=False)

        # BIGRAM
        # replace words that need unk from unigram unk list
        unked_bigram = self.bigram_df.replace(unked_words, "<UNK>")
        # sum counts with same w1, w2
        unked_bigram = unked_bigram.groupby(['w1', 'w2']).sum()
        # new dataframe to save groupby with correct indexes and column names
        unked_bigram2 = pd.DataFrame()
        for tup in unked_bigram.index:
            w1, w2 = tup
            row = pd.Series(data=[w1, w2, unked_bigram.loc[w1, w2]["cnt"]], name=w1 + " " + w2)
            unked_bigram2 = unked_bigram2.append(row, ignore_index=False)
        unked_bigram2.columns = ["w1", "w2", "cnt"]
        # save to class variable
        self.bigram_df = unked_bigram2


    # include smoothing in train_prob
    def train_prob(self):
        """
        Adds a MLE column to the bigram df.
        """
        # for each row in bigram df
        for index, row in self.bigram_df.iterrows():
            # bigram count
            num = row['cnt'] + 1
            # w1 count + vocab count (no <s> already)
            denom = self.unigram_df.loc[row['w1'], 'cnt'] + len(self.unigram_df.index) - 1
            # prob: count / w1 count + vocab count
            # add column to bigram df
            self.bigram_df.loc[index, 'MLE'] = log2(float(num)/denom)

    # prints bigrams and their MLEs, rounded to 3 decimal places
    # and sorted by MLE (descending) and then bigram (alphabetical)
    def print_ngram(self):
        self.bigram_df.index.name = "index"
        self.bigram_df.sort_values(by=['MLE', "index"], ascending = [False, True], inplace = True)
        # print
        for index, row in self.bigram_df.iterrows():
    	    print(index, round(row['MLE'], 3))

    def train(self, train_corpus):
        """
        pass in the training corpus. prints each bigram with its logged MLE.
        creates bigram_df.csv and bigram_uni_df.csv for efficiency after training the model once
        """
        filename = "bigram_df.csv"
        filename2 = "bigram_uni_df.csv"
        if Path(filename).exists() and Path(filename2).exists():
            self.bigram_df = pd.read_csv(filename, index_col=0)
            self.unigram_df = pd.read_csv(filename2, index_col=0)
        else:
            self.read_data(train_corpus)
            self.train_unk()
            self.train_prob()
            self.bigram_df.to_csv(filename)
            self.unigram_df.to_csv(filename2)
        self.print_ngram()

    # removes punctuation and adds stop tokens
    # adds start tokens if passed True (default is False)
    def normalize_line(self, line, start_token=False):
        line = re.sub(pattern=r"[^a-zA-Z0-9\s]", repl="", string=line)
        line = line.strip()
        line = line + " </s>"
        if start_token == True:
            line = "<s> " + line
        return line

    # returns an UNKed sentence with start and end tokens
    def score_unk(self, sent):
        sent = self.normalize_line(sent, True)
        sent_list = sent.split()
        i = 0
        for word in sent_list:
            if word not in self.unigram_df.index:
                sent_list[i] = "<UNK>"
            i += 1
        # re-form the sentence and return
        return " ".join(sent_list)

    def score_prob(self, sent):
        # adapted from trigram
        prob = 0
        sent_list = sent.split()
        for i in range(len(sent_list)):
            if (i + 1) < len(sent_list):
                uni_idx = sent_list[i]
                idx = uni_idx + " " + sent_list[i+1]
                if idx in self.bigram_df.index:
                    MLE = self.bigram_df.loc[idx, 'MLE']
                elif uni_idx in self.unigram_df.index:
                    denom = self.unigram_df.loc[uni_idx, 'cnt'] + len(self.unigram_df.index) - 1
                    MLE = log2(1.0/denom)
                else:
                    denom = len(self.unigram.index) - 1
                    MLE = log2(1.0/denom)
                prob += MLE
        return prob

    def calc_perplex(self, sum, count):
        """
        takes in the the sum of all the probabilities for all the sentences,
        and the count of words in the test file, returns the perplexity for
        the model
        """
        H = -sum/count
        return round(2 ** H, 3)

    # takes a line, unks it, prints it and its probability, and returns the probability
    def score_line(self, line):
        unked_line = self.score_unk(line)
        prob = self.score_prob(unked_line)
        print(line.strip() + " " + str(prob))
        return prob

    def score(self, test_corpus): 
        # read in file
        f = open(test_corpus, 'r')
        lines = f.readlines()
        f.close()

        # convert to Series in order to apply functions to each line
        lines_series = pd.Series(lines, dtype = "string")
        total_prob = lines_series.apply(self.score_line).sum()

        # calculate N = total words (including the stop but not the start tokens)
        normalized_lines = lines_series.apply(self.normalize_line)
        entire_file = " ".join(normalized_lines)
        N = len(entire_file.split())

        # print perplexity
        perplexity = self.calc_perplex(total_prob, N)
        print("Perplexity = " + str(round(perplexity, 3)))

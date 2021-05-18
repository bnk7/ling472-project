# !/usr/bin/python3

from math import log2
import pandas as pd
import re

class LanguageModel:

    def __init__(self):
        self.unigram_df = pd.DataFrame(columns=["cnt"])
        self.bigram_df = pd.DataFrame(columns=["word1", "word2", "cnt"])

    def read_data(self, corpus): # Brynna
        lines = corpus.readlines()
        entire_file = ""
        for line in lines:
            line = line.strip()
            # get rid of most punctuation
            line = re.sub(pattern=r" [^a-zA-Z0-9\s-]", repl="", string=line)
            # remove if "--" doesn't mark the end of a sentence
            line = re.sub(pattern="--", repl="</s> <s>", string=line)
            # add beginning and end of sentence tokens
            entire_file += "<s> " + line + " </s> "

        # adapt Anna's unigram code to make unigram df
        entire_file = entire_file.split()
        for word in entire_file:
            if word in self.unigram_df.index:
                self.unigram_df.loc[word, "cnt"] += 1
            elif word != "<s>":
                row = pd.Series(data={"cnt": 1}, name=word)
                self.unigram_df = self.unigram_df.append(row, ignore_index=False)

        # make bigram df with bigram as index and each individual word as a column
        # I used Anna's code as a starting point here too
        for i in range(len(entire_file)-1):
            first_word = entire_file[i]
            second_word = entire_file[i+1]
            bigram = first_word + " " + second_word
            if bigram in self.bigram_df.index:
                self.bigram_df.loc[bigram, "cnt"] += 1
            elif second_word != "<s>":
                row = pd.Series(data={"cnt": 1, "word1": first_word, "word2": second_word}, name=bigram)
                self.bigram_df = self.bigram_df.append(row, ignore_index=False)

    def train_unk(self): # Anna
        """
        Unks the unigram df. Using the unked word list from the unigram,
        also unks the bigram df.
        """
        # UNIGRAM
        is_one = self.unigram_df["cnt"] == 1
        # count of words to unk
        unk_count = self.unigram_df[is_one].size
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

    def add_one(x):
        return x + 1

    def smoothing(self): # Arshana
        # for each w1 in vocab list
        for w1 in unigram_df.index:
            # for each w2 in vocab list
            for w2 in unigram_df.index:
                # if "<w1> <w2>" is not an index in df then add it to df with 0 occurrences
                idx = w1 + w2
                if idx not in self.unigram_df.index:
                    row = pd.Series(data={"cnt": 0}, name=idx)

        # loop through df
            # add 1 to each ngram in df
        self.bigram_df['cnt'] = self.bigram_df['cnt'].apply(add_one)

    def train_prob(self): # Anna
        """
        Adds a probability column to the bigram df.
        """
        # for each row in bigram df
        for index, row in self.bigram_df.iterrows():
            # bigram count
            num = row['cnt']
            # w1 count + vocab count (no <s> already)
            denom = self.unigram_df.loc[row['w1'], 'cnt'] + self.unigram_df.size
            # prob: count / w1 count + vocab count
            # add column to bigram df
            self.bigram_df.loc[index, 'prob'] = float(num)/denom

    # prints bigrams and their logged MLEs, rounded to 3 decimal places
    # and sorted by logged MLE (descending) and then bigram (alphabetical)
    def print_ngram(self): # Brynna
        # add loggedMLE column
        self.bigram_df['loggedMLE'] = self.bigram_df.apply(log2)
        # sort alphabetically
        self.bigram_df.sort_index(inplace = True)
        # sort by loggedMLE
        self.bigram_df.sort_values(['loggedMLE'], ascending = False, inplace = True)
        # print
        print(round(self.bigram_df.loggedMLE, 3))

    def train(self, train_corpus):
        print('I am an unimplemented BIGRAM train() method.')  # delete this!
        # In what format is train_corpus being passed in? 
        # read_data(train_corpus)

    def score(self, test_corpus):
        print('I am an unimplemented BIGRAM score() method.')  # delete this!

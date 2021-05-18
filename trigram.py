from math import log2
import pandas as pd
import re
# !/usr/bin/python3

class LanguageModel:

    def __init__(self):
        self.unigram = pd.DataFrame(columns=["cnt"])
        self.bigram = pd.DataFrame(columns=["w1", "w2", "cnt"])
        self.trigram = pd.DataFrame(columns=["w1", "w2", "w3", "cnt"])

    def read_data(self, corpus): # Arshana
        # trigram index and three columns for words

        # loop through list
            # for df
                # create trigram indices
                # add w1, w2, w3 columns
                # add count column and increment as appropriate
            # for vocab df
                # create unigram indices
                # add count column and increment as appropriate
        # adapt code from bigram file
        entire_file = corpus.read()
        # extra start and stop token on first and last lines
        entire file = "<s> " + entire_file + "</s>"
        # get rid of most punctuation
        entire_file = re.sub(pattern=r'[^a-zA-Z0-9\s-]', repl="", string=entire_file)
        # add beginning and end of sentence tokens
        for line in entire_file:
            line = line.strip()
            line = "<s> " + line + " </s>"
        # remove if -- doesn't mark the end of a sentence
        entire_file = re.sub(pattern="--", repl="</s> <s>", string=entire_file)

        # unigram df
        entire_file = entire_file.split()
        for word in entire_file:
            if word in self.unigram.index:
                self.unigram.loc[word, "cnt"] += 1
            elif word != "<s>" and word != "</s":
                row = pd.Series(data={"cnt": 1}, name=word)
                self.unigram = self.unigram.append(row, ignore_index=False)
        # bigram df
        # trigram df
        pass

    # changes tokens only seen once into <UNK> and updates all dataframes
    def train_unk(self): # Brynna
        # adapt Arshana's unigram train_unk to UNK unigram_df
        num_unk = self.unigram_df.loc[self.unigram_df["cnt"] == 1].size
        unked_words = self.unigram_df[self.unigram_df["cnt"] == 1].index
        df = self.unigram_df[self.unigram_df["cnt"] != 1]
        row = pd.Series(data={"cnt": num_unk}, name="<UNK>")
        self.unigram_df = df.append(row, ignore_index=False)

        # adapt Anna's bigram train_unk to UNK bigram_df
        self.bigram_df = self.bigram_df.replace(unked_words, "<UNK>")
        self.bigram_df = self.bigram_df.groupby(['word1', 'word2']).sum()
        unked_bigram = pd.DataFrame()
        for tup in self.bigram_df.index:
            w1, w2 = tup
            row = pd.Series(data=[w1, w2, self.bigram_df.loc[w1, w2]["cnt"]], name=w1 + " " + w2)
            unked_bigram = unked_bigram.append(row, ignore_index=False)
            unked_bigram.columns = ["word1", "word2", "cnt"]
        unked_bigram["cnt"] = unked_bigram.cnt.apply(int)
        self.bigram_df = unked_bigram

        # trigram
        self.trigram_df = self.trigram_df.replace(unked_words, "<UNK>")
        self.trigram_df = self.trigram_df.groupby(['word1', 'word2', 'word3']).sum()
        unked_trigram = pd.DataFrame()
        for tup in self.trigram_df.index:
            w1, w2, w3 = tup
            row = pd.Series(data=[w1, w2, w3, self.trigram_df.loc[w1, w2, w3]["cnt"]], \
                name=w1 + " " + w2 + " " + w3)
            unked_trigram = unked_trigram.append(row, ignore_index=False)
            unked_trigram.columns = ["word1", "word2", "word3", "cnt"]
        unked_trigram["cnt"] = unked_trigram.cnt.apply(int)
        self.trigram_df = unked_trigram

    def smoothing(self): # Anna
        pass

    def train_prob(self): # Arshana
        pass

    def print_ngram(self): # Anna
        """
        log2 the probability and save to new MLE column
        prints out each trigram with its logged MLE
        """
        # <w1> <w2> <w3> <log2(P(w3|w1 w2))>
        # highest to lowest prob, 3 decimal place rounded
        # then alphabetical
        self.trigram["loggedMLE"] = self.trigram["prob"].apply(log2)
        self.trigram.sort_index(inplace=True)
        self.trigram.sort_values(by=['prob'], inplace=True, ascending=False)
        print(round(self.trigram["loggedMLE"], 3))


    def train(self, train_corpus):
        print('I am an unimplemented TRIGRAM train() method.')  # delete this!

    def score(self, test_corpus):
        print('I am an unimplemented TRIGRAM score() method.')  # delete this!

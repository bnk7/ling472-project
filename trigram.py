from math import log2
import pandas as pd
import re
# !/usr/bin/python3


# TODO: Implement a Laplace-smoothed trigram model :)
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

    def train_unk(self): # Brynna
        pass

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

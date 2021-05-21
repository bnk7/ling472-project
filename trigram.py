from math import log2
import pandas as pd
import re
# !/usr/bin/python3

class LanguageModel:

    def __init__(self):
        self.unigram = pd.DataFrame(columns=["cnt"])
        self.bigram = pd.DataFrame(columns=["word1", "word2", "cnt"])
        self.trigram = pd.DataFrame(columns=["word1", "word2", "word3", "cnt"])

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
        # adapt code from bigram.py
        entire_file = corpus.read()
        # extra start and stop token on first and last lines
        entire_file = "<s> " + entire_file
        # get rid of most punctuation
        entire_file = re.sub(pattern=r'[^a-zA-Z0-9\s]', repl="", string=entire_file)
        # add beginning and end of sentence tokens
        for line in entire_file:
            line = line.strip()
            line = "<s> " + line + " </s>"

        # unigram df - adapted from bigram.py
        entire_file = entire_file.split()
        for word in entire_file:
            if word in self.unigram.index:
                self.unigram.loc[word, "cnt"] += 1
            else:
                row = pd.Series(data={"cnt": 1}, name=word)
                self.unigram = self.unigram.append(row, ignore_index=False)
        # bigram df - adapted from bigram.py
        for i in range(len(entire_file)-1):
            first_word = entire_file[i]
            second_word = entire_file[i+1]
            gram = first_word + " " + second_word
            if gram in self.bigram.index:
                self.bigram.loc[gram, "cnt"] += 1
            else:
                row = pd.Series(data={"cnt": 1, "word1": first_word, "word2": second_word}, name=gram)
                self.bigram = self.bigram.append(row, ignore_index=False)
        # trigram df - adapted from bigram.py
        for i in range(len(entire_file)-2):
            w1 = entire_file[i]
            w2 = entire_file[i+1]
            w3 = entire_file[i+2]
            gram = w1 + " " + w2 + " " + w3
            if gram in self.trigram.index:
                self.trigram.loc[gram, "cnt"] += 1
            else:
                row = pd.Series(data={"cnt": 1, "word1": w1, "word2": w2, "word3": w3}, name=gram)
                self.trigram = self.trigram.append(row, ignore_index=False)


    # changes tokens only seen once into <UNK> and updates all dataframes
    def train_unk(self): # Brynna
        # adapt Arshana's unigram train_unk to UNK unigram_df
        num_unk = self.unigram.loc[self.unigram["cnt"] == 1].size
        unked_words = self.unigram[self.unigram["cnt"] == 1].index
        df = self.unigram[self.unigram["cnt"] != 1]
        row = pd.Series(data={"cnt": num_unk}, name="<UNK>")
        self.unigram = df.append(row, ignore_index=False)

        # adapt Anna's bigram train_unk to UNK bigram_df
        self.bigram = self.bigram.replace(unked_words, "<UNK>")
        self.bigram = self.bigram.groupby(['word1', 'word2']).sum()
        unked_bigram = pd.DataFrame()
        for tup in self.bigram.index:
            w1, w2 = tup
            row = pd.Series(data=[w1, w2, self.bigram.loc[w1, w2]["cnt"]], name=w1 + " " + w2)
            unked_bigram = unked_bigram.append(row, ignore_index=False)
        unked_bigram.columns = ["word1", "word2", "cnt"]
        unked_bigram["cnt"] = unked_bigram.cnt.apply(int)
        self.bigram = unked_bigram

        # trigram
        self.trigram = self.trigram.replace(unked_words, "<UNK>")
        self.trigram = self.trigram.groupby(['word1', 'word2', 'word3']).sum()
        unked_trigram = pd.DataFrame()
        for tup in self.trigram.index:
            w1, w2, w3 = tup
            row = pd.Series(data=[w1, w2, w3, self.trigram.loc[w1, w2, w3]["cnt"]], \
                name=w1 + " " + w2 + " " + w3)
            unked_trigram = unked_trigram.append(row, ignore_index=False)
        unked_trigram.columns = ["word1", "word2", "word3", "cnt"]
        unked_trigram["cnt"] = unked_trigram.cnt.apply(int)
        self.trigram = unked_trigram

        # add something if unk if missing from data possibly

    def train_prob(self): # Arshana
        # folded smoothing into this method
        for index, row in self.trigram.iterrows():
            count = row['cnt'] + 1
            denom = self.bigram.loc[row['word1'] + " " + row['word2'], "cnt"] + len(self.unigram.index) - 1
            self.trigram.loc[index, 'MLE'] = log2(float(count)/denom)

    def print_ngram(self): # Anna
        """
        log2 the probability and save to new MLE column
        prints out each trigram with its logged MLE
        """
        # <w1> <w2> <w3> <log2(P(w3|w1 w2))>
        # highest to lowest prob, 3 decimal place rounded
        # then alphabetical
        # adapted from both bigram and unigram
        self.trigram.index.name = "index"
        self.trigram.sort_values(by=['MLE', "index"], ascending = [False, True], inplace = True)
        for index, row in self.trigram.iterrows():
    	    print(index, round(row['MLE'], 3))


    def train(self, train_corpus):
        file = open(train_corpus, 'r')
        self.read_data(file)
        file.close()
        self.train_unk()
        self.train_prob()
        self.print_ngram()

    def score_unk(self, sent): # Arshana
        # untested
        unked_sent = ""
        for w in sent.split():
            if w in self.unigram.index:
                unked_sent += w + " "
            else:
                unked_sent += "<UNK> "
        return unked_sent

    def score_prob(self, sent): # Anna
        pass

    def calc_perplex(self, sum, count): # Brynna
        pass

    def score(self, test_corpus): # Arshana
        # untested
        total_prob = 0
        num_sent = 0

        # read in file
        f = open(test_corpus, 'r')
        lines = f.readlines()
        f.close()

        # iterate through lines, outputting individual prob
        for line in lines:
            num_sent += 1
            unked_line = score_unk (re.sub(pattern=r'[^a-zA-Z0-9\s]', repl="", string=line))
            prob = score_prob(unked_line)
            total_prob += prob
            print(line + " " + str(prob))

        # determine perplexity
        perp = self.calc_perplex(total_prob, num_sent)
        print("Perplexity = " + str(perp))

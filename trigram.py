from math import log2
from numpy import tri
import pandas as pd
import re
from pathlib import Path
# !/usr/bin/python3

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
        lines = corpus.readlines()
        # extra start on first line
        entire_file = "<s> "
        for line in lines:
            line = line.strip()
            # get rid of most punctuation
            line = re.sub(pattern=r'[^a-zA-Z0-9\s]', repl="", string=line)
            # add beginning and end of sentence tokens
            entire_file += "<s> " + line + " </s> "
        # extra stop on last line
        entire_file += " </s>"

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
        filename1 = "trigram_df.csv"
        filename2 = "trigram_bi_df.csv"
        filename3 = "trigram_uni_df.csv"
        if Path(filename1).exists() and Path(filename2).exists() and Path(filename3).exists():
            self.trigram = pd.read_csv(filename1, index_col=0)
            self.bigram = pd.read_csv(filename2, index_col=0)
            self.unigram = pd.read_csv(filename3, index_col=0)
        else:
            file = open(train_corpus, 'r')
            self.read_data(file)
            file.close()
            self.train_unk()
            self.train_prob()
            self.trigram.to_csv(filename1)
            self.bigram.to_csv(filename2)
            self.unigram.to_csv(filename3)
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
        # not tested
        prob = 0
        sent_list = sent.split()
        for i in range(len(sent_list)):
            if (i + 2) < len(sent_list):
                bi_index = sent_list[i] + " " + sent_list[i+1]
                index = bi_index + " " + sent_list[i+2]
                MLE = 0
                if index in self.trigram.index:
                    MLE = self.trigram.loc[index, 'MLE']
                elif bi_index in self.bigram.index:
                    denom = self.bigram.loc[bi_index, "cnt"] + len(self.unigram.index) - 1
                    MLE = log2(1.0/denom)
                else:
                    denom = len(self.unigram.index) - 1
                    MLE = log2(1.0/denom)
                prob += MLE
        return prob

    def calc_perplex(self, sum, N): # Brynna
        H = (-1/N) * sum
        return round(2 ** H, 3)

    def score(self, test_corpus): # Arshana
        # untested
        total_prob = 0
        num_words = 0

        # read in file
        f = open(test_corpus, 'r')
        lines = f.readlines()
        f.close()

        # iterate through lines, outputting individual prob
        for i in range(len(lines)):
            line = lines[i]
            unked_line = self.score_unk(re.sub(pattern=r'[^a-zA-Z0-9\s]', repl="", string=line))

            if i == 0:
                prob_line = "<s> <s> " + unked_line + " </s>"
            else:
                prob_line = "</s> <s> " + unked_line + " </s>"
            if i != len(lines) - 1:
                prob_line += " <s>"

            # +1 for </s>
            num_words += len(unked_line.split()) + 1

            prob = self.score_prob(prob_line)
            total_prob += prob
            print(line.strip() + " " + str(prob))

        # determine perplexity
        perp = self.calc_perplex(total_prob, num_words)
        print("Perplexity = " + str(perp))


    def generate(self):
        # randomize for the first trigram with start token </s> <s> word or <s> <s> word
        # randomize top 3 probabilities
        # until it hits a stop token
        # print with no start/stop, add period

        # loop until creation of a valid sentence
        start_new_sentence = True
        while start_new_sentence:
            # change boolean to indicate we are now continuing an existing sentence
            start_new_sentence = False
            # df with all word2 as <s>, ignore UNK
            start = self.trigram[(self.trigram.word2 == "<s>") & (self.trigram.word3 != "<UNK>")]
            # randomize for one row
            start_row = start.sample()
            sent_list = ["<s>"]
            # retrieve word3 and add to sentence list
            word = start_row.word3.iloc[0]
            sent_list.append(word)
            # keep track of index in sentence list
            i = 1

            while word != "</s>" and not start_new_sentence:
                first = sent_list[i-1]
                # df with all word1 as first and word2 as word, ignore UNK
                trigrams = self.trigram[(self.trigram.word1 == first) & (self.trigram.word2 == word) & (self.trigram.word3 != "<UNK>")]
                # if trigrams is empty, we need to start over
                start_new_sentence = trigrams.index.size == 0
                # if trigrams isn't empty, add the next word
                if not start_new_sentence:
                    # finds rows of top 3 prob
                    top_3 = trigrams.nlargest(3, "MLE")
                    # randomizes for one row and retrieves word3, appending to sentence list
                    word = top_3.sample().word3.iloc[0]
                    sent_list.append(word)
                i += 1
        sent = " ".join(sent_list[1:len(sent_list)-1]) + "."
        print(sent)

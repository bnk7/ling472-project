# !/usr/bin/python3


# TODO: Implement a Laplace-smoothed bigram model :)
class LanguageModel:

    def __init__(self):
        pass

    def read_data(self, corpus): #Brynna
        # start and stop tokens
            # sentence on each line
            # or double dashes
        # two dataframes: unigram index, bigram as index and each individual word as a column
        pass

    def train_unk(self): # Anna
        pass

    def smoothing(self): # Arshana
	# for each w1 in vocab list
		# for each w2 in vocab list
			# if "<w1> <w2>" is not an index in df then add it to df with 0 occurrences
	# loop through df
		# add 1 to each ngram in df
        pass

    def probability(self): # Anna
        pass

    def calculate_MLE(self): # Arshana
	# loop through df
		# add a column with count / count(w2)
        pass

    def print_ngram(self): # Brynna
        pass

    def train(self, train_corpus):
        print('I am an unimplemented BIGRAM train() method.')  # delete this!

    def score(self, test_corpus):
        print('I am an unimplemented BIGRAM score() method.')  # delete this!

# Contributors: Anna Batra, Arshana Jain, Brynna Kilcline

## Instructions:

**Training:** `python main.py <training data path> -n <n-gram size>`

`<n-gram size>` is 1 for the unigram model, 2 for the bigram model, and 3 for the trigram model

**Testing:** `python main.py <training data path> -n <n-gram size> -t <testing data path>`

**Sentence generation:** `python main.py <training data path> -n 3 -g 3`

Sentence generation applies only for the trigram model.

**A note:** The program creates CSV files and saves them locally for efficiency. If you wish to change the training corpus, you must manually delete the CSVs before running.

## Introduction:

**Description of an N-gram model:**

This estimates the probability of the last word of an n-gram based on the previous (n - 1) words in the n-gram. It also assigns probabilities to entire n-gram sequences.
The model typically uses log probabilities to avoid numerical underflow.
The model uses Maximum Likelihood Estimation (MLE) which counts the number of instances in a corpus and then normalizes the number to something in the range [0,1].

**Smoothing:**

We implemented Laplace smoothing, in which you add one to each co-occurrence before calculating the MLE.

**Perplexity:**

Perplexity is a metric for the intrinsic evaluation of a language model. A smaller perplexity indicates a model that is better at predicting, due to it being the inverse probability of the dataset.

## Data Statement:
The data originally comes from the Jane Austen novels Emma, Persuasion, and Sense and Sensibility. Austen was a member of the English gentry in Hampshire. Her novels, a combination of realistic fiction and social commentary, were published in the 1810s and written in early 19th century British English.

We were assigned these texts with the goal of using the sentences to create n-gram models. Taken from e-books on Project Gutenberg’s website, the data was processed by NLTK and then by the instructors of LING 472. The sentences were shuffled across the books and 80% of the sentences were assigned to the training set, 10% to the development set, and 10% to the test set. Each dataset is formatted with one sentence per line, where each sentence comes pre-tokenized, with whitespace delimiting token boundaries.

## Results:

|| Perplexity for dev sets | Perplexity for test sets | Perplexity for train sets |
-|-|-|-
| Unigram | 555.179 | 559.895 | 582.655 |
| Bigram | 93.709 | 93.709 | 93.709 |
| Trigram | 4074.13 | 4114.151 | 2473.393 |

Our performance with perplexity on the test set peaked at 93.709 on the bigram. Our unigram came in second with 559.895. Our trigram’s perplexity was much higher at 4114.151. We find it strangely unique that our perplexity was the same across the board for bigram over the train, dev, and test sets.

## Error Analysis:

The following sentences were generated by the trigram language model. We generated these sentences by first selecting a random word that started a sentence and then from there building the sentence till we reached a stop token. Each consecutive word was randomly chosen from the three trigrams with the highest probability based on the previous two words.

1. on Tuesday as you are not going from Hartfield.
2. Early in February.
3. Jane s letter and every other being upon earth.
4. Goddard s.
5. Thomas s intelligence seemed over.
6. Michaelmas came and now it may all advise him against it but I was very well and happy was she Supposing even that particular riddle which you have not had a great many more people to be a very good natured woman after all.
7. Give him a little while ago it would not be a great regard for his own heart.
8. Alas.
9. poor Miss Taylor.
10. Married women you know it.
11. the coldness of a young lady s and that the living it was not a creature as his own.
12. Such was her own.
13. continued Mrs Smith.
14. Again it was all that had ever been at all.
15. Facts shall speak to her and I have no doubt of its doing them any pain.

A few of the sentences produced are syntactically and semantically grammatical. These include “Such was her own,” and “Again it was all that had ever been at all.” If a comma is added to indicate the women are the addressees, “Married women, you know it” is acceptable. Similarly, “Facts shall speak to her and I have no doubt of its doing them any pain” could make sense in a particular context. As exclamations or fragments of a sentence, outputs 2, 8, 9, and 13 are also well formed. 

Another few are syntactically but not semantically well formed. “Thomas's intelligence seemed over” is grammatical, and you can infer some kind of meaning from it, but it may need context as it may be otherwise ambiguous. Thomas’s intelligence could be done for, or over the roof. Similarly, “Jane's letter and every other being upon earth” is a grammatical noun phrase, but the classification of letters as beings does not make sense here and it is not a complete sentence.

The first type of syntactic error generated is in argument structure. Sometimes a head is paired with the wrong type of argument, such as in sentences 7 and 11. In sentence 7, “Give him a little while ago…”, the verb should take two noun phrase complements, but it is given a noun phrase and an adverb phrase instead. In sentence 11, “and” conjoins a noun phrase and a complementizer phrase, which is ungrammatical. Sometimes a head is missing an argument. This can be found in the missing main NPs after the possessors in sentences 4 (“Goddard s.”) and 11 (“lady s and”). Additionally, in sentence 1, there seems to be some kind of missing argument (CP) before the clause phrase, either before or after “on Tuesday.” A CP is also missing from sentence 13, “continued Mrs Smith.” The way the data was split into sentences clearly impacted this occurrence because it split dialogue from narration.

The second type is run-on sentences. Sentence 7 could be parsed as two clauses not linked by a conjunction. The phrases making up sentence 6 share so many arguments it’s almost impossible to make any sense of.

## References:
Jurafsky, D., & Martin, J. H. (2020). Speech and Language Processing (3rd ed.).

Southam, B. C. (2021). Jane Austen. In Britannica. Retrieved from Encyclopædia Britannica, inc. (n.d.). Jane Austen. Encyclopædia Britannica. https://www.britannica.com/biography/Jane-Austen. 

The influence of Jane Austen's social background on two of her novels. Jane Austen Centre and the Jane Austen Online Gift Shop. (n.d.). https://janeausten.co.uk/blogs/jane-austen-books-and-characters/the-influence-of-jane-austens-social-background-on-two-of-her-novels. 

Project Gutenberg. (n.d.). https://www.gutenberg.org/. 

Python packages used: Pandas, re, math, pathlib, numpy

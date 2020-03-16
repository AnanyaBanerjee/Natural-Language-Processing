# Part Of Speech Tagging with Naive Bayes

Please run the program: > POSTagging_with_NaiveBayes.py

It asks for your sentence which you need to type and press Enter.

Then the program displays POS Tags for words in the typed sentence along with its probability.

If any word in your sentence is not in the given corpus, then it gives an error.

**Output Files :**

- **File: 1** "unigram_postag.csv" contains the unigrams and their corresponding POS Tags.

- **File: 2** "bigram_wordtag_counts.csv" contains the bigrams[word, tag] and their corresponding counts.

- **File: 3** "bigram_tag_prevtag_counts.csv” contains the bigram[tag, prev_tag] and their corresponding counts.

- **File: 4** "unigram_tag_counts.csv” contains the tags and their corresponding counts.

- **File: 5** "bigram_word_given_tag_prob.csv” contains the bigram [word, tag] and their corresponding Probability P(word/tag)

- **File :6** "bigram_tag_given_prevtag_prob.csv" contains the bigram [tag, prev_tag] with their computer probability P(tag/prev_tag) 



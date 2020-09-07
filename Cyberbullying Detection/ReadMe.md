## ReadMe File of Cyber Bullying Detection Project

This project is a cyber bullying detection project which uses *Formspring
cyber bullying dataset*.

The code in **“cyberbullying_detection.py”** is divided into different
functions. Each function performs one specific task.

Here, a series of pre-processing steps have been done on the data such as
*removing stopwords, lemmatization, removing special characters,
punctuations, converting to lower case and solving class imbalance*.

Next, two kinds of models are used to capture features of the data: *bag of
words model (CountVectorizer)* and *tfidf model (TFIDFVectorizer)*.

Next, these are used with 5 algorithms such as **Logistic Regression**, **Random
Forest**, **Decision Tree**, **SVM** and **Neural Networks**.

This uses default **randomized hyperparameter optimization** for improving the
efficiency of the above models. However, you can opt for grid search or no
optimization as well.

We use **5 evaluation metrics**: *Precision, Recall, Accuracy, F1 score and the
classification report* to judge the quality of our trained models.

Post this, we also test our model on two test phrases:
Test Phrase: 1 “stupid bitch! fuck off!'”
Test Phrase: 2 “I am a normal girl!”

We find that our system correctly identifies the first as a comment intended to
bully someone while the second comment is deemed harmless.

**Procedure to run:**

Simply navigate to this folder and run the “cyberbullying_detection .py” file.
Please note that this needs libraries such as nltk, sklearn, numpy, pandas,
textblob, scipy and warnings to be installed on your pc, in order to run
accurately.

Thank you!

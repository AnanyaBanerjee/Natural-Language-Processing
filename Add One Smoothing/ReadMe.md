# Add One Smoothing

### Assumed Dependencies:

**Libraries:** numpy, pandas, runpy, nltk, os, sys, itertools, operator

**Python version:** Python 3.7.3

### How to run:

**Command:** Command : This program runs Program 2 which implements Add One Smoothing

> python3 homework2_AnanyaBanerjee.py add_one_smoothing


### Program 2:

**File:** add_one_smoothing.py

Running this file generates: 

- **File 1:**  "bigrams_add_one_smoothing.csv” contains the bigrams and their corresponding probabilities

- **File 2:** "unigrams_add_one_smoothing.csv” contains the unigrams and their corresponding probabilities

- **File 3:**  "counts_unigram_add_one_smoothing.csv " contains the unigrams and their corresponding counts in given corpus

- **File 4:** “counts_bigram_add_one_smoothing.csv " contains the bigrams and their corresponding counts in given corpus


You can open any of these files to have a look at the results.

My calculations of the given sentence “The standard Turbo engine is hard to work ” is as shown below:

### Bigrams:

['the', 'standard']
 ['standard', 'turbo']
 ['turbo', 'engine']
 ['engine', 'is']
 ['is', 'hard']
 ['hard', 'to']
 ['to', 'work']


**Add One Smoothing Calculations Example:**

V=28268

<pre>

bigram : ['the', 'standard'\]

Count(['the', 'standard'])=3

Count([‘the’])=3676

Prob1 = Count(['the', 'standard']) + 1 / Count([‘the’]) + V
          

</pre>         

<pre>

bigram : ['standard', 'turbo'\] 

Count(['standard', 'turbo'] )=2

Count([‘standard’])=10

Prob2 = Count(['standard', 'turbo'] )+ 1 / Count([‘standard’]) + V
         

           
</pre>


<pre>
bigram : ['turbo', 'engine'\]  

Count(['turbo', 'engine'])=0

Count([‘turbo’])=2

Prob3 = Count(['turbo', 'engine']) + 1 / Count([‘turbo’]) + V
          

</pre>         

<pre>

Count(['engine', 'is'])=0

Count([‘engine’])=17

Prob4 = Count(['engine', 'is']) + 1 / Count([‘engine’]) + V
          

</pre>

<pre>

bigram : ['is', 'hard'\] 

Count(['is', 'hard'])=0

Count([‘is’])=447

Prob5 = Count(['is', 'hard']) + 1 / Count([‘is’]) + V
         

</pre>
           



<pre>

bigram : ['hard', 'to'\]

Count(['hard', 'to'])=3

Count([‘hard’])= 4

Prob6 = Count(['hard', 'to']) + 1 / Count([‘hard’]) + V

</pre>
          

<pre>
bigram : ['to', 'work'\] 

Count(['to', 'work'])=7

Count([‘to’])= 1551

Prob7 = Count(['to', 'work']) + 1 / Count([‘to’]) + V

</pre>
          


<pre>
Prob= prob1 * prob2 * prob3 * prob4 * prob5 * prob6 * prob7
</pre>



# No Smoothing

### Assumed Dependencies:

**Libraries:** numpy, pandas, runpy, nltk, os, sys, itertools, operator

**Python version:** Python 3.7.3

### How to run:

**Command:** This program runs Program 1 which implements No Smoothing

> python3 program_AnanyaBanerjee.py no_smoothing

The probability for given sentence is printed on console output.

### Program 1:

**File:** no_smoothing.py

Running this file generates: 

- **File 1:** "bigrams_no_smoothing.csv” contains the bigrams and their corresponding probabilities

- **File 2:** "unigrams_no_smoothing.csv" contains the unigrams and their corresponding probabilities

- **File 3:** "counts_unigram_no_smoothing.csv" contains the unigrams and their corresponding counts in given corpus

- **File 4:** "counts_bigram_no_smoothing.csv" contains the bigrams and their corresponding counts in given corpus

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


**No Smoothing Calculations Example:**

<pre>
bigram : ['the', 'standard'\]


Count(['the', 'standard'])=3

Count([‘the’])=3676

Prob1 = Count(['the', 'standard']) / Count([‘the’])

</pre>         

<pre>
bigram : ['standard', 'turbo'\] 

Count(['standard', 'turbo'] )=2

Count([‘standard’])=10

Prob2 = Count(['standard', 'turbo'] ) / Count([‘standard’]) 

           
</pre>


<pre>
bigram : ['turbo', 'engine'\]  

Count(['turbo', 'engine'])=0

Count([‘turbo’])=2

Prob3 = Count(['turbo', 'engine']) / Count([‘turbo’]) 
</pre>         

<pre>
bigram : ['engine', 'is'\]

Count(['engine', 'is'])=0

Count([‘engine’])=17

Prob4 = Count(['engine', 'is'])  / Count([‘engine’]) 
</pre>

<pre>
bigram : ['is', 'hard'\] 

Count(['is', 'hard'])=0

Count([‘is’])=447

Prob5 = Count(['is', 'hard'])  / Count([‘is’]) 
</pre>
           



<pre>
bigram : ['hard', 'to'\]

Count(['hard', 'to'])=3

Count([‘hard’])= 4

Prob6 = Count(['hard', 'to']) / Count([‘hard’]) 
</pre>
          

<pre>
bigram : ['to', 'work'\] 

Count(['to', 'work'])=7

Count([‘to’])= 1551

Prob7 = Count(['to', 'work']) / Count([‘to’])
</pre>
          


<pre>
Prob= prob1 * prob2 * prob3 * prob4 * prob5 * prob6 * prob7
</pre>



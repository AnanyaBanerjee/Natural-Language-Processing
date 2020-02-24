# Good Turing Discounting

### Assumed Dependencies:

**Libraries:** numpy, pandas, runpy, nltk, os, sys, itertools, operator

**Python version:** Python 3.7.3

### How to run:

**Command:**: This program runs Program 3 which implements Good Turing Discounting

> python3 homework2_AnanyaBanerjee.py good_turing_discounting           


### Program 2:

**File:** good_turing_discounting.py

Running this file generates: 

- **File 1:** "bigrams_good_turing_discounting_smoothing.csv” contains the bigrams and their corresponding probabilities

- **File 2:** "unigrams_good_turing_discounting_smoothing.csv" contains the unigrams and their corresponding probabilities

- **File 3:** "unigrams_buckets_good_turing_discounting_smoothing.csv " contains the unigrams and their corresponding counts or buckets in given corpus

- **File 4:** "bigrams_buckets_good_turing_discounting_smoothing.csv " contains the bigrams and their corresponding counts or buckets in given corpus
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


**Good Turing Smoothing Calculations Example:**

N=66517

<pre>

bigram : ['the', 'standard'\]

Count(['the', 'standard'])=3

N4: len(bucket with val 4) = 1363

N3: len(bucket with val 3) = 1813

Count_star = [ (Count(['the', 'standard'])+1 ]*[N4/N3]
                     
Prob1 = Count_star / N  

</pre>         

<pre>

bigram : ['standard', 'turbo'\] 

Count(['standard', 'turbo'] )=2

N2: len(bucket with val 2) = 6376

N3: len(bucket with val 3) = 1813

Count_star = [(Count(['standard', 'turbo'] )+ 1 ]*[N3/N2]
            
Prob2 = Count_star / N

           
</pre>


<pre>
bigram : ['turbo', 'engine'\]  

Count(['turbo', 'engine'])=0

Bigram not in corpus!!

N1: len(bucket with val 1) = 16520

Prob3 = N1/ N


</pre>         

<pre>

bigram : ['engine', 'is'\] 

Count(['engine', 'is'])=0

Bigram not in corpus!!

N1: len(bucket with val 1) = 16520

Prob4 = N1/ N 


</pre>

<pre>

bigram : ['is', 'hard'\] 

Count(['is', 'hard'])=0

Bigram not in corpus!!

N1: len(bucket with val 1) = 16520

Prob5 = N1/ N
         

</pre>
           

<pre>

bigram : ['hard', 'to'\]

Count(['hard', 'to'])=3

N4: len(bucket with val 4) = 1363

N3: len(bucket with val 3) = 1813

Count_star = [(Count(['hard', 'to'] )+ 1 ]*[N4/N3]

Prob6 = Count_star / N

</pre>
          

<pre>

bigram : ['to', 'work'\] 

Count(['to', 'work'])=7

N7: len(bucket with val 7) = 205

N8: len(bucket with val 8) = 184

Count_star = [(Count(['to, 'work'] )+ 1 ]*[N8/N7]
                 
Prob7 = Count_star/ N


</pre>
          


<pre>

Prob= prob1 * prob2 * prob3 * prob4 * prob5 * prob6 * prob7

</pre>



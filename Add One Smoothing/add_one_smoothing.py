#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 12:46:03 2020

@author: ananyabanerjee

Add One Smoothing

"""


from itertools import chain
from nltk.util import ngrams
import pandas as pd
import numpy as np

file=open("NLP6320_POSTaggedTrainingSet-Unix.txt", "r")

obj=file.read()

#splitting on the basis of new line char "\n" into sentences
list_of_sentences=obj.split("\n")

print("Number of sentences are", len(list_of_sentences))

#create a dictionary to store count of each word
#key=word
#value= count of word in corpus for unigram
counts=dict()

#unigram dictionary
#key=word
#value= P(word)=Count(word)/N
unigram_dict=dict()

#bigram dictionary
#key=word
#value= P(word)
bigram_dict=dict()

#corpus of unigram
corpus_unigram=[]

#create a dictionary to store count of each word1 and word2 occuring together
#key=[word1, word2]
#value= count of word in corpus for bigram
counts_bi=dict()

#create a dictionary to store count of each word1 occuring t
#key=[word1]
#value= count of word in corpus for unigram
counts_uni=dict()

#dictionary counts_star for add one smooothing count for Unigram
#key=word
#val= C*(word) = (Count(word) +1)*(N/(N+V)
counts_star=dict()


#dictionary counts_star for add one smooothing count
#key=word
#val= C*(word) = (Count(word) +1)*(N/(N+V)
counts_star_bi=dict()

#bigram dictionary
#key=word
#value= P(word)
bigram_dict=dict()

#corpus is a list of all word tokens
corpus_bigram=[]


#function to tokenize each sentence
def tokenize(sent):
    #split each sentence on space or tab
    tokens_with_pos=sent.split(" ")
    tokens_with_pos.remove('')
    tokens_in_sent=[]
    pos_in_sent=[]
    
    for i in range(len(tokens_with_pos)):
        tokens=tokens_with_pos[i].split("_")[0]
        pos=tokens_with_pos[i].split("_")[1]
        tokens_in_sent.append(tokens)
        pos_in_sent.append(pos)
    
    #tokens=tokens_with_pos[0].split("_")[0]
    return tokens_in_sent

#function to create dict(word]=counts 
def create_counts_no_smoothing(corpus1, vocab1, type_of_model):
    if type_of_model=="unigram":
        print("unigram model")
        for x in corpus1:
            if x in list(counts_uni.keys()):
                counts_uni[x]+=1
            else:
                counts_uni[x]=1
        return counts_uni
    
    else:
        print("bigram model")
        for x in corpus1:
            print("x is", x)
            if str(x) in list(counts_bi.keys()):
                counts_bi[str(x)]+=1
            else:
                counts_bi[str(x)]=1
        return counts_bi

#function to create dict(word]=counts_star 
def create_counts_addone_smoothing(corpus, vocab):
    #dictionary counts_star for add one smooothing count
    #key=word
    #val= C*(word) = (Count(word) +1)*(N/(N+V)
    N=len(corpus)
    for x in vocab:
        #print("***x in ***", x)
        co=corpus.count(x)
        counts_star[str(x)]=(co+1)*(N/(N+V))

    return counts_star


#function to create unigram model
def create_unigram(vocab, counts, smoothing_type, N):
    
    if smoothing_type=="no smoothing":
        #P(word)=Count(word)/N
        for x in vocab:
            #print("counts is", counts_uni[x])
            unigram_dict[x]=counts_uni[x]/N
            
    elif smoothing_type=="add one smoothing":
        #C*(word) = (Count(word) +1)*(N/(N+V))
        #P(word)= (Count(word)+1)/(N+V) = C*(word)/N
        for x in vocab:
            #unigram_dict[x]=(counts[x]+1)/(N+V)
            unigram_dict[x]=counts_star[x]/N
        
    
    return 

#function to create bigram with addone smoothing
def create_bigrams_with_addone(corpus_bigram):
    #P(w2/w1)=Count(w1, w2) + 1 / Count(w1) +V 
    for w in corpus_bigram:
        cnt=counts_bi[str([w[0],w[1]])] +1
        cnt1=counts_uni[w[0]] + V
        prob=cnt/cnt1
        bigram_dict[str([w[1],w[0]])]=prob
        
    return

#function to find prob of add one smooting
def find_prob_add_one_smoothing(token_, type_of_model, V, N):
    prob=1
    
    #initialize the list of prob with length
    prob_=[]
    
    if type_of_model=='unigram':
        uni=list(counts_uni.keys())
        for w in token_:
            #is given unigram in corpus?
            if w in uni:
                cnt=counts_uni[w[0]]+1
                prob=cnt/N
            else:
                cnt=1
                prob=cnt/N
            prob_.append(prob)
        
    elif type_of_model=='bigram':
        bi=list(counts_bi.keys())
        uni=list(counts_uni.keys())
        for w in token_:
            #print("w is", w)
            #Is given bigram in your corpus?
            if str([w[0],w[1]]) in bi:
                cnt=counts_bi[str([w[0],w[1]])] +1
                cnt1=counts_uni[w[0]] + V
                prob=cnt/cnt1
                #print("w is",w, "and counts_bi",counts_bi[str([w[0],w[1]])], "and count of w[0]=", w[0], "is", counts_uni[w[0]] )
            else:
                #if not then its count(w[0],w[1])=0
                cnt=1
                #if the given word is in corpus?
                if w[0] in uni:
                    cnt1=counts_uni[w[0]] + V
                    #print("w has counts_bi=0",w,  "and count of w[0]=", w[0], "is", counts_uni[w[0]], "and V is", V )
            
                else:
                    #if the word w[0] not in corpus, then count(word[0])=0
                    cnt1=V
                    #print("w has counts_bi=0",w,  "and count of w[0]=", w[0], "is 0", "and V is", V )
            
                prob=cnt/cnt1
            #print("w is",w, "and prob is", prob)
            prob_.append(prob)
       
    print ("list is", prob_)
    #multiply all prob of given token to get final result
    prob=np.prod(prob_)
    return prob

#function to take a sentence and gives prob based on type pf bow model
def find_prob_of_sentence(sent, type_of_model, type_of_smoothing, V, N):
    #splitting the sentence into tokenz
    token_=sent.split(" ")
    #converting all tokens to lower case
    token_=list(map(str.lower,token_))
    
    #initialize prob
    prob=1
    
    if type_of_model=="unigram":
        key_=list(unigram_dict.keys())
        if type_of_smoothing=="no smoothing":
            for x in token_:
                if x in key_ or str(x) in key_:
                    prob=prob*unigram_dict[x]
                else:
                    prob=0
        elif type_of_smoothing=="add one smoothing":
            prob=find_prob_add_one_smoothing(token_, "unigram", V, N)
        else:
            #good turing discounting
            for x in token_:
                prob=prob*unigram_dict[x]
    
    elif type_of_model=="bigram":
        #converting given sentence to its bigram
        token_=list(ngrams(token_, 2))
        token_=[list(map(str.lower,list(x))) for x in token_]
        key_b=list(bigram_dict.keys())
        if type_of_smoothing=="no smoothing":
            for x in token_:
                if str(x) in key_b:
                    prob=prob*bigram_dict[str(x)]
                else:
                    prob=0
        elif type_of_smoothing=="add one smoothing":
                prob=find_prob_add_one_smoothing(token_, "bigram", V, N)
        else:
            #good turing discounting
            list_keys=list(counts_bi.keys())
            for x in token_:
                if x in list_keys:
                    bucket_no=list_keys[x]
                else:
                    bucket_no=0
                    
                prob=prob*bigram_dict[bucket_no]
    
    return prob


"""
UNIGRAM
"""


for sent in list_of_sentences:
    tokens_in_sent=tokenize(sent)
    corpus_unigram.append(tokens_in_sent)
    
#converting list of lists into one single list
corpus_unigram=list(chain.from_iterable(corpus_unigram))

#converting everything to lower case
corpus_unigram=list(map(str.lower,corpus_unigram))

#finding unique words
vocab=list(set(corpus_unigram))

#length of vocab=length of unigram corpus for bigram
V=len(vocab)

print("Length of corpus for unigram is", len(corpus_unigram) ,"and length of vocab_unigram is", len(vocab))


#counts with unigram C(word W1)
counts_uni=create_counts_no_smoothing(corpus_unigram, vocab, "unigram")

#counts_star with unigram
counts_star=create_counts_addone_smoothing(corpus_unigram, vocab)

#create unigram model with no smoothing
create_unigram(vocab, counts, "add one smoothing", len(corpus_unigram))



"""
BIGRAMS
"""



for sent in list_of_sentences:
    tokens_in_sent=tokenize(sent)
    #bigram model
    output = list(ngrams(tokens_in_sent, 2))
    #converting everything to lower case
    output=[list(map(str.lower,list(x))) for x in output]
    #print("ouput", output)
    corpus_bigram.append(output)


#converting list of lists into one single list
corpus_bigram=list(chain.from_iterable(corpus_bigram))

#length of corpus N
N=len(corpus_bigram)
   
#removing duplicate words in a corpus makes it unique bigram
unique_corpus_bigram = [list(x) for x in set(tuple(x) for x in corpus_bigram)]

#initialization of counts dictionary
#counts with bigram C(word W1, word W2)
counts_bi=create_counts_no_smoothing(corpus_bigram, unique_corpus_bigram, "bigram")

print("Length of corpus for bigram is", len(corpus_bigram) ,"and length of vocab_bigram is", len(unique_corpus_bigram))

#counts_star with bigram
#counts_star_bi=create_counts_addone_smoothing(corpus_bigram, unique_corpus_bigram)

#create bigram with no smoothing
create_bigrams_with_addone(corpus_bigram)


#finding probability of given sentence
sentence="The standard Turbo engine is hard to work"

#unigram + no smoothing
prob_of_sentence=find_prob_of_sentence(sentence, "bigram", "add one smoothing", len(unique_corpus_bigram), N)

print("Your Sentence is ", sentence)
print("Probability of the given sentence is", prob_of_sentence)


"""
Saving file

"""
#saving bigrams into file
col=['Bigram','Probability']
df=pd.DataFrame(columns=col)
df['Bigram']=list(bigram_dict.keys())
df['Probability']=list(bigram_dict.values())
df.to_csv("bigrams_add_one_smoothing.csv", index=False)

#saving bigrams with their counts into file
col2=['Bigram','Counts']
df2=pd.DataFrame(columns=col2)
df2['Bigram']=list(counts_bi.keys())
df2['Counts']=list(counts_bi.values())
df2.to_csv("counts_bigram_add_one_smoothing.csv", index=False)


#saving unigrams into file
col1=['Unigram','Probability']
df1=pd.DataFrame(columns=col1)
df1['Unigram']=list(unigram_dict.keys())
df1['Probability']=list(unigram_dict.values())
df1.to_csv("unigrams_add_one_smoothing.csv",index=False)

#saving unigrams with their counts into file
col3=['Unigram','Counts']
df3=pd.DataFrame(columns=col3)
df3['Unigram']=list(counts_uni.keys())
df3['Counts']=list(counts_uni.values())
df3.to_csv("counts_unigram_add_one_smoothing.csv", index=False)

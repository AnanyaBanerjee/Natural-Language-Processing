#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 13:14:07 2020

@author: ananyabanerjee

NLP Assignment 3

POS Tagging with Naive Bayes
"""
import pandas as pd
from itertools import chain
from nltk.util import ngrams
import ast 
import itertools
import numpy as np

file=open("NLP6320_POSTaggedTrainingSet-Unix.txt", "r")

obj=file.read()

#splitting on the basis of new line char "\n" into sentences
orig_list_of_sentences=obj.split("\n")

print("Number of sentences are", len(orig_list_of_sentences))

#adding start token <s> and end token </s>
#list_of_sentences=[ '<s>_START '+x+' </s>_END' for x in list_of_sentences]
list_of_sentences=[ '<s> '+x+' </s>' for x in orig_list_of_sentences]


#dict to store word and their possible tags
#key=word
#value=list of tags
pos_tags=dict()


#dict to store counts of [word, tag]
#key=[word, tag]
#value=counts
counts_word_tag=dict()

#dict to store counts of [tag_i, tag_i-1]
#key=[tag_i, tag_i-1]
#value=counts
counts_tag_i_tag=dict()

#dict to store counts of tag_i]
#key=tag_i
#value=counts
counts_tag=dict()

#dictionary to store prob
#key=[word, tag]
#value= P(w/t)
prob_word_tag=dict()

#dictionary to store prob
#key=[tagi, tagi-1]
#value= P(ti/ti-1)
prob_tag=dict()



#list of tags as per Penn Tree Bank
list_of_tags=['CC',	'CD', 'DT', 'EX','FW', 'IN',
              'JJ','JJR','JJS','LS','MD','NN',
              'NNS','NNP','NNPS','PDT','POS','PRP',
              'PRP$','RB','RBR', 'RBS','RP','SYM',
              'TO','UH','VB','VBD','VBG','VBN',
              'VBP','VBZ','WDT','WP','WP$','WRB','$','#',
              '"','"', '(',')', ',','.', ':', '<s>', '</s>'] 
              

#corpus_bigram_word is a list of all [word, tag] bigrams
corpus_bigram_word=[]

#corpus_bigram_tag is a list of all [tag_i, tag_i-1] bigrams
corpus_bigram_tag=[]

#corpus_unigram_word is a list of all word tokens
corpus_unigram_word=[]

#corpus_unigram_tag is a list of all POS tags
corpus_unigram_tag=[]

#corpus sentences after making pos tags and words seperately
corpus_sent=[]


#function to find pos tags for given words in corpus
def find_pos_tags():
    print("Generating POS Tags")
    #dict: pos_tags
    #key=word
    #word=list of pos tags
    for sent in orig_list_of_sentences:
        #sent=sent.replace('_', ' ').replace(', ', ' ').split()
        sent=sent.split()
        for i in range(len(sent)):
            li=sent[i].split("_")
            
            #Case 1: li: [word, Pos tag]
            #example: ["NP", "Jonny"]
           
            if li[0] in list_of_tags:
                # first ele is pos tag
                keys=pos_tags.keys()
                
                if li[1] in keys or str(li[1]) in keys:
                    pos_tags[li[1]].append(li[0])
               
                else:
                    pos_tags[li[1]]=[li[0]]
             
                    
            #Case 2: li: [Pos tag, word]
            #example: ["Jonny", "NP"]
            
            else:#if li[0] not in list_of_tags:
                # first ele is word
                keys=pos_tags.keys()
                if len(li)==2:
                    if li[0] in keys or str(li[0]) in keys:
                        pos_tags[li[0]].append(li[1])
                   
                    else:
                        pos_tags[li[0]]=[li[1]]
                
            
    #removing duplicates
    for x in list(pos_tags.keys()):
        li=pos_tags[x]
        li = list(set(li)) 
        pos_tags[x]=li
    
    return

#function to create ngram where n=1,2 for each sentence
def create_ngram():
    
    for sent in list_of_sentences:
        sent=sent.replace('_', ' ').replace(', ', ' ').split()
        corpus_sent.append(sent)
        
        #unigram model
        #add it to list
        corpus_unigram_word.append(sent)
        #bigram model
        output_bi = list(ngrams(sent, 2))
        #converting list of tuples into list of lists
        output_bi=[ list(x) for x in output_bi]
        #get first ele
        first=output_bi[0]
        #get last ele
        last=output_bi[len(output_bi)-1]
        #removing spurious data
        for x in output_bi:
            if x[0] in list_of_tags:
                output_bi.remove(x)
          
        #inserting first and last to output_bi
        output_bi.insert(0,first)
        output_bi.insert(len(output_bi),last)
        
        
        #adding to corpus_bigram
        corpus_bigram_word.append(output_bi)
        #tag is a empty list
        tag=[]
        for x in sent:
            if x in list_of_tags:
                tag.append(x)
            corpus_unigram_tag.append(tag)
            
        #adding all tags to bigram list
        #bigram model
        output_tag = list(ngrams(tag, 2))
        #converting list of tuples into list of lists
        output_tag=[ list(x) for x in output_tag]
        #adding output_tag to bigram tag
        corpus_bigram_tag.append(output_tag)
        

      
#function to create counts 
def create_counts():
    print("Creating Bigram model")
    #calculate count of [tag_i, tag_i-1]
    for x in corpus_bigram_tag[0]:
        keys=list(counts_tag_i_tag.keys())
        if str(x) not in keys:
            counts_tag_i_tag[str(x)]=1
        else:
            counts_tag_i_tag[str(x)]+=1

    #calculate count of [word, tag]
    for x in corpus_bigram_word[0]:
        keys=list(counts_word_tag.keys())
        #print (x, str(x))
        if x not in keys:
            counts_word_tag[str(x)]=1
        else:
            counts_word_tag[str(x)]+=1

    #calculate count of tag
    for x in corpus_unigram_tag[0]:
        keys=list(counts_tag.keys())
        if x not in keys:
            counts_tag[str(x)]=1
        else:
            counts_tag[str(x)]+=1


#function to find probability P(w/t) and P(ti/ti-1)
def find_prob():
    print("Finding Prob of P(w/t)")
    #find prob of word w given tag t: P(w/t)
    #P(w/t)=Counts(t, w)/ Counts(t)
    keys_word_tag=list(counts_word_tag.keys())
    for x in keys_word_tag:
        #print("x is", x, " str(x)", str(x))
        x = ast.literal_eval(x) 
        #print("+", x,"  ", x[0], " ", x[1])
        #Case 1: x[0]=pos tag, x[1]=word
        if x[0] in list_of_tags and x[1] not in list_of_tags:
            #counts(t,w)= counts_word_tag[str([t,w])]
            #Counts(t)= counts_tag[t]
            counts_t_w=counts_word_tag[str(x)]
            counts_t=counts_tag[x[0]]
            #prob(w/t)
            prob=counts_t_w/counts_t
            prob_word_tag[str(x)]=prob
            #print("I was here")
        else:
            #Case 2: Case 1: x[0]=word, x[1]=pos tag
            #counts(t,w)= counts_word_tag[str([t,w])]
            #Counts(t)= counts_tag[t]
            if(len(x)==2 and x[1] in list_of_tags):
                counts_t_w=counts_word_tag[str(x)]
                counts_t=counts_tag[x[1]]
                #prob(w/t)
                prob=counts_t_w/counts_t
                prob_word_tag[str(x)]=prob
            #print("I am not here")
        
        
    print("Finding Prob of P(ti/ti-1)")
    #find prob of word w given tag t: P(ti/ti-1)
    #P(ti/ti-1)=Counts(ti-1, ti)/ Counts(ti-1)
    keys_tag=list(counts_tag_i_tag.keys())
    for x in keys_tag:
        #print("x is", x, " str(x)", str(x))
        x = ast.literal_eval(x) 
        #print("hehe", x,"  ", x[0], " ", x[1])
        if x[0] in list_of_tags and x[1] in list_of_tags:
            #ounts(ti-1, ti)= counts_word_tag[str([ti-1,ti])]
            #Counts(ti-1)= counts_tag[ti-1]
            counts_ti_tmin=counts_tag_i_tag[str(x)]
            counts_t=counts_tag[x[0]]
            #prob(w/t)
            prob= counts_ti_tmin/counts_t
            prob_tag[str(x)]=prob
 
    
    
#function to create all combination of word and pos tag
def create_combo(token_tag):
    words=list(token_tag.keys())
    possible_tags=list(token_tag.values())
    
    combinations=list(itertools.product(*possible_tags))
    #dictionary
    #key= combination of pos tags
    #value= word list
    combo=dict()
    
    for i in range(len(combinations)):
        combo[str(list(combinations[i]))]=words
        
    return combo
    
   
#function to calculate naive bayes prob
def calc_prob_NB(combo):
    #dict
    #key= tag seq
    #val= prob
    ARG_Prob=dict()
    
    #for each combo
    for x in list(combo.keys()):
        #get word token list
        word=combo[x]
        #x contains the POS Tag 
        x = ast.literal_eval(x) 
        #print("For Tag Sequence", x)
        #P(word/tag)
        prob_word_given_tag=[]
        #keys of prob_word_tag
        keys_=list(prob_word_tag.keys())
        #calculating P(word/tag)
        for i in range(len(word)):
            if str([word[i],x[i]]) in keys_:
                #prob=prob_word_tag[str([word[i],x[i]])]
                prob=prob_word_tag[str([word[i],str(x[i])])]
            else:
                prob=0
            prob_word_given_tag.append(prob)
         
        #print("P1: prob of word given pos tag", prob_word_given_tag)
        #P(tag i/ tag i-1 )
        #example: P(tag2/tag1)
        prob_tag_given_tag_prev=[]
        #make bigrams of x
        list_bigram_x=list(ngrams(x, 2))
        #add P(tag1/<s>) and P(</s>/tag_last) to bigram of tags list
        fi1=['<s>',x[0]]
        fi2=[x[len(x)-1],'</s>']
        list_bigram_x.insert(0,fi1)
        list_bigram_x.append(fi2)
        #
        list_bigram_x=[list(x) for x in list_bigram_x]
        #keys of prob_tag
        keyss_=list(prob_tag.keys())
        #calculating P(tag i /tag i-1)
        for i in range(len(list_bigram_x)):
           if str(list_bigram_x[i]) in keyss_:
               #print("poppy")
               prob=prob_tag[str(list_bigram_x[i])]
           else:
               #print("no poppy")
               prob=0
           prob_tag_given_tag_prev.append(prob)
         
        # print("P2: prob of tag given prev tag", prob_tag_given_tag_prev)
        #finding final prob 
        final_prob1=np.prod(prob_word_given_tag)
        final_prob2=np.prod(prob_tag_given_tag_prev)
        
        PROB=final_prob1*final_prob2
        ARG_Prob[str(x)]=PROB
        
    return ARG_Prob
        
    
    

#function to create naive bayesian classification based POS Tagging
def create_naive_bayes_for_POSTagging(sent):
    #tokens of given sentence
    tokens=sent.split()
    #get corresponding probable POStags for each token
    token_tag=dict()
    
    list_of_tags=list(pos_tags)
    for x in tokens:
        if x not in list_of_tags:
            print("Error, word '", x, "'not in voacb. Please Try Again")
            exit(1)
            #quit()
        list_=pos_tags[x]
        list_=list(set(list_))
        token_tag[x]=list_
        
    #for all combo of token tags, calculate their prob acc to Naive Bayes
    #combo has key=combination of pos tags
    #combo has val=word list
    combo=create_combo(token_tag)
    
    #for each combo of token tags, find prob
    ARG_Prob=calc_prob_NB(combo)
    
    #print("Choosing arg max of the prob:", ARG_Prob)
    #list of pos tagging
    pos_final_list=[]
    
    #get its arg max
    list_of_prob=list(ARG_Prob.values())
    
    #get max value of prob
    max_=max(list_of_prob)
    
    #get corresponding key value from ARG_Prob dictionary
    for x in list(ARG_Prob.keys()):
        if ARG_Prob[x]==max_:
            #print("Yay got it")
            pos_final_list=x
            break
    
    
    pos_tag=pos_final_list
    prob=max_
    
    
    return pos_tag, prob, tokens


#sentence
#sentences="John went to work ."
sentences=input("Enter the sentence for POS Tagging by Naive Bayes ")


#find pos tags
find_pos_tags()

#create ngrams
create_ngram()

#corpus_unigram_tag is a list of all POS tags
#converting list of lists into one single list
corpus_unigram_tag=[list(chain.from_iterable(corpus_unigram_tag))]


#converting list of lists into one single list
corpus_bigram_tag=[list(chain.from_iterable(corpus_bigram_tag))]


#converting list of lists into one single list
corpus_bigram_word=[list(chain.from_iterable(corpus_bigram_word))]

#creating counts
create_counts()

#finding prob of all P(w/t) and P(t_i/t_i-1)
find_prob()


#get best POS Tagging and Corresponding prob
pos_tag, prob, tokens=create_naive_bayes_for_POSTagging(sentences)

pos_tag=ast.literal_eval(pos_tag)

print("The POS tag chosen is", pos_tag, " and prob=", prob) 

for i in range(len(tokens)):
    print("The POS Tag for ", tokens[i], " is", pos_tag[i])
    



"""
Saving file

"""
#dict to store word and their possible tags
#key=word
#value=list of tags

#saving unigrams with their pos tag into file
col=['Word','POS_Tags']
df=pd.DataFrame(columns=col)
df['Word']=list(pos_tags.keys())
df['POS_Tags']=list(pos_tags.values())
df.to_csv("unigram_postag.csv", index=False)

#dict to store counts of [word, tag]
#key=[word, tag]
#value=counts

#saving bigrams [word, tag] with their coounts into file
col=['[word, tag]','Counts']
df=pd.DataFrame(columns=col)
df['[word, tag]']=list(counts_word_tag.keys())
df['Counts']=list(counts_word_tag.values())
df.to_csv("bigram_wordtag_counts.csv", index=False)

#dict to store counts of [tag_i, tag_i-1]
#key=[tag_i, tag_i-1]
#value=counts

#saving bigrams [tag, prev_tag] with their counts into file
col=['[tag, prev_tag]','Counts']
df=pd.DataFrame(columns=col)
df['[tag, prev_tag]']=list(counts_tag_i_tag.keys())
df['Counts']=list(counts_tag_i_tag.values())
df.to_csv("bigram_tag_prevtag_counts.csv", index=False)


#dict to store counts of tag_i]
#key=tag_i
#value=counts


#saving given Tag i with their counts into file
col=['Tag','Counts']
df=pd.DataFrame(columns=col)
df['Tag']=list(counts_tag.keys())
df['Counts']=list(counts_tag.values())
df.to_csv("unigram_tag_counts.csv", index=False)


#dictionary to store prob
#key=[word, tag]
#value= P(w/t)


#saving bigrams [word, tag] with their Porb P(w/t) into file
col=['[word, tag]','Probability']
df=pd.DataFrame(columns=col)
df['[word, tag]']=list(prob_word_tag.keys())
df['Probability']=list(prob_word_tag.values())
df.to_csv("bigram_word_given_tag_prob.csv", index=False)


#dictionary to store prob
#key=[tagi, tagi-1]
#value= P(ti/ti-1)

#saving bigrams [tag, prev_tag] with their prob P(ti/ti-1) into file
col=['[tag, prev_tag]','Probability']
df=pd.DataFrame(columns=col)
df['[tag, prev_tag]']=list(prob_tag.keys())
df['Probability']=list(prob_tag.values())
df.to_csv("bigram_tag_given_prevtag_prob.csv", index=False)

#

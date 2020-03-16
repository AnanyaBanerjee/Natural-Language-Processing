#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 21:54:29 2020

@author: ananyabanerjee

NLP: Viterbi Algorithm
"""

import pandas as pd
import numpy as np

"""
#Reading Observation Likelihood Prob from file
"""
xl_file_obv = pd.ExcelFile("Observation_Likelihood_prob.xlsx")

dfs_obv = {sheet_name: xl_file_obv.parse(sheet_name) 
          for sheet_name in xl_file_obv.sheet_names}

#dataframe containing Observation Likelihood
df_obv=dfs_obv['Sheet1']

#get tag sequence for obv likelihood prob
tag_seq_obv=list(df_obv['Unnamed: 0'])

#dropping a col
df_obv.drop(['Unnamed: 0'],axis=1)

#all columns in obv file
col_obv=list(df_obv.columns)
col_obv.remove('Unnamed: 0')

#dict for observation likelihood prob
#key=[tag, word]
#val=obv_prob
obv_lhood=dict()

for i in range(len(tag_seq_obv)):
    for j in range(len(col_obv)):
        y=list(df_obv[col_obv[j]])
        if str([tag_seq_obv[i],col_obv[j]]) not in list(obv_lhood.keys()):
            obv_lhood[str([tag_seq_obv[i],col_obv[j]])]= y[i] 
       
        
        
"""
#Reading Transition Prob from file
"""
xl_file_trans = pd.ExcelFile("transition_prob.xlsx")

dfs_trans = {sheet_name: xl_file_trans.parse(sheet_name) 
          for sheet_name in xl_file_trans.sheet_names}

#dataframe containing transition prob
df_trans=dfs_trans['Sheet1']

#get tag sequence for transition prob
tag_seq_trans=list(df_trans['Unnamed: 0'])

#dropping a col
df_trans.drop(['Unnamed: 0'],axis=1)

#all col in trans prob file
col_trans=list(df_trans.columns)
col_trans.remove('Unnamed: 0')

#dict for transition prob
#key=[tag, word]
#val=obv_prob
trans_prob=dict()

for i in range(len(tag_seq_trans)):
    for j in range(len(col_trans)):
        y=list(df_trans[col_trans[j]])
        if str([ col_trans[j], tag_seq_trans[i]]) not in list(trans_prob.keys()):
            trans_prob[str([ col_trans[j], tag_seq_trans[i]])]= y[i]     

#Viterbi Algorithm
      
#viterbi matrix: POSTag s v/s time T matrix
#V_i=[[0]*len(col_trans)]*len(col_trans)
V_i=[[0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0]]

#backtrace matrix dict
back_trace=[]

#function to initialize viterbi algo
def initialize_viterbi(obj1):
    #Initialization step: ==> state=POStag, obv=word
    #At time T=1
    T=1
    
    #pi values for transition prob are of the form
    # "['NNP', '<s>']"
    
    #get obv1 for obv_lhood prob b_s(obv1) 
    #obv1=col_obv[0]
    
    v=[]
    back_=[]
    #print("Initializing for", obj1)
    
    for s in tag_seq_obv:
        
        # pi_s is transition prob for state s initially
        pi_s=trans_prob[str([s, '<s>'])]
       
        #obv likelihood for obv1="Janet" and corresponding state s
        b_obv_s=obv_lhood[str([s, obj1])]
        
        #int_vit= pie_s * b_obv_s
        int_vit=pi_s*b_obv_s
       
        #print("s is", s, " ", "pi(s)",pi_s, " b(s)",  b_obv_s, "fin",int_vit)
       
        #add it to V_i
        v.append(int_vit)
        
        #initialize backtrace
        bp=0
        
        #append it to back trace matrix
        back_.append(bp)
    
    V_i[0]=v  
    back_trace.append(back_)
    
    
#function to find viterbi[s.t] for calculating max
#arguments: time: t, postag: tag_seq_obv[s]
def find_viterbi_for_calc_max(t, s, obj):
    t=t-1
    #calculate viterbi[s,t] and add it for finding max_ list
    max_=[]
    cur_tag=tag_seq_obv[s]
    #print("\n, Str is", str([cur_tag, obj]))
    #print("Obv", obv_lhood, "['NNP', 'the ']"==str([cur_tag, obj]) )
    #print("cur tag", cur_tag, type(cur_tag))
    #print("obj is", obj, type(obj))
    #print("lis is",[cur_tag, obj] )
    #print("str of lis is",str([cur_tag, obj]))
    
    obs_prob=obv_lhood[str([cur_tag, obj])]
    
    for i in range(len(tag_seq_obv)):
        V=V_i[t][i]
        transition_prob=trans_prob[str([cur_tag, tag_seq_obv[i]])]
        fin=V*transition_prob*obs_prob
        max_.append(fin)
        
    return max_

#function to perform recursion
def perform_recursion_step(words):
    
    #Time T  
    T=len(words)
    
    #calculate Viterbi V_t(j)
    for t in range(1,T):
        obj=words[t]
        #print("Object is", obj)
        for s in range(len(tag_seq_obv)):
            max_list=find_viterbi_for_calc_max(t, s, obj)
            #find max of Viterbi[s,t]
            V_i[t][s]=max(max_list)
            #find arg max of backpointer[s,t]
            back_trace.append(max_list.index(max(max_list)))
    

#function to perform termination step
def perform_termination_step():  
    #find best score
    best_score=[]
    
    #iterate over number of words
    for i in range(len(V_i)):
        #for each word, get a list of tags taken from Viterbi
        #and find its max
        list_of_postags=V_i[i]
        max_=max(list_of_postags)
        #get index of max
        ind_of_max=list_of_postags.index(max_)
        #find max of scores in tag_i
        best_score.append(col_trans[ind_of_max])
         
    #displaying results
    for i in range(len(col_obv)):
        print("Pos Tag for",col_obv[i], " is", best_score[i] )

   

####Initialization Step####



#### Ask for Sentence Input ####
sentence=input("Enter the sentences : ")
 
sentence_tokens=sentence.split(' ')

#initialize for the viterbi algorithm
initialize_viterbi(sentence_tokens[0])

####Recursion Step####
perform_recursion_step(sentence_tokens)

####Termination Step####
perform_termination_step()
  
















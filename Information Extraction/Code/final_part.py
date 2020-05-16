#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 20:04:40 2020

@author: ananyabanerjee

https://becominghuman.ai/text-summarization-in-5-steps-using-nltk-65b21e352b65
https://www.analyticsvidhya.com/blog/2019/09/introduction-information-extraction-python-spacy/
https://towardsdatascience.com/nlp-text-preprocessing-a-practical-guide-and-template-d80874676e79

https://nlp.stanford.edu/software/openie.html

Try: Topic Identification using LDA< bow or tfidf or glove , doc similarity, coref resol, text summarization, 

Task 1: Implement a deep NLP pipeline to extract the following NLP based features
from the text articles/documents:

o Split the document into sentences

o Perform dependency parsing or full-syntactic parsing to get parse-tree based

o Tokenize the sentences into words

o Removing Stop words

o Removing digits??

o Remove Special Characters

o Lemmatize the words to extract lemmas as features

o NER

o Remove html tags

o Lower case all

o Part-of-speech (POS) tag the words to extract POS tag features
patterns as features

o Using WordNet, extract hypernymns, hyponyms, meronyms, AND holonyms as
features

o Some additional features that you can think of, which may make your
representation better
"""


#get files from the folder "train"
import os  
import nltk
import re
from nltk.tokenize import sent_tokenize, word_tokenize, WordPunctTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.parse import CoreNLPParser
from nltk.corpus import wordnet as wn
from bs4 import BeautifulSoup
import spacy
import en_core_web_sm
import itertools
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
import re
import numpy as np
import string
from itertools import combinations
import json
from geotext import GeoText

#all files
all_files = os.listdir("Dataset/")   # imagine you're one directory above test dir
print(all_files)

#files data
#key=file name
#value= data inside the file
files_data=dict()

#reading the file
for file_name in all_files:
    #print("reading file", file_name)
    if file_name!=".DS_Store":
        try:
            f = open("Dataset/"+file_name, mode="r", encoding="utf-8")
            data=f.read()
            files_data[file_name]=data
        
        except:
            print("\n, ** Could not open file " + file_name)

      
#function to combine word vectors to form a sentence vector by creating min, max and  coordinate wise mean vector
def combine_vector(word_list, emb_keys,word_embeddings ):
    #y=['more', 'than', '40,000']
    #emb=[word_embeddings['more'],word_embeddings['than'],word_embeddings['40,000']]
    
    emb=[]
    for x in word_list:
        if x not in emb_keys:
            emb.append(out_of_vocab(x))
        else:
            emb.append(word_embeddings[x])
    
    min_emb=[]
    max_emb=[]
    mean_emb=[]
    
    for i in range(len(emb[0])):
        lis=[]
        for j in range(len(emb)):
            lis.append(emb[j][i])
        #min_emb.append(min(emb[0][i], emb[1][i], emb[2][i]))
        #max_emb.append(max(emb[0][i], emb[1][i], emb[2][i]))
        min_emb.append(min(lis))
        max_emb.append(max(lis))
        mean_emb.append(np.mean(lis))
        
    return min_emb, max_emb, mean_emb


#function to deal with Out of Vocab words
def out_of_vocab(word):
    embedding_oov = np.zeros((300,))
    return embedding_oov


#function to find sentences with the seeds
def find_sentence_with_seedtuple(seed,dict_lines_for_each_para):
    lines=list(dict_lines_for_each_para.values())
    lines=[item for sublist in lines for item in sublist]

    
    line_with_seed=[]
    
    for x in lines:
        #print("seed[0]", seed[0],"seed[0] in x", seed[0] in x)
        if seed[0] in x or seed[0].lower() in x:
            if seed[1] in x or seed[1].lower() in x:
                line_with_seed.append(x)
            
    
    return line_with_seed


#function to remove stopwords and punctuations from context words
def preprocess_context(list_, stop_words):
    #remove special char
    w=[token for token in list_ if token.isalnum()]
    #lemmatize words
    w=[WordNetLemmatizer().lemmatize(y) for y in w]
    #remove stop words
    w = [token for token in w if token or token.lower() not in stop_words]
    
    t=[]
    for x in w:
        if x not in stop_words or x.lower() not in stop_words:
            t.append(x)
            
    #remove empty string
    w = [s for s in t if s]
    #remove punctuation from string list
    w=[x for x in w if x not in list(string.punctuation)]
    
    return w

#function to create a context vector
def create_context_vec(seed, lines_of_seed, stop_words, emb_keys,word_embeddings):
   
    #context_vector for all lines
    context_between=[]
    context_before_m1=[]
    context_after_m2=[]
    
    #words in between context
    between=[]
    #words in before m1 context
    before=[]
    #words in after m2 contect
    after=[]
    
    #start: M1
    start=seed[0]
    #end: M2
    end=seed[1]
    
    for s in lines_of_seed:
        #definition of context1
        # words between both mentions
        context_str=s[s.find(start)+len(start):s.rfind(end)]
        #get words from context string
        context_word=context_str.split(" ")
        context_word=preprocess_context(context_word, stop_words)
        if len(context_word)==0:
           context_between.append([]) 
           between.append([])
        else:
            #for all context words, get glove embedding and combine its embedding to create a vector
            min_emb, max_emb, mean_emb= combine_vector(context_word,emb_keys,word_embeddings )
            #append the context vector to its list for each line containing seed
            context_between.append(mean_emb) 
            between.append(context_word)
        
        ##definition of context2
        # words before mention1
        before_str=s.partition(start)[0]
        before_word=before_str.split(" ")
        before_word=preprocess_context(before_word, stop_words)
        
        if len(before_word)==0:
            context_before_m1.append([])   
            before.append([])
        
        else:
            #for all context2 words, get glove embedding and combine its embedding to create a vector
            min_emb_2, max_emb_2, mean_emb_2= combine_vector(before_word,emb_keys,word_embeddings)
            #append the context vector to its list for each line containing seed
            context_before_m1.append(mean_emb_2)   
            before.append( before_word)
            
        ##definition of context3
        # words after mention2
        after_str=s.partition(end)[2]
        after_word=after_str.split(" ")
        after_word=preprocess_context(after_word, stop_words)
        if len(after_word)==0:
           context_after_m2.append([]) 
           after.append([])
        else:
            #for all context3 words, get glove embedding and combine its embedding to create a vector
            min_emb_3, max_emb_3, mean_emb_3= combine_vector(after_word,emb_keys,word_embeddings)
            #append the context vector to its list for each line containing seed
            context_after_m2.append(mean_emb_3) 
            after.append(after_word)
            
    
    return context_between, context_before_m1, context_after_m2, before, after, between


#function to generalize the context
def generalize_context(context_word, named_entities):
    
    
    #before generalizing context, remove ners
    for i in range(len(context_word)):
        if context_word[i] in named_entities:
            context_word.remove(context_word[i])
            
    #making a substring
    k=[]
    for i in range(len(context_word)):
        b=' '.join(word for word in context_word[i])
        k.append(b)
            
    
    return context_word, k
            
   
#https://towardsdatascience.com/understanding-feature-engineering-part-4-deep-learning-methods-for-text-data-96c44370bbfa
    
#function to extract patterns from context vector
def extract_pattern( context, context_substr):
    #r"\b[a-zA-z]{1,}\b bought \b[a-zA-z]{1,}\b"
    pat=[]
    for con in context:
        for c2 in con:
            c1=r'\b[a-zA-z]{1,}\b '
            c3=r' \b[a-zA-z]{1,}\b'
            p=c1+c2+c3
            pat.append(p)
            
    pat_str=[]
    for con in context_substr:
        c1=r'\b[a-zA-z]{1,}\b '
        c3=r' \b[a-zA-z]{1,}\b'
        p=c1+con+c3
        pat_str.append(p)
        
        
    return pat, pat_str
        
    
    
#function to generate seeds from generated patterns
def generate_seed_from_patterns(pattern_,dict_lines_for_each_para):
    #look for pattern matches in all sentences
    #all lines
    lines=list(dict_lines_for_each_para.values())
    lines=[item for sublist in lines for item in sublist]
    
    #get all lines with the given pattern_
    lines_with_pattern=[]
    
    for x in lines:
        mat=re.search(pattern_, x)
        if mat:
           lines_with_pattern.append(x)
           
    #generate the seed tuples
    #1. Identify ner in lines_with_pattern
    #we only want ner which can show relation buy
    ner_we_want=['PER','ORG','FAC','PRODUCT']
    
    seed_tuples=[]
    nlp= en_core_web_sm.load()
    
    for y in lines_with_pattern:
        doc=nlp(y)
        
        #ne_=[(X.text, X.label_) for X in doc.ents]
        ne_=[]
        entities=[]
        for X in doc.ents:
            if X.label_ in ner_we_want:
                ne_.append((X.text, X.label_))
                entities.append(X.text)
                
        print("Named Entities", ne_)
        #for all such named entities, get seed tuples
        seed_=list(combinations(entities, 2))
        seed_tuples.append(seed_)
        
        
    
    return lines_with_pattern, seed_tuples
        
    
#function to check confidence scores
def calc_confidence_score(origin_seed, line_containing_seed, patt, pat_str, seeds,dict_lines_for_each_para  ):
    scores=[]
    lines=list(dict_lines_for_each_para.values())
    lines=[item for sublist in lines for item in sublist]
    
    #for calculating score, see if the patterns extracted occur for each seeds
    #for every extracted pattern
    for pattern_ in patt:
        #check how many lines contain that pattern
        no_of_lines=0
        for x in lines:
            mat=re.search(pattern_, x)
            if mat:
               no_of_lines+=1
        
        print("\n,No of lines containing pattern", pattern_, " is", no_of_lines)
        score=float(no_of_lines)/float(len(seeds))
        print("The score is", score)
        if score>=0.5:
            patt.remove(pattern_)
        
        scores.append(score)
    
    print("\n")
    return patt, pat_str
        

#function to refine patterns by removing named entities, digits or Nouns incorrectly determined as relations
def refine_pattern(patt, named_entities):
    
    for p in patt:
        pi=p.split(" ")
        #print("**",nltk.pos_tag([pi[1]]),nltk.pos_tag([pi[1]])[0][1]=='NNP', nltk.pos_tag([pi[1]])[0][1]=='NNS' )
        if pi[1] in named_entities:
            patt.remove(p)
        elif pi[1].isdigit():
            patt.remove(p)
        elif nltk.pos_tag([pi[1]])[0][1]=='NNS':
            patt.remove(p)
        elif nltk.pos_tag([pi[1]])[0][1]=='NNP' :
            patt.remove(p)
            
    return patt
        
   
#function to write to a json file
def write_to_json(LINES_FINAL):
    #writing to a json file
    
    
    #Json Dictionary
    json_dict=dict()
    
    """
    {
    	"document": "Amazon_com.txt",
        "extraction": [
    		{
    			"template": "BUY",
    			"sentences": ["In 2017, Amazon acquired Whole Foods Market for US$13.4 billion, which vastly increased Amazon's presence as a brick-and-mortar retailer."],
    			"arguments": {
    				"1": "Amazon",
    				"2": "Whole Foods Market",
    				"3": "US$13.7 billion",
    				"4": "",
    				"5": ""
    			}
    		},
    		{
    			"template": "PART",
    			"sentences": [ "Amazon was founded by Jeff Bezos in Bellevue, Washington, in July 1994."],
    			"arguments": {
    				"1": "Bellevue",
    				"2": "Washington"
    			}
    		}
    	]
    }
    """
    for doc in list(LINES_FINAL.keys()):
        # {key:value mapping} 
        json_dict["document"]=doc
        ext=[]
        c=LINES_FINAL[doc]
        for pat_ in list(c.keys()):
            dict_=dict()
            dict_["template"]="PART"
            for line in c[pat_]:
                dict_["sentences"]=line
                arg=dict()
                m=re.search(pat_,line)
                arguments=list(range(2))
                #arguments=len(m.group(0))
                i=0
                for x in arguments:
                    arg[str(x)]=m.group(0).split(" ")[i]
                    i+=2
                dict_["arguments"]=arg
            ext.append(dict_)
        
        json_dict["extraction"]=ext
        # conversion to JSON done by dumps() function 
        #b = json.dumps(json_dict,indent=3) 
          
        # saving the object b which is json object
        with open("Output/part_i_"+doc+".json", "w") as write_file:
            json.dump(json_dict, write_file, indent=4)

  
#function to find all lines carrying a pattern
def find_lines_carrying_a_pattern(pat,dict_lines_for_each_para):
    
    lines=list(dict_lines_for_each_para.values())
    lines=[item for sublist in lines for item in sublist]
    
    lines_carrying=[]
    
    for x in lines:
        m=re.search(pat,x)
        if m:
           lines_carrying.append(x)
            
    return lines_carrying
    
#function to write to a json file
def write_json_files_geo(doc_name, rels_tuples):

    #rels_tuples
    #rels_tuples=[sub, obj, x]
    
    print("@@@ Writing into json file")
    #writing to a json file
    #Json Dictionary
    json_dict=dict()
    
    """
    {
    	"document": "Amazon_com.txt",
        "extraction": [
    		{
    			"template": "BUY",
    			"sentences": ["In 2017, Amazon acquired Whole Foods Market for US$13.4 billion, which vastly increased Amazon's presence as a brick-and-mortar retailer."],
    			"arguments": {
    				"1": "Amazon",
    				"2": "Whole Foods Market",
    				"3": "US$13.7 billion",
    				"4": "",
    				"5": ""
    			}
    		},
    		{
    			"template": "PART",
    			"sentences": [ "Amazon was founded by Jeff Bezos in Bellevue, Washington, in July 1994."],
    			"arguments": {
    				"1": "Bellevue",
    				"2": "Washington"
    			}
    		}
    	]
    }
    """
    #rels_tuples
    #rels_tuples=[sub, obj, x]
    
    # {key:value mapping} 
    json_dict["document"]=doc_name
    #extraction list
    ext=[]
    for i in range(len(rels_tuples)):
        dict_=dict()
        dict_["template"]="PART"
        dict_["sentences"]=rels_tuples[i][2]
        arg=dict()
        arguments=list(range(2))
        s=[rels_tuples[i][0], rels_tuples[i][1]]
        #arguments=len(m.group(0))
        j=0
        for x in arguments:
            arg[str(x)]=s[j]
            j+=1
        dict_["arguments"]=arg
        ext.append(dict_)
    
    json_dict["extraction"]=ext
    # conversion to JSON done by dumps() function 
    #b = json.dumps(json_dict,indent=3) 
      
    # saving the object b which is json object
    with open("Output/geo_part_"+doc_name+".json", "w") as write_file:
        json.dump(json_dict, write_file, indent=4)


def extract_part_reln(dict_lines_for_each_para):
    
    lines=list(dict_lines_for_each_para.values())
    lines=[item for sublist in lines for item in sublist]
    if [] in lines:
        lines.remove([])

    nlp=spacy.load("en_core_web_sm")
    #dependency parse
    
    #check if root is not a verb
    #find 'bought' 
    #and check if its children are nsubj and dobj, the you are done
    #else: 
    #    # check if there is a nsubj in the children
    #        ## 
    #    #else check if there is a dobj in the children
    #       ## check if root is NN or NNP or NNS and nsubj=ROOT
    #    #else: return nothing
    lis=['situated','located']
    #reln tuple is [ sub, obj, x]
    reln_tuple=[]
    
    for x in lines:
        doc=nlp(x)
        cod=list(doc.sents)
        roots=[f.root for f in cod]
        for token in doc:
            #if the word is in lis
            if token.text in lis:
                #get its children from dep parse
                children=[child for child in token.children]
                sub="none"
                obj="none"
                for c in children:
                    if c.dep_=='nsubj':
                        sub=str(c)
                    elif c.dep_=='dobj':
                        obj=str(c)
                #if we get both sub and obj then break
                if sub!="none" and obj!="none":
                    print("yay done", sub,token.text, obj)
                    r=[sub, obj, x]
                    reln_tuple.append(r)
                    break
                #if we only get object then check for subject in roots
                #if any root is a NN or NNP or NNS, then make that subject
                if obj!="none" and sub=="none":
                    for root in roots:
                        if root.pos_=='NNP' or root.pos_=='NNS' or root.pos_=='NN':
                            sub=root
                            r=[sub, obj, x]
                            reln_tuple.append(r)
                            break
                
                #if both obj and subj is not none
                if sub!="none" and obj!="none":
                    r=[sub, obj, x]
                    reln_tuple.append(r)
    
              
    #plac contains [part1, part2, line]
    plac=[]
    
    for x in lines:
        places=GeoText(x)
        if places.cities and places.countries:
            plac.append([places.cities,places.countries, x])
        elif places.cities:
             v=list(places.country_mentions)
             for z in places.cities:
                 plac.append([z, v[0], x])
             
    #appending plac and reln_tuple together
    for x in plac:
        reln_tuple.append(x)
        
    #try comma seperated NN, NN detection as well
    #gpe, gpe 
    GPE=[]
    for li in lines:
        doc=nlp(li)
        geo=[]
        for p in doc.ents:
            if p.label_=='GPE':
                geo.append(p.text)
        #if you have 2 geo entities
        if len(geo)==2:
            #check if they are comma seperated in sentence
            new_str=geo[0]+", "+geo[1]
            if new_str in li:
                GPE.append([geo[0],geo[1], li])
                
    for x in GPE:
        reln_tuple.append(x)
    
    print("reln is", reln_tuple)
    #remove duplicates
    reln_tuple = list(set(tuple(sub) for sub in reln_tuple)) 
    reln_tuple=[list(x) for x in reln_tuple]            
    
    return reln_tuple
    
        
  

def perform_relationship_extraction(data, seeds, file_name,word_embeddings, emb_keys):
    
    """#o Split the document into sentences and remove html tags"""
    
    #reading paragraphs
    paragraphs=data.split("\n\n")
    
    #
    no_of_para=len(paragraphs)
    num=list(range(no_of_para))
    
    #reading all lines in each paragraphs
    #dictionary where key is para number and value is list of lines
    dict_lines_for_each_para=dict()
    
    for x in num:
        key='p'+str(x)
        p=paragraphs[x]
        #val=re.split(r"\.|\?|\!", p)
        val=sent_tokenize(p)
        #removing html tags if any
        val=[BeautifulSoup(se, "lxml").text for se in val]
        dict_lines_for_each_para[key]=val
        
    
    reln_tuples=extract_part_reln(dict_lines_for_each_para)
    write_json_files_geo(file_name, reln_tuples)
    
    
    
    """ o Perform dependency parsing or full-syntactic parsing to get parse-tree based """ 
    """
    #parsing dictionary: key=para, value=list of parse tree corresponding to each sentence
    dict_parsetree_for_para=dict()
    
    parser = CoreNLPParser(url='http://localhost:9000', tagtype='pos')
    
    vals=list(dict_lines_for_each_para.values())
    keys=list(dict_lines_for_each_para.keys())
    
    for i in range(len(vals)):
        key=keys[i]
        l=[]
        for y in vals[i]:
            (pf, )=parser.raw_parse(y)
            l.append(pf)  
        dict_parsetree_for_para[key]=l
    """     
    
    """
    
    o Named Entity Recognition
    
    """
    #dictionary where
    #key=paragraph number and val=Named Entities in that para
    NER=dict()
    
    nlp = en_core_web_sm.load()
    
    para_no=list(dict_lines_for_each_para.keys())
    
    for x in para_no:
        ner=[]
        for y in dict_lines_for_each_para[x]:
            doc=nlp(y)
            ner1=[(X.text, X.label_) for X in doc.ents]
            ner.append(ner1)
            if [] in ner:
                ner.remove([])
        NER[x]=ner
        
        
    """ o Tokenize the sentences into words """
    """ o Lemmatize the words to extract lemmas as features """  
    """ o Removing Stop words and lower case convertion"""
    """ o Removing SPecial char"""
    
    stop_words = nltk.corpus.stopwords.words('english')
    
    """
    #dictionary: key=paragraphs number, val=list of sent tokens
    dict_token_for_para=dict()
    
    vals=list(dict_lines_for_each_para.values())
    keys=list(dict_lines_for_each_para.keys())
    
    #Using WordPunctTokenizer
    tk = WordPunctTokenizer() 
    
    for i in range(len(keys)):
        key=keys[i]
        val=vals[i]
        l=[]
        for x in val:
            #w=word_tokenize(x)
            w=tk.tokenize(x)
            #remove stop words and convert to lower case
            w = [token.lower() for token in w if token not in stop_words]
            #remove special char
            w=[token for token in w if token.isalnum()]
            #lemmatize words
            w=[WordNetLemmatizer().lemmatize(y) for y in w]
            l.append(w)
        
        dict_token_for_para[key]=l
        
    """
    """o Part-of-speech (POS) tag the words to extract POS tag features"""
    """
    #dictionary: key=paragraphs number, val=list of postags for tokens
    dict_postag_for_para=dict()
    
    #https://www.geeksforgeeks.org/nlp-wordnet-for-tagging/?ref=rp
    
    vals_w=list(dict_token_for_para.values())
    keys_w=list(dict_token_for_para.keys())
    
    for i in range(len(keys_w)):
        key=keys_w[i]
        val=vals_w[i]
        l=[]
        for x in val:
            pos=nltk.pos_tag(x)  
            pos=[list(y) for y in pos]
            l.append(pos)
        
        dict_postag_for_para[key]=l
    
    """
    
   
    
    
       
    """ o Using WordNet, extract hypernymns, hyponyms, meronyms, AND holonyms as
    features
    """
    """
    #https://blog.xrds.acm.org/2017/07/power-wordnet-use-python/
    #https://medium.com/parrot-prediction/dive-into-wordnet-with-nltk-b313c480e788
    #https://vprusso.github.io/blog/2018/natural-language-processing-python-4/
    #https://www.geeksforgeeks.org/nlp-synsets-for-a-word-in-wordnet/
    #https://www.geeksforgeeks.org/get-synonymsantonyms-nltk-wordnet-python/?ref=rp
    
    synonyms = [] 
    antonyms = [] 
    
    for syn in wn.synsets("good"): 
        for l in syn.lemmas(): 
            synonyms.append(l.name()) 
            if l.antonyms(): 
                antonyms.append(l.antonyms()[0].name()) 
    
    print("\n, synonyms", set(synonyms)) 
    print("\n, antonyms",set(antonyms))
    """
    
    
    
    """
    Getting all unique tokens
    
    """
    """
    k=list(dict_token_for_para.keys())
    tokens=set()
    
    for x in k:
        tok=dict_token_for_para[x]
        merged = list(itertools.chain.from_iterable(tok))
        for y in merged:
            tokens.add(y)
    """        
    """
    Getting all sentences
    
    """
    """
    k=list(dict_lines_for_each_para.keys())
    lines=[]
    
    for x in k:
        tok=dict_lines_for_each_para[x]
        #merged = list(itertools.chain.from_iterable(tok))
        for y in tok:
             lines.append(y)
    
    #lines=list(lines)
    """
            
    """
    Tf-IDF Model
    https://machinelearningmastery.com/prepare-text-data-machine-learning-scikit-learn/
    
    """
    """
    vectorizer = TfidfVectorizer()
    
    #vectors = vectorizer.fit_transform([documentA, documentB])
    #vectors = vectorizer.fit_transform(tokens)
    vectors = vectorizer.fit_transform(lines)
    
    feature_names = vectorizer.get_feature_names()
    
    dense = vectors.todense()
    
    denselist = dense.tolist()
    
    df = pd.DataFrame(denselist, columns=feature_names)
    
    df.to_csv("n_gram_tfidf_"+file_name+".csv", index=False)
    
    print("\n, N grams: n of features=", len(list(df.columns)), " and shape", df.shape )
    """
    """
    BOW Model, Bag of Words Bigram
    https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
    
    """
    """
    vectorizer1 = CountVectorizer(analyzer='word',ngram_range=(2, 2))
    
    vectors1 = vectorizer1.fit_transform(lines)
    
    feature_names1 = vectorizer1.get_feature_names()
    
    dense1 = vectors1.toarray()
    
    denselist1 = dense1.tolist()
    
    df1 = pd.DataFrame(denselist1, columns=feature_names1)
    
    df1.to_csv("bag_of_words_bigram_"+file_name+".csv", index=False)
    
    print("\n, BOW : n of features=", len(list(df1.columns)), " and shape", df1.shape )
    
    """
    """
    Relation Ship Extraction
    
    """
    
    
    """
    
    Look at	the	context	between	or around the pair	
    and	generalize the context to create	 patterns
    
    Features to extract context:
    1. Get glove  vectors for your words from word_embeddings
    2. Head words of both mentions
    3. Words between two mentions
    4. ner of both mentions
    5. Base syntactic chunk seq
    6. Dependency path
    
    """
    
        
    """
    -----------------
    Combining Vectors:
    -------------------    
    compute the vector for each word 
    in the document, and then 
    aggregate them using the 
    coordinate-wise mean, min, or max.
    
    """
    """
    Gather	a	set	of	seed	pairs	that	have	relation	R
    
    •Iterate:
        1. Find	sentences	with	these	pairs
        2. Look	at	the	context	between	or	around	the	pair	
        and	generalize	 the	context	to	create	patterns
        3. Use	the	patterns	 for	grep for	more	pairs
    """
    
   
    
    
    
        
    #making a list of Named Entities
    n=list(NER.values())
    n1=[item for sublist in n for item in sublist]
    n2=[item for sublist in n1 for item in sublist]
    n3=set(n2)
    named_entities_list=list(n3)
    named_entities=list(list(zip(*named_entities_list))[0]) 
    
    # Bootstrapping
    
    #Initialization
    #key=seed
    #value=lines containing seed
    line_with_seed=dict()
    for x in seeds:
        line_with_seed[x]=[]
        
    SEEDS=[]
    PATTERNS=[]
    print("\n,generating patterms")
    for seed in seeds:
        ls=find_sentence_with_seedtuple(seed, dict_lines_for_each_para)
        line_with_seed[seed].append(ls)
        context_between, context_before_m1, context_after_m2, before, after, between=create_context_vec(seed,ls, stop_words,emb_keys,word_embeddings)
        context_fin, context_substr=generalize_context(between, named_entities)    
        pat, pat_str=extract_pattern(context_fin, context_substr)   
        
        #calculate confidence of new seed tuples
        pat, pat_str=calc_confidence_score(seed, line_with_seed, pat, pat_str, seeds, dict_lines_for_each_para )
        #refine patterns
        pat=refine_pattern(pat,named_entities)
        print("gen,", pat)
        #generate new seed tuples
        for p in pat:
            lines_with_pattern, new_seed=generate_seed_from_patterns(p,dict_lines_for_each_para)
            new_seed=[item for sublist in new_seed for item in sublist]
            SEEDS.append(new_seed)
            PATTERNS.append(p)
    
    
    
    #refine patterns and seeds
    PATTERNS=set(PATTERNS)
    PATTERNS=list(PATTERNS)
    SEEDS=[item for sublist in SEEDS for item in sublist]
    SEEDS=set(SEEDS)
    SEEDS=list(SEEDS)
    
    ##key=pattern
    #value= lines containing that pattern
    file_lines=dict()
    
    for po in PATTERNS:
        l=find_lines_carrying_a_pattern(po,dict_lines_for_each_para)
        if l==[]:
            PATTERNS.remove(po)
        else:
            file_lines[po]=l
    
    return PATTERNS, SEEDS, file_lines
    





 
""" Glove Word Embedding for all words"""

print("\n,Generating Word Embeddings")
# Extract word vectors in dict where word=key and val =300d vector
word_embeddings = {}
f = open('glove.6B.300d.txt', encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    word_embeddings[word] = coefs
f.close()
   
emb_keys=list(word_embeddings.keys()) 

PATTERNS_FINAL=[]
SEEDS_FINAL=[]
LINES_FINAL=dict() #key=doc name #val is dict

print("\n, Starting to generate patterns")


for file_name in list(files_data.keys()):
    #file_name='Amazon_com.txt'
    data=files_data[file_name]
    
    file_name=file_name.replace('.txt','')
    
    print("+++ File Processing", file_name)
    
    #Relationship Seeds
    
    #BUY
    #PART
    #WORK
    
    #Template #1:
    #BUY(Buyer, Item, Price, Quantity, Source)
    
    #Template #2:
    #WORK(Person, Organization, Position, Location)
    
    #Template #3:
    #PART(Location, Location)
    
    
    seeds=[("Kentucky","U.S."),
           ("Sinking Spring Farm","Hodgenville, Kentucky"),
           ("Hingham","sNorfolk"),
           ("Cupertino","California"),
           ("Nebraska","United States"),
           ("Norcross","Georgia"),
           ("China","East Asia"),
           ("Yellow River","North China Plain"),
           ("Himalayas","India"),
           ("Mount Everest"," Sino-Nepalese border"),
           ("Ayding Lake","Turpan Depression"),
           ("China","Asia"),
           ("Dallas","United States"),
           ("Dallas","North Texas"),
           ("Fort Worth","East Texas")]
           #("",""),
           #("",""),
           #("",""),
           #("",""),
           #("",""),
           #("",""),
           #("","")]
    
    PT, Final_Seeds,File_lines=perform_relationship_extraction(data, seeds, file_name,word_embeddings,emb_keys) 
    PATTERNS_FINAL.append(PT)
    SEEDS_FINAL.append(Final_Seeds)
    print ("\n, Pat",PT)
    print ("\n, See",Final_Seeds)
    LINES_FINAL[file_name]=File_lines 
    


    
PATTERNS_FINAL=[item for sublist in PATTERNS_FINAL for item in sublist]
PATTERNS_FINAL=set(PATTERNS_FINAL)
PATTERNS_FINAL=list(PATTERNS_FINAL)

SEEDS_FINAL=[item for sublist in SEEDS_FINAL for item in sublist]
SEEDS_FINAL=set(SEEDS_FINAL)
SEEDS_FINAL=list(SEEDS_FINAL)

col1=['PATTERNS']
col2=['SEED1', 'SEED2']

df_pat=pd.DataFrame(PATTERNS_FINAL, columns=col1)
#df_pat.to_csv("PATTERNS_part.csv",index=False)
    
df_seed=pd.DataFrame(SEEDS_FINAL, columns=col2)
#df_seed.to_csv("SEEDS_part.csv",index=False)
    
write_to_json(LINES_FINAL)


"""
Deleting unnecessary files and combining output
"""


for doc_name in all_files:
    doc_name=doc_name.replace('.txt','')
    
    with open('Output/geo_part_'+doc_name+'.json') as f2: 
      f2data = f2.read() 
    
    with open('Output/part_i_'+doc_name+'.json') as f1: 
      f1data = f1.read() 
     
    f1data += "\n"
    f1data += f2data
    
    with open ('Output/part_'+doc_name+'.json', 'a') as f3: 
      f3.write(f1data)
    
    if os.path.exists('Output/geo_part_'+doc_name+'.json'):
      os.remove('Output/geo_part_'+doc_name+'.json')
    
    if os.path.exists('Output/part_i_'+doc_name+'.json'):
      os.remove('Output/part_i_'+doc_name+'.json')
    








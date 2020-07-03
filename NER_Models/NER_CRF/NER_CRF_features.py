#Feature for sumerian language
import nltk
import numpy as np
import pandas as pd
import re


def features(sentence,index):
    

    ### sentence is of the form [w1,w2,w3,..], index is the position of the word in the sentence
   
    #Dictionary of features
    d={}
    
    word = sentence[index]
    left = ""
    right = ""
    if (index!=0):
        left=sentence[index-1]
    if (index!=len(sentence)-1):
        right=sentence[index+1]
        
    
    ###### ------- Rules/Features to Identify POS Tags  -------#######
    
    d['Is_ON_1'] = 1 if re.search(r'{gesz}',word) else 0
    
    d['Is_ON_2'] = 1 if re.search(r'{gi}',word) else 0
    
    d['Is_ON_3'] = 1 if re.search(r'{tug2}',word) else 0
    
    d['Is_ON_4'] = 1 if re.search(r'{munus}',word) else 0
    
    d['Is_ON_5'] = 1 if re.search(r'{u2}',word) else 0
    
    d['Is_ON_6'] = 1 if re.search(r'{kusz}',word) else 0
    
    d['Is_ON_7'] = 1 if re.search(r'{uruda}',word) else 0 
    
    d['Is_first_word'] = 1 if index==0 else 0

    d['Is_last_word'] = 1 if index==len(sentence)-1 else 0

    d['previous_word'] = "" if left=="" else left

    d['next_word'] = "" if right=="" else right

    d['distorted_word_1'] = 1 if re.search(r'x',word) else 0

    d['distorted_word_2'] = 1 if re.search(r'\.\.\.',sentence[index]) else 0

    d['Is_number_form_1'] = 1 if re.search(r'\d+\(.+\)',word) else 0
    
    d['Is_number_inword'] = 1 if re.search(r'\d',word) else 0
    
    d['Is_Verb'] = 1 if re.search(r'a\d',word) else 0
    
    d['Is_Verb_1'] = 1 if re.search(r'a-a',word) else 0
    
    d['Is_Verb_2'] = 1 if re.search(r'z-z',word) else 0
        
    d['Is_child_left'] = 1 if left=='dumu' else 0
    
    d['Is_child_right'] = 1 if right=='dumu' else 0
    
    d['Is_place'] = 1 if left=='ki' or right=='ki' else 0
    
    d['Is_witness_1'] = 1 if left=='igi' or right=='igi' else 0
    
    d['Is_witness_2'] = 1 if left=='igi' or right=='igi' and word.endswith('sze3') else 0
    
    d['Is_selling_tablet'] = 1 if left=='kiszib3' or right=='kiszib3' else 0
    
    d['Is_behalf_business'] = 1 if left=='giri3' or right=='giri3' else 0
    
    d['Is_selling_tablet'] = 1 if left=='kiszib3' or right=='kiszib3' else 0
    
    d['Is_selling_tablet'] = 1 if left=='kiszib3' or right=='kiszib3' else 0
    
    d['Is_PN_1'] = 1 if word.startswith('ur-') else 0
    
    d['Is_PN_2'] = 1 if word.startswith('lu2-') else 0
    
    d['Is_PN_3'] = 1 if word.endswith('-mu') else 0
    
    d['Is_PN_4'] = 1 if re.search(r'{d}',word) else 0
    
    d['Is_SN'] = 1 if re.search(r'{ki}',word) else 0
    
    d['Is_determinative'] = 1 if re.search(r'{',word) else 0
    
    d['Is_PN_5'] = 1 if re.search(r'lugal',word) else 0
    
    d['contains_numer'] = 1 if re.search(r'\d',word) else 0
    
    d['Is_PN_6'] = 1 if right=='sag' else 0
    
    d['Is_PN_7'] = 1 if right=='zarin' else 0
    
    #d['Is_quantity'] = 1 if re.search(r'\d+\(\w+\)',left) else 0
    
    d['Is_month'] = 1 if left=='iti' else 0
    
    d['Is_mont_year'] = 1 if word=='iti' or word=='mu' else 0
    
    d['Is_PN_8'] = 1 if word[0].isupper() else 0
    
    d['Is_PN_9'] = 1 if word.startswith('{d}') else 0
    
    d['Is_hyphen'] = 1 if re.search(r'-',word) else 0
    
    d['prefix_1'] = word[0]
    
    d['prefix_2'] = word[:2]
    
    d['prefix_3'] = word[:3]
    
    d['prefix_4'] = word[:4]
    
    d['suffix_1'] = word[-1]
    
    d['suffix_2'] = word[-2:]
    
    d['suffix_3'] = word[-3:]
    
    d['suffix_4'] = word[-4:]
    
    return d
    
if __name__=='__main__':
    print("Features/Rules for sumerian POS tagging")

    


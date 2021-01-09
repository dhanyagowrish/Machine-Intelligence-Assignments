#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import random


def entropy_helper(col):
    entropy=0
    category,count=np.unique(col,return_counts=True)
    
    inter_entropy=[]
    for i in range(len(category)):
        inter_entropy.append((-count[i]/np.sum(count))*np.log2(count[i]/np.sum(count)))
    
    entropy=np.sum(inter_entropy)
    return entropy


def get_entropy_of_dataset(df):
    target=df.iloc[:,-1]
    entropy = entropy_helper(target)
    return entropy



def get_entropy_of_attribute(df,attribute):
    entropy_of_attribute = 0
    category,count= np.unique(df[attribute],return_counts=True)
    
    category_entropy=[]
    for i in range(len(category)):
        data=df.where(df[attribute]==category[i]).dropna()
        df_i=data.iloc[:,-1]
        cat_ent=entropy_helper(df_i)
        category_entropy.append((count[i]/np.sum(count))*cat_ent)
    
    entropy_of_attribute=np.sum(category_entropy)
    
    return abs(entropy_of_attribute)



def get_information_gain(df,attribute):
    information_gain = 0
    
    category,count= np.unique(df[attribute],return_counts=True)
    entropy_dataset=get_entropy_of_dataset(df)
    average_information=get_entropy_of_attribute(df,attribute)
    
    information_gain=entropy_dataset-average_information
    
    return information_gain



def get_selected_attribute(df):
   
    information_gains={}
    selected_column=''
    
    df_i=df.iloc[:,:-1]
    
    for cols in df_i.columns:
        information_gains[cols]=get_information_gain(df,cols)
    
    sel_col = max(information_gains, key=information_gains.get)
    return (information_gains,sel_col)


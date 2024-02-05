#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import sys

module_path='/Users/DAHS/Desktop/ECP_CONT/ECP_SCL/Cohort_selection/'
if module_path not in sys.path:
    sys.path.append(module_path)
    
from cohort_loader import data_split
from sklearn.manifold import TSNE


def integrating(data_path, emb_path_trn, emb_path_vld, mode):
    
    dataset = pd.read_csv(data_path, compression='gzip', index_col = 0)
    
    if mode == 'eicu':
        dataset = dataset.rename(columns = {'SaO2_fillna':'SpO2_fillna'})
        dataset = dataset[~(dataset['gender']==2)].reset_index(drop=True)
        save_load = np.load(emb_path_trn)
        num_columns = len(save_load[0])
        column_names = ['emb_{}'.format(i+1) for i in range(num_columns)] 
        embedding = pd.DataFrame(save_load, columns = column_names)
        del save_load
        
        emb_integ = pd.concat([dataset, embedding], axis = 1)
        
        return emb_integ
        
    else:    

        train, valid = data_split(dataset, 9040, 0.7, Threshold=0.05, n_trial=1, mode = mode)
        
        trn_save_load = np.load(emb_path_trn)
        vld_save_load = np.load(emb_path_vld)
        
        num_columns = len(trn_save_load[0])
        column_names = ['emb_{}'.format(i+1) for i in range(num_columns)]
        trn_embedding = pd.DataFrame(trn_save_load, columns = column_names)
        del trn_save_load
        
        trn_emb_integ = pd.concat([train, trn_embedding], axis = 1)
        
        vld_embedding = pd.DataFrame(vld_save_load, columns = column_names)
        del vld_save_load
        
        vld_emb_integ = pd.concat([valid, vld_embedding], axis = 1)
    
        return trn_emb_integ, vld_emb_integ

def integrating_tsne(data_path, emb_path_trn, emb_path_vld, mode):
    
    dataset = pd.read_csv(data_path, compression='gzip', index_col = 0)
    
    
    if mode == 'eicu':
        dataset = dataset.rename(columns = {'SaO2_fillna':'SpO2_fillna'})
        dataset = dataset[~(dataset['gender']==2)].reset_index(drop=True)
        save_load = np.load(emb_path_trn)
        num_columns = len(save_load[0])
        column_names = ['emb_{}'.format(i+1) for i in range(num_columns)] 
        embedding = pd.DataFrame(save_load, columns = column_names)
        del save_load
        
        model = TSNE(n_components=2)
        X_embedded = model.fit_transform(embedding)
        tsne_emb_feat_df = pd.DataFrame(X_embedded, columns = ['emb 0', 'emb 1'])
        
        emb_integ = pd.concat([dataset, tsne_emb_feat_df], axis = 1)
        
        return emb_integ
        
    else:    

        train, valid = data_split(dataset, 9040, 0.7, Threshold=0.05, n_trial=1, mode = mode)
        
        trn_save_load = np.load(emb_path_trn)
        vld_save_load = np.load(emb_path_vld)
        
        num_columns = len(trn_save_load[0])
        column_names = ['emb_{}'.format(i+1) for i in range(num_columns)]
        trn_embedding = pd.DataFrame(trn_save_load, columns = column_names)
        del trn_save_load
        
        model = TSNE(n_components=2)
        X_embedded = model.fit_transform(trn_embedding)
        tsne_emb_feat_df = pd.DataFrame(X_embedded, columns = ['emb 0', 'emb 1'])
        trn_emb_integ = pd.concat([train, tsne_emb_feat_df], axis = 1)
        
        vld_embedding = pd.DataFrame(vld_save_load, columns = column_names)
        del vld_save_load
        
        X_embedded = model.fit_transform(vld_embedding)
        tsne_emb_feat_df = pd.DataFrame(X_embedded, columns = ['emb 0', 'emb 1'])
        vld_emb_integ = pd.concat([valid, tsne_emb_feat_df], axis = 1)
    
        return trn_emb_integ, vld_emb_integ

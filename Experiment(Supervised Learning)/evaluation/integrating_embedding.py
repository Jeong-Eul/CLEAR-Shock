#!/usr/bin/env python
# coding: utf-8

# In[1]:

from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import sys



module_path='/Users/DAHS/Desktop/ECP_CONT/ECP_SCL/Cohort_selection/'
if module_path not in sys.path:
    sys.path.append(module_path)

from cohort_loader import *

mimic_data_dir = '/Users/DAHS/Desktop/ECP_CONT/ECP_SCL/Case Labeling/mimic_analysis.csv.gz'
eicu_data_dir = '/Users/DAHS/Desktop/ECP_CONT/ECP_SCL/Case Labeling/eicu_analysis.csv.gz'


def integrating(data_path, emb_path_trn, emb_path_vld, emb_path_event, mode):
    
    data = pd.read_csv(data_path, compression='gzip')
    data = data.drop('Shock_next_12h', axis = 1)
    
    mimic_interest_col = ['subject_id', 'stay_id', 'Annotation', 'Case', 'INDEX','hadm_id', 'ethnicity', 'Shock_next_8h',
                          'Fluids(ml)', 'lactate_up', 'hypovolemia_external', 'gender',
       'Tropinin-T_fillna', 'progress', 'Cardiac Output', 'Sofa_Urine',
       'Time_since_ICU_admission', 'vasoactive/inotropic', 'RedBloodCell',
       'RedBloodCell_change_11h', 'weight', 'PaO2/FiO2',
       'O2 Sat (%)_fillna', 'hypovolemia_internal', 'Lactate',
       'Sofa_Coagulation', 'Sofa_Respiration', 'Sodium_fillna', 'Glucose',
       'Adv__Valvular_HD', 'Platelets', 'cum_use_vaso',
       'suspected_infection', 'Age', 'Sofa_Liver',
       'Acute_Myocardial_Infarction', 'Alkaline phosphatase', 'RASS',
       'pH_change_11h', 'AST', 'Sofa_GCS', 'HR_change_11h', 'ALT',
       'HR_change_7h', 'Cardiac Output_fillna', 'FIO2 (%)',
       'Creatinine_change_11h', 'HR_change_9h', 'RedBloodCell_change_9h',
       'Hematocrit', 'Lactate_fillna', 'Mechanical_circ_support', 'PEEP',
       'HR', 'Lactate_clearance_11h', 'HR_change_5h',
       'RedBloodCell_change_5h']
    
    eicu_interest_col = ['uniquepid', 'patientunitstayid', 'Annotation', 'Case', 'INDEX', 'ethnicity', 'Shock_next_8h'
                         ,'Fluids(ml)', 'lactate_up', 'hypovolemia_external', 'gender',
       'Tropinin-T_fillna', 'progress', 'Cardiac Output', 'Sofa_Urine',
       'Time_since_ICU_admission', 'vasoactive/inotropic', 'RedBloodCell',
       'RedBloodCell_change_11h', 'weight', 'PaO2/FiO2',
       'O2 Sat (%)_fillna', 'hypovolemia_internal', 'Lactate',
       'Sofa_Coagulation', 'Sofa_Respiration', 'Sodium_fillna', 'Glucose',
       'Adv__Valvular_HD', 'Platelets', 'cum_use_vaso',
       'suspected_infection', 'Age', 'Sofa_Liver',
       'Acute_Myocardial_Infarction', 'Alkaline phosphatase', 'RASS',
       'pH_change_11h', 'AST', 'Sofa_GCS', 'HR_change_11h', 'ALT',
       'HR_change_7h', 'Cardiac Output_fillna', 'FIO2 (%)',
       'Creatinine_change_11h', 'HR_change_9h', 'RedBloodCell_change_9h',
       'Hematocrit', 'Lactate_fillna', 'Mechanical_circ_support', 'PEEP',
       'HR', 'Lactate_clearance_11h', 'HR_change_5h',
       'RedBloodCell_change_5h']
    
    
    if mode == 'eicu': 
        dataset = data[~(data['gender']==2)].reset_index(drop=True)
        dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
        dataset.fillna(0, inplace=True) 
        
        eventset = data[(data['Case']=='event')].reset_index(drop=True)
        dataset = dataset[~(dataset['Case']=='event')]
        dataset['Case'] = pd.to_numeric(dataset['Case'], errors='coerce')
        
        save_load = np.load(emb_path_trn)
        num_columns = len(save_load[0])
        column_names = ['emb_{}'.format(i+1) for i in range(num_columns)] 
        embedding = pd.DataFrame(save_load, columns = column_names)
        del save_load
        
        emb_integ = pd.concat([dataset.reset_index(drop=True), embedding], axis = 1)
        
        return emb_integ, eventset
        
    else:    
        dataset = data[~(data['gender']==2)].reset_index(drop=True)

        dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
        dataset.fillna(0, inplace=True) 
        
        dataset = dataset[~(dataset['Case']=='event')]
        
        eventset = data[(data['Case']=='event')].reset_index(drop=True)
        
        event_save_load = np.load(emb_path_event)
        
        num_columns = len(event_save_load[0])
        column_names = ['emb_{}'.format(i+1) for i in range(num_columns)]
        event_embedding = pd.DataFrame(event_save_load, columns = column_names)
        del event_save_load
        evt_emb_integ = pd.concat([eventset, event_embedding], axis = 1)
        
        dataset['Case'] = pd.to_numeric(dataset['Case'], errors='coerce')
        
        train, valid = data_split(dataset, 9040, 0.9, Threshold=0.05, n_trial=1, mode = mode)
        
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
    
        return trn_emb_integ, vld_emb_integ, eventset, evt_emb_integ


def integrating_dimensionality_reduction(data_path, emb_path_trn, emb_path_vld, mode):
    
    data = pd.read_csv(data_path, compression='gzip')
    data = data.drop('Shock_next_12h', axis = 1)
    if mode == 'eicu': 
        dataset = data[~(data['gender']==2)].reset_index(drop=True)
        dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
        dataset.fillna(0, inplace=True) 
        
        eventset = data[(data['Annotation']=='event')].reset_index(drop=True)
        
        dataset = dataset[~(dataset['Case']=='event')]
        dataset['Case'] = pd.to_numeric(dataset['Case'], errors='coerce')
        
        save_load = np.load(emb_path_trn)
        num_columns = len(save_load[0])
        column_names = ['emb_{}'.format(i+1) for i in range(num_columns)] 
        embedding = pd.DataFrame(save_load, columns = column_names)
        
        n_components = 3
        model = TSNE(n_components=n_components,perplexity=30.0, random_state=42)
        X_embedded = model.fit_transform(embedding)
        feat_df = pd.DataFrame(X_embedded, columns = ['emb 0', 'emb 1', 'emb 2'])
        del save_load
        
        emb_integ = pd.concat([dataset.reset_index(drop=True), feat_df], axis = 1)
        
        return emb_integ, eventset
        
    else:    
        dataset = data[~(data['gender']==2)].reset_index(drop=True)

        dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
        dataset.fillna(0, inplace=True) 
        
        dataset = dataset[~(dataset['Case']=='event')]
        
        eventset = data[(data['Case']=='event')].reset_index(drop=True)
        dataset['Case'] = pd.to_numeric(dataset['Case'], errors='coerce')
        
        train, valid = data_split(dataset, 9040, 0.9, Threshold=0.05, n_trial=1, mode = mode)
        
        trn_save_load = np.load(emb_path_trn)
        vld_save_load = np.load(emb_path_vld)
        
        num_columns = len(trn_save_load[0])
        column_names = ['emb_{}'.format(i+1) for i in range(num_columns)]
        trn_embedding = pd.DataFrame(trn_save_load, columns = column_names)
        del trn_save_load
        
        n_components = 3
        model = TSNE(n_components=n_components,perplexity=30.0, random_state=42)
        X_embedded = model.fit_transform(trn_embedding)
        feat_df_trn = pd.DataFrame(X_embedded, columns = ['emb 0', 'emb 1', 'emb 2'])
        
        
        trn_emb_integ = pd.concat([train, feat_df_trn], axis = 1)
        del feat_df_trn
        
        vld_embedding = pd.DataFrame(vld_save_load, columns = column_names)
        del vld_save_load
        
        n_components = 3
        model = TSNE(n_components=n_components,perplexity=30.0, random_state=42)
        X_embedded = model.fit_transform(vld_embedding)
        feat_df_vld = pd.DataFrame(X_embedded, columns = ['emb 0', 'emb 1', 'emb 2'])
        
        vld_emb_integ = pd.concat([valid, feat_df_vld], axis = 1)
        del feat_df_vld
        
        return trn_emb_integ, vld_emb_integ, eventset
#!/usr/bin/env python
# coding: utf-8

# In[ ]:
# 변수 통일 및 파생 변수 제작

from preproc import integration_source_target
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gc

# patient = pd.read_csv("/Users/DAHS/Desktop/early_prediction_of_circ_scl/eicu-crd/2.0/patient.csv.gz", compression='gzip')
# patient[patient.uniquepid.isin(eicu.uniquepid.unique())].hospitalid.nunique()

vital_target = ['MAP', 'HR', 'ABPs', 'ABPd', 'Respiratory Rate', 'SVO2', 'O2 Sat (%)']
lab_target = ['pH', 'Creatinine', 'Lactate', 'ALT', 'AST', 'Total Bilirubin', 'RedBloodCell', 'Troponin-T']


def return_result_df():
    mimic_path = "/Users/wjddm/OneDrive/바탕 화면/local_dahs/Data/MIMICIV-CIRC(12h).csv.gz"
    eicu_path = "/Users/wjddm/OneDrive/바탕 화면/local_dahs/Data/eICU-CIRC(12h).csv.gz"

    mimic, eicu = integration_source_target(mimic_path, eicu_path)

    mimic_v1 = make_derived_variable(mimic, 'mimic')
    eicu_v1 = make_derived_variable(eicu, 'eicu')
    return mimic_v1, eicu_v1


def make_derived_variable(df, mode):
    dataset = df.copy()
    result_1 = Lactate_clearance(dataset, mode)
    result_2 = cumsum_vasoactive(result_1, mode)
    result_3 = Rate_of_change(result_2, mode)

    return result_3


#Lactate clearance

def Lactate_clearance(data, mode):
    df = data.copy()
    gc.collect()
    
    if mode == 'mimic':
        stayid = 'stay_id'
    elif mode == 'eicu':
        stayid == 'patientunitstayid'

    # Calculating Lactate clearance for different time intervals (1, 3, 5, 7, 9 ,11 hours) with corrected logic
    for hours in [1, 3, 5, 7, 9, 11]:
        # Shifting the Lactate values to get the initial Lactate (1, 3, 5, 7, 9 ,11 hours ago)
        df[f'Lactate_initial_{hours}h'] = df.groupby(stayid)['Lactate'].shift(hours)
        
        # Calculating Lactate clearance
        df[f'Lactate_clearance_{hours}h'] = (df[f'Lactate_initial_{hours}h'] - df['Lactate']) / df[f'Lactate_initial_{hours}h'] * 100
        df = df.drop([f'Lactate_initial_{hours}h'], axis =1)
    return df.fillna(0)



#Cumulative number of uses of vasoactive/inotropic drugs

def cumsum_vasoactive(data, mode):
    df = data.copy()
    gc.collect()

    if mode == 'mimic':
        stayid = 'stay_id'
    elif mode == 'eicu':
        stayid == 'patientunitstayid'

    df['cum_use_vaso'] = df.groupby(stayid)['vasoactive/inotropic'].cumsum()

    return df



#rate of change in measurement over time

def Rate_of_change(data, mode):
    use_cols = vital_target + lab_target

    df = data.copy()
    gc.collect()
    
    if mode == 'mimic':
        stayid = 'stay_id'
    elif mode == 'eicu':
        stayid == 'patientunitstayid'

    # Calculating rate of change in measurment for different time intervals (1, 3, 5, 7, 9 ,11 hours) with corrected logic
    for hours in [1, 3, 5, 7, 9, 11]:
        for target in use_cols:

            # Shifting the target values to get the initial target (1, 3, 5, 7, 9 ,11 hours ago)
            df[f'measure_initial_{hours}h'] = df.groupby(stayid)[f'{target}'].shift(hours)
            
            # Calculating target rate of change
            df[f'{target}_change_{hours}h'] = (df[f'measure_initial_{hours}h'] - df[f'{target}']) / df[f'measure_initial_{hours}h'] * 100
            df = df.drop([f'measure_initial_{hours}h'], axis =1)
    return df.fillna(0)


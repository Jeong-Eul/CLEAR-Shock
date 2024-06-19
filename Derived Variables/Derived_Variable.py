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
lab_target = ['pH', 'Creatinine', 'ALT', 'AST', 'Total Bilirubin', 'RedBloodCell', 'Troponin-T']


def return_result_df():
    mimic_path = "/Users/DAHS/Desktop/MIMICIV2.2_PREPROC/preprocessing/MIMICIV-CIRC(12h).csv.gz"
    eicu_path = "/Users/DAHS/Desktop/eICU2.0_PREPROC/eICU-CIRC(12h).csv.gz"

    mimic, eicu = integration_source_target(mimic_path, eicu_path)
    
    
    null_dict = {'NIBPs' : 125, 'HR' : 70, 'MAP' : 91.6,'O2 Sat (%)':95, 'PaO2/FiO2':400,
            'NIBPd' : 75, 'Temperature': 36.5, 'RASS' : 0, 'ABPs': 125, 'PEEP' : 0, 'Peak Insp. Pressure':0,
            'ABPd':75, 'CVP':4, 'EtCO2': 40, 'SVO2':70, 'SpO2': 95, 'Lactate':1.1,
            'Creatinine':0.7, 'Glucose':5 , 'Potassium':4, 'PaCO2':40, 'Ca+':8.5, 'Hemoglobin':12,
            'PaO2':75, 'Alkaline phosphatase':85, 'BUN':23, 'ALT':25, 'AST':25, 'FIO2 (%)':21, 'CO2':25, 'INR':1,
            'Hematocrit':42, 'Platelets':300, 'WBC':10}
    
    columns_to_check = ['HR', 'ABPd', 'ABPs', 'Temperature', 'RASS',
                        'PEEP', 'Peak Insp. Pressure', 'CVP', 'EtCO2','SpO2',
                        'ABPd', 'MAP', 'FIO2 (%)', 'SVO2', 'Lactate', 'Creatinine',
                        'Potassium', 'PaCO2', 'Ca+', 'PaO2', 'BUN', 'INR']
    
    
    thresholds = {
    'HR': (0, 300),
    'ABPd': (10, 175),
    'Temperature': (0, 70),
    'RASS' : (-5, 4),
    'ABPs': (10, 300),
    'PEEP': (0, 9999),
    'Peak Insp. Pressure': (5, 100),
    'CVP': (0, 50),
    'EtCO2': (0, 45),
    'SpO2': (0, 100), 
    'ABPd': (0, 120), 
    'MAP': (0, 150), 
    'FIO2 (%)': (0, 100),
    'SVO2': (0, 85),
    'Lactate': (0, 15),
    'Creatinine':(0, 1500),
    'Potassium' : (2, 12),
    'PaCO2': (0, 1000),
    'Ca+': (0, 3),
    'PaO2': (0, 150),
    'BUN': (0, 1000),
    'INR': (0, 100),
    'O2 Sat (%)' : (0, 300),
    'PaO2/FiO2': (0, 1000)
    }

    for column in columns_to_check:
        lower, upper = thresholds[column]
        mask = (mimic[column] < lower) | (mimic[column] > upper)
        mimic.loc[mask, column] = np.nan
        mimic[column + '_fillna'] = mimic[column].isnull().astype(int)
        mimic[column] = mimic[column].fillna(null_dict[column])
        
        mask = (eicu[column] < lower) | (eicu[column] > upper)
        eicu.loc[mask, column] = np.nan
        eicu[column + '_fillna'] = eicu[column].isnull().astype(int)
        eicu[column] = eicu[column].fillna(null_dict[column])
    
    
    print('---Complete Integration---')

    mimic_v1 = make_derived_variable(mimic, 'mimic')

    print('---Complete make derived variable for MIMIC-IV---')

    eicu_v1 = make_derived_variable(eicu, 'eicu')

    print('---Complete make derived variable for eICU---')

    return mimic_v1, eicu_v1


def make_derived_variable(df, mode):
    dataset = df.copy()
    result_1 = Lactate_clearance(dataset, mode)
    print('Complete Lactate Clearance')
    result_2 = cumsum_vasoactive(result_1, mode)
    print('Complete cumulative use of vasoactive')
    result_3 = Rate_of_change(result_2, mode)
    print('Complete Rate of change in measurement')

    return result_3


#Lactate clearance

def Lactate_clearance(data, mode):
    df = data.copy()
    gc.collect()
    
    if mode == 'mimic':
        stayid = 'stay_id'
    elif mode == 'eicu':
        stayid = 'patientunitstayid'

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
        stayid = 'patientunitstayid'

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
        stayid = 'patientunitstayid'

    # Calculating rate of change in measurment for different time intervals (1, 3, 5, 7, 9 ,11 hours) with corrected logic
    for hours in [1, 3, 5, 7, 9, 11]:
        for target in use_cols:

            # Shifting the target values to get the initial target (1, 3, 5, 7, 9 ,11 hours ago)
            df[f'measure_initial_{hours}h'] = df.groupby(stayid)[f'{target}'].shift(hours)
            
            # Calculating target rate of change
            df[f'{target}_change_{hours}h'] = (df[f'measure_initial_{hours}h'] - df[f'{target}']) / df[f'measure_initial_{hours}h'] * 100
            df = df.drop([f'measure_initial_{hours}h'], axis =1)
    return df.fillna(0)

mimic, eicu = return_result_df()
mimic.to_csv('mimic_df_cp.csv.gz', compression='gzip')
eicu.to_csv('eicu_df_cp.csv.gz', compression='gzip')
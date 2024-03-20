import pandas as pd
import numpy as np

def hospital(eicu):
    
    sample = eicu[['patientunitstayid', 'hospitalid', 'hospitaldischargeyear']].groupby('hospitalid').agg('nunique').reset_index()
    hospital = sample[sample['patientunitstayid']>=10]
    df_target = eicu[eicu['hospitalid'].isin(hospital['hospitalid'])]
    
    return hospital['hospitalid'].values, df_target

def eicu_subgroup(eicu):
    eicu_patient_path = '/Users/gwonjeong-eul/Desktop/ecp-scl-macbook/eicu_patients/patient.csv.gz'
    patient = pd.read_csv(eicu_patient_path, compression = 'gzip')
    # interest = patient[['patientunitstayid', 'hospitaladmitsource', 'unittype', 'unitstaytype', 'unitadmitsource', 'hospitaldischargeoffset', 'hospitaldischargestatus']]
    interest = patient[['patientunitstayid', 'hospitaladmitsource', 'unittype', 'unitstaytype', 'unitadmitsource']]

    
    #hospitaladmitsource---
    #'Direct Admit', 'Emergency Department', 'Floor', 'Operating Room', nan, 'Other Hospital', 'Other ICU', 'ICU to SDU',
    #'Step-Down Unit (SDU)', 'Recovery Room', 'Chest Pain Center','Acute Care/Floor', 'PACU', 'Observation', 'ICU', 'Other'
    
    #unittype
    #'Med-Surg ICU', 'CTICU', 'SICU', 'CCU-CTICU', 'MICU', 'Neuro ICU',
    #'Cardiac ICU', 'CSICU'
    
    #unitstaytype
    #'admit', 'stepdown/other', 'readmit', 'transfer'
    
    #unitadmitsource
    #'Direct Admit', 'Emergency Department', 'ICU to SDU', 'Floor',
    #'Operating Room', 'Other Hospital', nan, 'Step-Down Unit (SDU)',
    #'Other ICU', 'Recovery Room', 'Chest Pain Center', 'ICU',
    #'Acute Care/Floor', 'PACU', 'Observation', 'Other'
    
    interest = pd.merge(eicu, interest, how = 'left', on = ['patientunitstayid']).reset_index(drop=True)
    
    
    # admittsource = interest.groupby('hospitaladmitsource').agg(
    # ratio_obs=pd.NamedAgg(column='hospitaladmitsource', aggfunc=lambda x: len(x) / len(interest)),
    # n_stay=pd.NamedAgg(column='patientunitstayid', aggfunc='nunique')).reset_index()
    
    # admittsource.to_csv('eicu_admission_source_summary.csv')
    
    # unittype = interest.groupby('unittype').agg(
    # ratio_obs=pd.NamedAgg(column='unittype', aggfunc=lambda x: len(x) / len(interest)),
    # n_stay=pd.NamedAgg(column='patientunitstayid', aggfunc='nunique')).reset_index()
    
    # unittype.to_csv('eicu_unittype_summary.csv')
    
    # unitadmittsource = interest.groupby('unitadmitsource').agg(
    # ratio_obs=pd.NamedAgg(column='unitadmitsource', aggfunc=lambda x: len(x) / len(interest)),
    # n_stay=pd.NamedAgg(column='patientunitstayid', aggfunc='nunique')).reset_index()
    
    # unitadmittsource.to_csv('eicu_unitadmission_source_summary.csv')
    
    # unitstaytype = interest.groupby('unitstaytype').agg(
    # ratio_obs=pd.NamedAgg(column='unitstaytype', aggfunc=lambda x: len(x) / len(interest)),
    # n_stay=pd.NamedAgg(column='patientunitstayid', aggfunc='nunique')).reset_index()
    
    # unitstaytype.to_csv('eicu_unitstaytype_summary.csv')
    
    return interest

def make_eicu_dataset(eicu):
    
    unitadmitsource = {}
    for type in eicu.unitadmitsource.unique():
        unitadmitsource[type] = eicu[eicu['unitadmitsource']==type].sort_values(by=['patientunitstayid', 'Time_since_ICU_admission']).drop(['hospitaladmitsource','unittype','unitstaytype','unitadmitsource'], axis = 1).reset_index(drop=True)
        
    unittype = {}
    for type in eicu.unittype.unique():
        unittype[type] = eicu[eicu['unittype']==type].sort_values(by=['patientunitstayid', 'Time_since_ICU_admission']).drop(['hospitaladmitsource','unittype','unitstaytype','unitadmitsource'], axis = 1).reset_index(drop=True)
    
    unitstaytype = {}
    for type in eicu.unitstaytype.unique():
        unitstaytype[type] = eicu[eicu['unitstaytype']==type].sort_values(by=['patientunitstayid', 'Time_since_ICU_admission']).drop(['hospitaladmitsource','unittype','unitstaytype','unitadmitsource'], axis = 1).reset_index(drop=True)
    
    return unitadmitsource, unittype, unitstaytype


def count_event_by_subpop(unitadmitsource, unittype, unitstaytype, event_eicu):
    result = pd.DataFrame()

    for type in unitadmitsource.keys():
        result_sub = pd.DataFrame()
        eval_set = unitadmitsource[type]
        num_events = eval_set[eval_set['patientunitstayid'].isin(event_eicu.patientunitstayid.unique())].patientunitstayid.nunique()
        
        result_sub = pd.DataFrame({'main population': ['UnitAdmitSource'], 'sub population': [type], 'num of event': [num_events]})
        result = result.append(result_sub, ignore_index=True)
        
    for type in unittype.keys():
        result_sub = pd.DataFrame()
        eval_set = unittype[type]
        num_events = eval_set[eval_set['patientunitstayid'].isin(event_eicu.patientunitstayid.unique())].patientunitstayid.nunique()
        
        result_sub = pd.DataFrame({'main population': ['UnitType'], 'sub population': [type], 'num of event': [num_events]})
        result = result.append(result_sub, ignore_index=True)
        
    for type in unitstaytype.keys():
        result_sub = pd.DataFrame()
        eval_set = unitstaytype[type]
        num_events = eval_set[eval_set['patientunitstayid'].isin(event_eicu.patientunitstayid.unique())].patientunitstayid.nunique()
        
        result_sub = pd.DataFrame({'main population': ['UnitStayType'], 'sub population': [type], 'num of event': [num_events]})
        result = result.append(result_sub, ignore_index=True)
        
    result.set_index(['main population', 'sub population']).to_excel(excel_writer='eicu_subpop_events.xlsx')
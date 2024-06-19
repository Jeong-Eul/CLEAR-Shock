import pandas as pd
import numpy as np
from sklearn.metrics import precision_score
from tqdm import tqdm



def split_X_Y(df, mode):
    
    if mode == 'mimic':
        X = df.drop(['subject_id', 'Unnamed: 0', 'stay_id', 'hadm_id','Annotation', 'Case', 'after_shock_annotation', 'Shock_next_8h', 'INDEX'], axis = 1)
        y = df['Case'].values
        output = df[['stay_id', 'Time_since_ICU_admission', 'Annotation', 'after_shock_annotation', 'progress', 'Case', 'INDEX']].copy().reset_index(drop=True)
        output['Case'] = output['Case'].astype(int)
        
    else:
        X = df.drop(['uniquepid', 'Unnamed: 0', 'patientunitstayid','Annotation', 'Case','after_shock_annotation', 'Shock_next_8h', 'INDEX', 'hospitalid', 'hospitaldischargeyear'], axis = 1)
        y = df['Case'].values
        output = df[['patientunitstayid', 'Time_since_ICU_admission', 'Annotation','after_shock_annotation', 'progress','Case', 'INDEX']].copy().reset_index(drop=True)
        output['Case'] = output['Case'].astype(int)

    return X, y, output

def split_X_Y_for_VIZ(df, mode):
    
    if mode == 'mimic':
        X = df.drop(['subject_id', 'Unnamed: 0', 'stay_id', 'hadm_id','Annotation', 'Case', 'after_shock_annotation', 'Shock_next_8h', 'INDEX'], axis = 1)
        y = df['Case'].values
        output = df[['stay_id', 'Time_since_ICU_admission', 'vasoactive/inotropic', 'Fluids(ml)', 'Annotation', 'after_shock_annotation', 'Case', 'INDEX', 'Lactate', 'MAP']].copy().reset_index(drop=True)
        output['Case'] = output['Case'].astype(int)
        
    else:
        X = df.drop(['uniquepid', 'Unnamed: 0', 'patientunitstayid','Annotation', 'Case', 'after_shock_annotation', 'Shock_next_8h', 'INDEX', 'hospitalid', 'hospitaldischargeyear'], axis = 1)
        y = df['Case'].values
        output = df[['patientunitstayid', 'Time_since_ICU_admission', 'vasoactive/inotropic', 'Fluids(ml)', 'Annotation','after_shock_annotation','Case', 'INDEX', 'Lactate', 'MAP']].copy().reset_index(drop=True)
        output['Case'] = output['Case'].astype(int)

    return X, y, output


def split_X_Y_MORT(df, mode):
    
    if mode == 'mimic':
        X = df.drop(['Unnamed: 0', 'stay_id', 'hadm_id', 'death','progress'], axis = 1)
        y = df['death'].astype(int).values
        output = df[['stay_id', 'Time_since_ICU_admission', 'death']].copy().reset_index(drop=True)
        output['death'] = output['death'].astype(int)
        
    else:
        X = df.drop(['Unnamed: 0', 'patientunitstayid', 'death','hospitalid', 'hospitaldischargeyear','progress'], axis = 1)
        y = df['death'].astype(int).values
        output = df[['patientunitstayid', 'Time_since_ICU_admission', 'death']].copy().reset_index(drop=True)
        output['death'] = output['death'].astype(int)

    return X, y, output


def split_X_Y_LOS(df, mode):
    
    if mode == 'mimic':
        X = df.drop(['Unnamed: 0', 'stay_id', 'hadm_id', 'remain_los', 'progress'], axis = 1)
        y = df['remain_los'].astype(int).values
        output = df[['stay_id', 'Time_since_ICU_admission', 'remain_los']].copy().reset_index(drop=True)
        output['remain_los'] = output['remain_los'].astype(int)
        
    else:
        X = df.drop(['Unnamed: 0', 'patientunitstayid', 'remain_los','hospitalid', 'hospitaldischargeyear','progress'], axis = 1)
        y = df['remain_los'].astype(int).values
        output = df[['patientunitstayid', 'Time_since_ICU_admission', 'remain_los']].copy().reset_index(drop=True)
        output['remain_los'] = output['remain_los'].astype(int)

    return X, y, output


def split_X_Y_ARDS4h(df, mode):
    
    if mode == 'mimic':
        X = df.drop(['Unnamed: 0', 'subject_id', 'stay_id', 'hadm_id', 'Annotation_ARDS','ARDS_next_4h'], axis = 1)
        y = df['ARDS_next_4h'].astype(int).values
        output = df[['stay_id', 'Time_since_ICU_admission', 'Annotation_ARDS','ARDS_next_4h']].copy().reset_index(drop=True)
        output['ARDS_next_4h'] = output['ARDS_next_4h'].astype(int)
        
    else:
        X = df.drop(['Unnamed: 0', 'uniquepid', 'patientunitstayid', 'Annotation_ARDS','ARDS_next_4h'], axis = 1)
        y = df['ARDS_next_4h'].astype(int).values
        output = df[['patientunitstayid', 'Time_since_ICU_admission', 'Annotation_ARDS','ARDS_next_4h']].copy().reset_index(drop=True)
        output['ARDS_next_4h'] = output['ARDS_next_4h'].astype(int)

    return X, y, output

def split_X_Y_ARDS8h(df, mode):
    
    if mode == 'mimic':
        X = df.drop(['Unnamed: 0', 'subject_id', 'stay_id', 'hadm_id', 'Annotation_ARDS','ARDS_next_8h'], axis = 1)
        y = df['ARDS_next_8h'].astype(int).values
        output = df[['stay_id', 'Time_since_ICU_admission', 'Annotation_ARDS','ARDS_next_8h']].copy().reset_index(drop=True)
        output['ARDS_next_8h'] = output['ARDS_next_8h'].astype(int)
        
    else:
        X = df.drop(['Unnamed: 0', 'uniquepid', 'patientunitstayid', 'Annotation_ARDS','ARDS_next_8h'], axis = 1)
        y = df['ARDS_next_8h'].astype(int).values
        output = df[['patientunitstayid', 'Time_since_ICU_admission', 'Annotation_ARDS','ARDS_next_8h']].copy().reset_index(drop=True)
        output['ARDS_next_8h'] = output['ARDS_next_8h'].astype(int)

    return X, y, output


def split_X_Y_SIC4h(df, mode):
    
    if mode == 'mimic':
        X = df.drop(['Unnamed: 0', 'subject_id', 'stay_id', 'hadm_id', 'Annotation_SIC','SIC_next_4h'], axis = 1)
        y = df['SIC_next_4h'].astype(int).values
        output = df[['stay_id', 'Time_since_ICU_admission', 'Annotation_SIC','SIC_next_4h']].copy().reset_index(drop=True)
        output['SIC_next_4h'] = output['SIC_next_4h'].astype(int)
        
    else:
        X = df.drop(['Unnamed: 0', 'uniquepid', 'patientunitstayid', 'Annotation_SIC','SIC_next_4h'], axis = 1)
        y = df['SIC_next_4h'].astype(int).values
        output = df[['patientunitstayid', 'Time_since_ICU_admission', 'Annotation_SIC','SIC_next_4h']].copy().reset_index(drop=True)
        output['SIC_next_4h'] = output['SIC_next_4h'].astype(int)

    return X, y, output

def split_X_Y_SIC8h(df, mode):
    
    if mode == 'mimic':
        X = df.drop(['Unnamed: 0', 'subject_id', 'stay_id', 'hadm_id', 'Annotation_SIC','SIC_next_8h'], axis = 1)
        y = df['SIC_next_8h'].astype(int).values
        output = df[['stay_id', 'Time_since_ICU_admission', 'Annotation_SIC','SIC_next_8h']].copy().reset_index(drop=True)
        output['SIC_next_8h'] = output['SIC_next_8h'].astype(int)
        
    else:
        X = df.drop(['Unnamed: 0', 'uniquepid', 'patientunitstayid', 'Annotation_SIC','SIC_next_8h'], axis = 1)
        y = df['SIC_next_8h'].astype(int).values
        output = df[['patientunitstayid', 'Time_since_ICU_admission', 'Annotation_SIC','SIC_next_8h']].copy().reset_index(drop=True)
        output['SIC_next_8h'] = output['SIC_next_8h'].astype(int)

    return X, y, output
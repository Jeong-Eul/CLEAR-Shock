import pandas as pd
import numpy as np
from sklearn.metrics import precision_score
from tqdm import tqdm



def split_X_Y(df, mode):
    
    if mode == 'mimic':
        X = df.drop(['subject_id', 'stay_id', 'hadm_id','Annotation', 'Case', 'ethnicity', 'Shock_next_8h', 'INDEX'], axis = 1)
        y = df['Case'].values
        output = df[['stay_id', 'Time_since_ICU_admission', 'Annotation', 'progress', 'Case', 'INDEX']].copy().reset_index(drop=True)
        output['Case'] = output['Case'].astype(int)
        
    else:
        X = df.drop(['uniquepid', 'patientunitstayid','Annotation', 'Case', 'ethnicity', 'Shock_next_8h', 'INDEX'], axis = 1)
        y = df['Case'].values
        output = df[['patientunitstayid', 'Time_since_ICU_admission', 'Annotation','progress','Case', 'INDEX']].copy().reset_index(drop=True)
        output['Case'] = output['Case'].astype(int)

    return X, y, output
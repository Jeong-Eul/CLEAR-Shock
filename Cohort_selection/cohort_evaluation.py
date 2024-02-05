import pandas as pd
import numpy as np
from sklearn.metrics import precision_score
from tqdm import tqdm



def split_X_Y(df, mode):
    
    if mode == 'mimic':
        X = df.drop(['subject_id', 'stay_id', 'hadm_id','Annotation', 'classes', 'ethnicity', 'Shock_next_12h'], axis = 1)
        y = df['Shock_next_12h'].values
        output = df[['stay_id', 'Time_since_ICU_admission', 'Annotation', 'Shock_next_12h']].copy().reset_index(drop=True)
        output['Shock_next_12h'] = output['Shock_next_12h'].astype(int)
        
    else:
        X = df.drop(['uniquepid', 'patientunitstayid','Annotation', 'classes', 'ethnicity', 'Shock_next_12h'], axis = 1)
        y = df['Shock_next_12h'].values
        output = df[['patientunitstayid', 'Time_since_ICU_admission', 'Annotation', 'Shock_next_12h']].copy().reset_index(drop=True)
        output['Shock_next_12h'] = output['Shock_next_12h'].astype(int)

    return X, y, output


def get_evaluation(sample, mode):
    
# sample - > stay_id | Time_since_ICU_admission | Annotation | Shock_next_12h | prediction_label

    # sample = sample[~(sample['Annotation']=='amb_circ')].reset_index(drop=True)
    sample_precision = sample[(sample['Annotation']=='no_circ')|(sample['Annotation']=='ambiguous')]
    
    precision = precision_score(sample_precision.Shock_next_12h, sample_precision.prediction_label)
    
    if mode == 'mimic':
        stay_id = 'stay_id'
        
    else:
        stay_id = 'patientunitstayid'

    recall = []
    
    total_event = len(sample[sample['Annotation']=='circ'])
    captured_event = 0

    for stay in tqdm(sample[stay_id].unique()):
        
        pred = []
        true = []
        
        target = sample[sample[stay_id]==stay]
        target = target.sort_values(by='Time_since_ICU_admission')
        
        
        if (target['Annotation']=='circ').any():
            
            circ_idx = target[target['Annotation']=='circ'].index
            
            if len(circ_idx) >= 2:
                
                for idx in circ_idx:
                    last_obs_time = target['Time_since_ICU_admission'].loc[idx].values[0]
                    time_threshold = last_obs_time - 12
                    
                    capture_before_12 = target[(target['Time_since_ICU_admission'] >= time_threshold) & (target['Time_since_ICU_admission'] < last_obs_time)]

                    if capture_before_12['prediction_label'].sum() >= 1:
                        captured_event += 1
                        
            else:
                last_obs_time = target['Time_since_ICU_admission'].loc[circ_idx].values[0]
                time_threshold = last_obs_time - 12
                
                capture_before_12 = target[(target['Time_since_ICU_admission'] >= time_threshold) & (target['Time_since_ICU_admission'] < last_obs_time)]

                if capture_before_12['prediction_label'].sum() >= 1:
                    captured_event += 1

            
    stay_recall = captured_event / total_event
    recall.append(stay_recall)
    
    return np.round(np.mean(recall),4), precision
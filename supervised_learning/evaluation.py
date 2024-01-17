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


def split_data_by_event(data, event='circ'):
    split_data = []
    current_part = []
    event_occurred = False
    
    for index, row in data.iterrows():
        
        if event_occurred:
            if row['Annotation']==event:
                pass
            
            else:
                current_part.append(row)
                event_occurred = False
                    
        else:
            current_part.append(row)
            if row['Annotation']==event:
                split_data.append(pd.DataFrame(current_part))
                event_occurred = True
                current_part = []
              

    if current_part:
        split_data.append(pd.DataFrame(current_part))

    return split_data


def evaluation(sample, mode):

# sample - > stay_id | Time_since_ICU_admission | Annotation | Shock_next_12h | prediction_label

    if mode == 'mimic':
        stay_id = 'stay_id'
        
    else:
        stay_id = 'patientunitstayid'

    recall = []
    precision = []

    for stay in tqdm(sample[stay_id].unique()):
        
        pred = []
        true = []

        total_event = 0
        captured_event = 0
        
        target = sample[sample[stay_id]==stay]
        target = target.sort_values(by='Time_since_ICU_admission')
        
        if (target['Annotation'] == 'circ').all():
            continue 
        
        else:
            parts = split_data_by_event(target)
        
        for part in parts:
            
            if len(part) == 1:
                continue
            
            elif (part['Annotation']=='circ').any():
                
                total_event += 1
                # caputure_before_12 = part.iloc[-12:-1, -1]
                
                last_obs_time = part['Time_since_ICU_admission'].iloc[-1]
                time_threshold = last_obs_time - 12  # 12시간 이전

                capture_before_12 = part[(part['Time_since_ICU_admission'] >= time_threshold) & (part['Time_since_ICU_admission'] < last_obs_time)]

                if capture_before_12['prediction_label'].sum() >= 1:
                    captured_event += 1
                
                    stay_recall = captured_event / total_event
                    recall.append(stay_recall)
                
                pred_alarm = part.iloc[:-1, -1].values
                true_alarm = part.iloc[:-1, -2].values
                
                pred = np.concatenate([pred, pred_alarm], axis = 0)
                true = np.concatenate([true, true_alarm], axis = 0)
                
            else:
                
                pred_alarm = part['prediction_label'].values
                true_alarm = part['Shock_next_12h'].values
                
                pred = np.concatenate([pred, pred_alarm], axis = 0)
                true = np.concatenate([true, true_alarm], axis = 0)
                
        
        stay_precision = precision_score(true, pred)
        precision.append(stay_precision)
        
    return np.round(np.mean(recall),4), np.round(np.mean(precision), 4)
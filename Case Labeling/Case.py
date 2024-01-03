import pandas as pd
import numpy as np
from tqdm import tqdm

def filter_cohort(mimic, eicu):

    mimic_circ_ids = mimic[(mimic['Annotation'] == 'circ') | (mimic['Annotation'] == 'ambiguous')]['stay_id'].unique()
    eicu_circ_ids = eicu[(eicu['Annotation'] == 'circ') | (eicu['Annotation'] == 'ambiguous')]['patientunitstayid'].unique()
 
    return mimic[mimic['stay_id'].isin(mimic_circ_ids)], eicu[eicu['patientunitstayid'].isin(eicu_circ_ids)]

def early_event_prediction_label(df):
    
    data = df.copy()
    data['classes'] = 'undefined'

    class1 = data[(data['Shock_next_12h']==0) & ((data['Annotation']=='no_circ') | (data['Annotation']=='ambiguous'))].index
    data.loc[class1,'classes'] = 0
    
    class2 = data[(data['Shock_next_12h']==1) & ((data['Annotation']=='no_circ') | (data['Annotation']=='ambiguous'))].index
    data.loc[class2,'classes'] = 1
    
    return data

def optimized_recovered_labeler(df, mode):
    targ = df.copy()
    
    if mode == 'mimic':
        stay_id_id = 'stay_id'
    elif mode == 'eicu':
        stay_id_id = 'patientunitstayid'
    
    for stay_id in tqdm(targ[stay_id_id].unique()):
        stay_df = targ[targ[stay_id_id] == stay_id].sort_values(by='Time_since_ICU_admission')
        for idx, row in stay_df.iterrows():
            if row['Annotation'] == 'circ':
                current_time = row['Time_since_ICU_admission']
                endpoint_window = current_time + 20
    
                window = stay_df[(stay_df['Time_since_ICU_admission'] > current_time) & (stay_df['Time_since_ICU_admission'] <= endpoint_window)]
                if len(window) > 0:

                    counts = window['Annotation'].value_counts()
                    count_amb_no_circ = counts.get('ambiguous', 0) + counts.get('no_circ', 0)
                    count_amb_circ = counts.get('ambiguous', 0) + counts.get('circ', 0)
                    total_state = len(window)

                    recovery_ratio = count_amb_no_circ / total_state
                    no_recovery_ratio = count_amb_circ / total_state

                    if recovery_ratio >= 0.7 and counts.get('no_circ', 0) > 0:
                        targ.loc[idx, 'classes'] = 2
                    elif no_recovery_ratio >= 0.7 and counts.get('circ', 0) > 0:
                        targ.loc[idx, 'classes'] = 3

    return targ

# def count_ambiguous_in_flow(group):
#     annotations = group['Annotation'].tolist()
#     count = 0
#     found_no_circ = False
#     found_circ = False

#     for state in annotations:
#         if state == 'no_circ':
#             found_no_circ = True
#         elif found_no_circ and state == 'ambiguous':
#             count += 1
#         elif found_no_circ and state == 'circ':
#             found_circ = True
#             break

#     return count if found_no_circ and found_circ else 0

# # stay_id 별로 그룹화하고 각 그룹에 대해 패턴에서 'ambiguous' 상태 발생 횟수 계산
# ambiguous_counts = mimic.groupby('stay_id').apply(count_ambiguous_in_flow)

# # 전체 'ambiguous' 상태의 평균 발생 횟수 계산
# average_ambiguous_count = ambiguous_counts.max()

# print("평균 'ambiguous' 상태 변화 횟수:", average_ambiguous_count)

# veiw = pd.DataFrame(ambiguous_counts, columns=['count']).reset_index()
# veiw[veiw['count']>1]['count'].mean()
# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.histplot(veiw[veiw['count']>1]['count'], kde=True, color='blue', label='MIMIC IV', alpha=0.5)
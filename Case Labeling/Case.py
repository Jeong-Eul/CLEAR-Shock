import pandas as pd
import numpy as np
from tqdm import tqdm

pd.set_option('mode.chained_assignment',  None)

def annotation(df, mode):    
    if mode == 'mimic':
        stay_id_id = 'stay_id'
    elif mode == 'eicu':
        stay_id_id = 'patientunitstayid'
    
    targ = df.copy().reset_index(drop=True)

    targ['Annotation'] = np.nan
    for stay_id in tqdm(targ[stay_id_id].unique()):
        stay_df = targ[targ[stay_id_id] == stay_id].sort_values(by='Time_since_ICU_admission')

        for idx, row in stay_df.iterrows():
            current_time = row['Time_since_ICU_admission']
            pastpoint_window_1h = current_time - 1
        
            relevant_rows = stay_df[(stay_df['Time_since_ICU_admission'] <= current_time) & (stay_df['Time_since_ICU_admission'] >= pastpoint_window_1h)]
            no_circ_cond = (row['MAP'] > 65.0) & (row['vasoactive/inotropic'] == 0.0) & (row['Lactate'] < 2)
            circ_cond = ((relevant_rows['MAP'] <= 65.0).all() | (relevant_rows['vasoactive/inotropic'] == 1.0).all()) & (row['Lactate'] >= 2)
            
            if no_circ_cond:
                targ.at[idx, 'Annotation'] = 'no_circ'
            elif circ_cond:
                targ.at[idx, 'Annotation'] = 'circ'
            else:
                targ.at[idx, 'Annotation'] = 'ambiguous'

    return targ.reset_index(drop=True)


def optimized_shock_labeler(df, mode):
    
    if mode == 'mimic':
        stay_id_id = 'stay_id'
    elif mode == 'eicu':
        stay_id_id = 'patientunitstayid'
    
    targ = df.copy()
    targ['Shock_next_8h'] = np.nan
  
    for stay_id in tqdm(targ[stay_id_id].unique()):
        stay_df = targ[targ[stay_id_id] == stay_id].sort_values(by='Time_since_ICU_admission')
        stay_df['endpoint_window'] = stay_df['Time_since_ICU_admission'] + 8

        for idx, row in stay_df.iterrows():
            current_time = row['Time_since_ICU_admission']
            endpoint_window = row['endpoint_window']

       
            future_rows = stay_df[(stay_df['Time_since_ICU_admission'] > current_time) & (stay_df['Time_since_ICU_admission'] <= endpoint_window)]

            if any(future_rows['Annotation'] == 'circ'):
                targ.loc[idx, 'Shock_next_8h'] = 1
            else:
                targ.loc[idx, 'Shock_next_8h'] = 0

    return targ.reset_index(drop=True)


def find_invalid_columns(df):
    
    invalid_columns = []
    str_columns = []
    for column in df.columns:
        # 컬럼이 숫자형 데이터인지 확인
        if pd.api.types.is_numeric_dtype(df[column]):
            # 'inf' 또는 '-inf'를 포함하는지 검사
            if df[column].isin([np.inf, -np.inf]).any():
                invalid_columns.append(column)
            # 너무 큰 값이 포함되어 있는지 검사
            elif df[column].max() > np.finfo(np.float64).max or df[column].min() < np.finfo(np.float64).min:
                invalid_columns.append(column)
        else:
            str_columns.append(column)
    return invalid_columns, str_columns


def replace_inf_with_previous(df, column_list):

    for column in column_list:
        is_inf_or_neg_inf = df[column].isin([np.inf, -np.inf])
        df.loc[is_inf_or_neg_inf, column] = df.loc[is_inf_or_neg_inf, column].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return df


def get_case2(target, event='circ', mode = 'mimic'):
    
    data = target.copy()
    
    split_data = []
    current_part = []
    event_occurred = False
    
    if mode == 'mimic':
        stay_id = 'stay_id'
    else:
        stay_id = 'patientunitstayid'
        
    search_stay_id = set(data[stay_id].unique())
    
    for stayid in search_stay_id:
        dataset = data[data[stay_id]==stayid]
        
        for index, row in dataset.iterrows():
            
            if event_occurred:
                event_occurred = False
                break
                        
            else:
                current_part.append(row)
                if row['Annotation']==event:
                    split_data.append(pd.DataFrame(current_part))
                    event_occurred = True
                    current_part = []
            
        if current_part:
            split_data.append(pd.DataFrame(current_part))
            current_part = []

    return pd.concat(split_data).reset_index(drop=True)

def Case_definetion(df, mode):
    data = df.copy()
    data['Case'] = np.nan
    
    if mode == 'mimic':
        print('|Start MIMIC-IV process|')
        stay_id_id = 'stay_id'
    elif mode == 'eicu':
        print('|Start eICU-CRD process|')
        stay_id_id = 'patientunitstayid'
    
    case1_stay_ids = []
    for stay_id, group in data.groupby(stay_id_id):
        if any(group['Annotation'] == 'no_circ') & any(group['Annotation'] == 'circ') & any(group['Annotation'] == 'ambiguous'):
            case1_stay_ids.append(stay_id)
    data = data[data[stay_id_id].isin(case1_stay_ids)].reset_index(drop=True)
    
    print('Extract Case 1, Case 2')        
    case2 = get_case2(data, event='circ', mode = mode)
    
    case1_idx = case2[(case2['Annotation']=='no_circ')&(case2['Shock_next_8h']==0)].index
    amb_case1_idx = case2[(case2['Annotation']=='ambiguous')&(case2['Shock_next_8h']==0)].index
    case2_idx = case2[(case2['Annotation']=='no_circ')&(case2['Shock_next_8h']==1)].index
    amb_case2_idx = case2[(case2['Annotation']=='ambiguous')&(case2['Shock_next_8h']==1)].index
    
    case1_case2 = case2.copy()
    case1_case2['Case'].loc[case1_idx] = 1
    case1_case2['Case'].loc[amb_case1_idx] = 1
    case1_case2['Case'].loc[case2_idx] = 2
    case1_case2['Case'].loc[amb_case2_idx] = 2
    case1_case2['Case'] = case1_case2['Case'].fillna('event') 
    case1_case2['progress'] = 0
    case1_case2['after_shock_annotation'] = 'before_experience_shock'
    print('--------------')
    
    print('Extract Case 3, Case 4')        
    case3_4 = get_case3_4(data, case1_stay_ids, mode) 
    case3_case_4 = Case3_Case4_labeler(case3_4, mode) # psuedo
    case3_case_4['after_shock_annotation'] = 'Psuedo' # psuedo
    case3_case_4['progress'] = 1 # psuedo
    
    return case1_case2, case3_case_4

def Lactate_up(df, mode = 'mimic'):
    data = df.copy().reset_index(drop=True)
    data['lactate_up'] = 0
    if mode == 'mimic':
        stay_id_id = 'stay_id'
    else:
        stay_id_id = 'patientunitstayid'
    
    for stay in tqdm(data[stay_id_id].unique()):
        interest = data[data[stay_id_id]==stay].sort_values(by='Time_since_ICU_admission')
        idx = interest.index
        interest.set_index('Time_since_ICU_admission', inplace=True)
        interest['Lactate_3hr_avg'] = interest['Lactate'].rolling(window=3, min_periods=1).mean().shift(1) 

        interest['lactate_up'] = (interest['Lactate'] > interest['Lactate_3hr_avg']).astype(int)
        interest['Lactate_down'] = (interest['Lactate'] < interest['Lactate_3hr_avg']).astype(int)
        interest = interest.reset_index()
        interest = interest.fillna(0)
               
        
        data.loc[idx, 'lactate_up'] = interest['lactate_up'].values
        data.loc[idx, 'lactate_down'] = interest['Lactate_down'].values
    return data


def recov_Annotation(data, mode):
    targ = data.copy()
    # targ['progress'] = 0
    
    if mode == 'mimic':
        stay_id_id = 'stay_id'
    else:
        stay_id_id = 'patientunitstayid'
    
    
    for stay_trajectory in tqdm(targ[stay_id_id].unique()):
        interest = targ[targ[stay_id_id] == stay_trajectory]
        
        for idx, row in interest.iterrows():
            if row['INDEX']!='CASE1_CASE2_DF':
                recover_cond = (row['MAP_3hr_avg'] >= 65.0) & ((row['Lactate'] < 2) | (row['lactate_down']==1)) 
                
                if recover_cond:
                    targ.at[idx, 'after_shock_annotation'] = 'recov'
                else:
                    targ.at[idx, 'after_shock_annotation'] = 'not_recov'
            else:
                targ.at[idx, 'after_shock_annotation'] = 'before_experience_shock'

    return targ

def get_case3_4(target, case1_stay_ids, mode = 'mimic'):
    
    data = target.copy()
    
    split_data = []
    current_part = []
    event_occurred = False
    
    if mode == 'mimic':
        stay_id_id = 'stay_id'
    else:
        stay_id_id = 'patientunitstayid'
        
    search_stay_id = set(data[stay_id_id].unique())
    
    for stayid in search_stay_id:
        dataset = data[data[stay_id_id]==stayid]
        
        for index, row in dataset.iterrows():
            
            if event_occurred:
                current_part.append(row)
                if row['Annotation']=='no_circ':
                    split_data.append(pd.DataFrame(current_part))
                    event_occurred = False
                    current_part = []
                    break
                
                elif index == dataset.index[-1]:
                    if row['Annotation'] == 'ambiguous':
                        event_occurred = False
                        current_part = []
                        break
                    
                elif index == dataset.index[-1]:
                    if row['Annotation'] == 'circ':
                        event_occurred = False
                        current_part = []
                        break
                    
            else:
                if row['Annotation']=='circ':
                    current_part.append(row)
                    event_occurred = True
                elif row['Annotation']=='ambiguous':
                    current_part.append(row)
                    
                else:
                    current_part.append(row)
                    break
                

    return split_data


def Case3_Case4_labeler(parts, mode):
    targ = parts.copy()
  
    if mode == 'mimic':
        stay_id_id = 'stay_id'
    else:
        stay_id_id = 'patientunitstayid'
  
    for stayid in tqdm(targ[stay_id_id].unique()):
        interest = targ[targ[stay_id_id]==stayid]
        interest['Case'] = np.nan
        interest['endpoint_window'] = interest['Time_since_ICU_admission'] + 1

        for idx, row in interest.iterrows():
            current_time = row['Time_since_ICU_admission']
            endpoint_window = row['endpoint_window']

            future_rows = interest[(interest['Time_since_ICU_admission'] > current_time) & (interest['Time_since_ICU_admission'] <= endpoint_window)]

            if any(future_rows['progress'] == 'not_recov'):
                targ.loc[idx, 'Case'] = 4
            else:
                targ.loc[idx, 'Case'] = 3

    return targ.reset_index(drop=True)


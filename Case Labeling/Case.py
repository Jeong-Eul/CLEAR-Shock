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

def update_ambiguous_to_amb_circ(arr):
    updated_arr = np.array(arr, copy=True)
    n = len(arr)

    for i in range(1, n - 1):
        if arr[i] == 'ambiguous':
            # Check if there's 'circ' before and after the 'ambiguous' sequence
            if arr[i - 1] == 'circ' and 'circ' in arr[i + 1:]:
                # Find the end of the 'ambiguous' sequence
                end = i
                while end < n and arr[end] == 'ambiguous':
                    end += 1
                
                # Update the 'ambiguous' sequence to 'amb_circ'
                updated_arr[i:end] = 'amb_circ'

                # Skip the already updated part
                i = end

    return updated_arr

# def get_case3_case4(df, mode, case1_stay_ids):
#     targ = df.copy(deep=True)
    
#     if mode == 'mimic':
#         stay_id_id = 'stay_id'
#     elif mode == 'eicu':
#         stay_id_id = 'patientunitstayid'
        
#     search_stay_id = set(targ[stay_id_id].unique()) - set(case1_stay_ids)
    
#     for stay_id in tqdm(search_stay_id):
#         stay_df = targ[targ[stay_id_id] == stay_id].sort_values(by='Time_since_ICU_admission')
        
#         sample = stay_df[stay_df[stay_id_id]==stay_id]
        
#         index = sample.index
#         annotation_arr = sample['Annotation'].values

#         new_annotation_arr = update_ambiguous_to_amb_circ(annotation_arr)
#         stay_df['Annotation'].loc[index] = new_annotation_arr
        
#         for idx, row in stay_df.iterrows():
#             if (row['Annotation'] == 'circ')|(row['Annotation'] == 'amb_circ'):
#                 current_time = row['Time_since_ICU_admission']
#                 endpoint_window = current_time + 24
    
#                 window = stay_df[(stay_df['Time_since_ICU_admission'] > current_time) & (stay_df['Time_since_ICU_admission'] <= endpoint_window)]
#                 if len(window) > 0:

#                     counts = window['Annotation'].value_counts()
#                     count_amb_no_circ = counts.get('ambiguous', 0) + counts.get('no_circ', 0) + counts.get('amb_circ', 0)
#                     count_amb_circ = counts.get('ambiguous', 0) + counts.get('circ', 0) + counts.get('amb_circ', 0)
#                     total_state = len(window)

#                     recovery_ratio = count_amb_no_circ / total_state
#                     no_recovery_ratio = count_amb_circ / total_state

#                     if recovery_ratio >= 0.7 and counts.get('no_circ', 0) > 0:
#                         targ.loc[idx, 'Case'] = 3
#                     elif no_recovery_ratio >= 0.7 and counts.get('circ', 0) > 0:
#                         targ.loc[idx, 'Case'] = 4
                
#     return targ.dropna().reset_index(drop=True)

def get_case3_4(target, case1_stay_ids, mode = 'mimic'):
    
    data = target.copy()
    
    split_data = []
    current_part = []
    event_occurred = False
    
    if mode == 'mimic':
        stay_id_id = 'stay_id'
    else:
        stay_id_id = 'patientunitstayid'
        
    # all_circ_stay_ids = []
    # for stay_id, group in data.groupby(stay_id_id):
    #     if all(group['Annotation'] == 'circ'):
    #         all_circ_stay_ids.append(stay_id)
    
        
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

    return split_data


def recov_Annotation(data, mode):
    targ = data.copy()
    targ['progress'] = 0
    
    if mode == 'mimic':
        stay_id_id = 'stay_id'
    else:
        stay_id_id = 'patientunitstayid'
    
    
    for stay_trajectory in tqdm(targ[stay_id_id].unique()):
        interest = targ[targ[stay_id_id] == stay_trajectory]
        
        for idx, row in interest.iterrows():
            
            
            current_time = row['Time_since_ICU_admission']
            pastpoint_window_1h = current_time - 1
        
            relevant_rows = interest[(interest['Time_since_ICU_admission'] <= current_time) & (interest['Time_since_ICU_admission'] >= pastpoint_window_1h)]
            # no_recover_cond = (relevant_rows['MAP'] >= 65.0).all() & ((row['Lactate'] >= 2) | (row['lactate_up']==1))
            recover_cond = (relevant_rows['MAP'] >= 65.0).all() & ((row['Lactate'] <= 2) | (row['lactate_up']==0))
            
            if recover_cond:
                targ.at[idx, 'progress'] = 'recov'
            else:
                targ.at[idx, 'progress'] = 'not_recov'

    return targ

def Case3_Case4_labeler(parts, mode):
    targ = parts.copy()
  
    if mode == 'mimic':
        stay_id_id = 'stay_id'
    else:
        stay_id_id = 'patientunitstayid'
  
    for stayid in tqdm(targ[stay_id_id].unique()):
        interest = targ[targ[stay_id_id]==stayid]
        interest['Case'] = np.nan
        interest['endpoint_window'] = interest['Time_since_ICU_admission'] + 3

        for idx, row in interest.iterrows():
            current_time = row['Time_since_ICU_admission']
            endpoint_window = row['endpoint_window']

            future_rows = interest[(interest['Time_since_ICU_admission'] > current_time) & (interest['Time_since_ICU_admission'] <= endpoint_window)]

            if all(future_rows['progress'] == 'recov'):
                targ.loc[idx, 'Case'] = 3
            else:
                targ.loc[idx, 'Case'] = 4

    return targ.reset_index(drop=True)

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
        
        Lactate = interest['Lactate']
        Lactate_up_corrected = []
        
        previous_real_value = Lactate.iloc[0]
        Lactate_up_corrected.append(0)
        
        for i in range(1, len(Lactate)):
            if Lactate.iloc[i] != Lactate.iloc[i-1]:
                
                if Lactate.iloc[i] > previous_real_value:
                    Lactate_up_corrected.append(1)
                    
                else:
                    Lactate_up_corrected.append(0)
                    previous_real_value = Lactate.iloc[i]
            else:
                Lactate_up_corrected.append(Lactate_up_corrected[-1])
               
        
        data.loc[idx, 'lactate_up'] = np.array(Lactate_up_corrected)
    return data


def recovery_filter_stay_ids(group):
    if all(group.tail(3)['progress'] == 2):
        return True
    return False

def detorioration_filter_stay_ids(group):
    if all(group.tail(3)['progress'] == 1):
        return True
    return False


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
        if all(group['Annotation'] == 'no_circ'):
            case1_stay_ids.append(stay_id)
    
    print('Extract Case 1, Case 2')        
    case2 = get_case2(data, event='circ', mode = mode)
    # case2['Case'] = 2  
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
    case1_case2 = Lactate_up(case1_case2, mode)
    case1_case2['progress'] = 0
    print('--------------')
    
    print('Extract Case 3, Case 4')        
    case3_4 = get_case3_4(data, case1_stay_ids, mode) #case3_4에서는 마지막에 no_circ 관측치는 event가 되는 거임
    case3_case4 = Lactate_up(pd.concat(case3_4), mode)
    case3_case4 = recov_Annotation(case3_case4, mode)
    case3_case_4 = Case3_Case4_labeler(case3_case4, mode)
    case3_case_4['progress'] = case3_case_4['progress'].map({'not_recov': 1, 'recov': 2})
    
    return case1_case2, case3_case_4
    
    
    # 여기서부터는 다시 작성해야함 0227
    # case3_4 = recov_Annotation(case3_4)
    
    # case3_case_4 = Case3_Case4_labeler(case3_4)
    # print('--------------')
    
    # print('MAKE progress') 
    # case1_case2['progress'] = 0
    # case3_case_4['progress'] = case3_case_4['progress'].map({'not_recov': 1, 'recov': 2})
    
    # print('--------------')
    
    # print('Select target cohort')
    # recover_grouped = case3_case_4.groupby(stay_id_id)
    # valid_stay_ids = [name for name, group in recover_grouped if recovery_filter_stay_ids(group)]
    # recoverset = case3_case_4[case3_case_4[stay_id_id].isin(valid_stay_ids)].copy()
    
    # recoverclean = recoverset.copy()
    # for stayid in recoverclean[stay_id_id].unique():
    #     targ_index = recoverclean[recoverclean[stay_id_id]==stayid].index
        
    #     if recoverclean.loc[targ_index[-1]]['Case'] == 4:
    #         recoverclean.drop(index=targ_index[-1], axis = 0, inplace = True)
    #     else:
    #         pass
    
    # deterioation_grouped = case3_case_4.groupby(stay_id_id)
    
    # valid_stay_ids = [name for name, group in deterioation_grouped if detorioration_filter_stay_ids(group)]
    # detorioration = case3_case_4[case3_case_4[stay_id_id].isin(valid_stay_ids)].copy()
    
    # print('Finish, ....')
    # print('--------------')
    # return case1_case2, case3_case_4


# def filter_cohort(mimic, eicu):
    
#     filtered_stay_ids = []

#     for stay_id, group in mimic.groupby('stay_id'):
#         if group['Annotation'].iloc[0] == 'no_circ':
#             filtered_stay_ids.append(stay_id)

#     filtered_stay_ids_array = np.array(filtered_stay_ids)
#     mimic = mimic[mimic['stay_id'].isin(filtered_stay_ids_array)].reset_index(drop=True)
    
#     filtered_stay_ids = []

#     for stay_id, group in eicu.groupby('patientunitstayid'):
#         if group['Annotation'].iloc[0] == 'no_circ':
#             filtered_stay_ids.append(stay_id)

#     filtered_stay_ids_array = np.array(filtered_stay_ids)
#     eicu = eicu[eicu['patientunitstayid'].isin(filtered_stay_ids_array)].reset_index(drop=True)
    

#     mimic_circ_ids = mimic[(mimic['Annotation'] == 'circ') | (mimic['Annotation'] == 'ambiguous')]['stay_id'].unique()
#     eicu_circ_ids = eicu[(eicu['Annotation'] == 'circ') | (eicu['Annotation'] == 'ambiguous')]['patientunitstayid'].unique()
 
#     return mimic[mimic['stay_id'].isin(mimic_circ_ids)].reset_index(drop=True), eicu[eicu['patientunitstayid'].isin(eicu_circ_ids)].reset_index(drop=True)



# def define_ambcirc(df, mode):
#     data = df.copy()
    
#     if mode == 'mimic':
#         stay_id_id = 'stay_id'
#     elif mode == 'eicu':
#         stay_id_id = 'patientunitstayid'
    
#     # 모든 stay가 circ인 경우 제외    
#     selected_stay_ids = []
#     for stay_id, group in data.groupby(stay_id_id):
#         if all(group['Annotation'] == 'circ'):
#             selected_stay_ids.append(stay_id)
            
#     data = data[~(data[stay_id_id].isin(selected_stay_ids))].reset_index(drop=True)

#     for stay_id in tqdm(data[stay_id_id].unique()):
        
#         sample = data[data[stay_id_id]==stay_id]
        
#         index = sample.index
#         annotation_arr = sample['Annotation'].values
        
#         new_annotation_arr = update_ambiguous_to_amb_circ(annotation_arr)
#         data['Annotation'].loc[index] = new_annotation_arr
        
#     return data.reset_index(drop=True)


# def early_event_prediction_label(df):
    
#     data = df.copy()
#     data['classes'] = 'undefined'

#     class_unde = data[(data['Shock_next_12h']==0) & (data['Annotation']=='ambiguous')].index
#     data.loc[class_unde,'classes'] = 0

#     class1 = data[(data['Shock_next_12h']==0) & (data['Annotation']=='no_circ')].index
#     data.loc[class1,'classes'] = 1
    
   
    
#     ## 모두 case 2에 해당하지만 적절한 학습을 위해 label을 다르게 부여
    
#     class_ambcirc = data[(data['Shock_next_12h']==1) & (data['Annotation']=='ambiguous')].index
#     data.loc[class_ambcirc,'classes'] = 2
    
#     class2 = data[(data['Shock_next_12h']==1) & (data['Annotation']=='no_circ')].index
#     data.loc[class2,'classes'] = 3
    
#     return data.reset_index(drop=True)


# def optimized_recovered_labeler(df, mode):
#     targ = df.copy(deep=True)
    
#     amb_circ = targ[(targ['Shock_next_12h']==1) & (targ['Annotation']=='amb_circ')].index
#     targ.loc[amb_circ,'classes'] = 6
    
#     if mode == 'mimic':
#         stay_id_id = 'stay_id'
#     elif mode == 'eicu':
#         stay_id_id = 'patientunitstayid'
    
#     for stay_id in tqdm(targ[stay_id_id].unique()):
#         stay_df = targ[targ[stay_id_id] == stay_id].sort_values(by='Time_since_ICU_admission')
#         for idx, row in stay_df.iterrows():
#             if (row['Annotation'] == 'circ'):
#                 current_time = row['Time_since_ICU_admission']
#                 endpoint_window = current_time + 24
    
#                 window = stay_df[(stay_df['Time_since_ICU_admission'] > current_time) & (stay_df['Time_since_ICU_admission'] <= endpoint_window)]
#                 if len(window) > 0:

#                     counts = window['Annotation'].value_counts()
#                     count_amb_no_circ = counts.get('ambiguous', 0) + counts.get('no_circ', 0) + counts.get('amb_circ', 0)
#                     count_amb_circ = counts.get('ambiguous', 0) + counts.get('circ', 0) + counts.get('amb_circ', 0)
#                     total_state = len(window)

#                     recovery_ratio = count_amb_no_circ / total_state
#                     no_recovery_ratio = count_amb_circ / total_state

#                     if recovery_ratio >= 0.7 and counts.get('no_circ', 0) > 0:
#                         targ.loc[idx, 'classes'] = 4
#                     elif no_recovery_ratio >= 0.7 and counts.get('circ', 0) > 0:
#                         targ.loc[idx, 'classes'] = 5

#     return targ.reset_index(drop=True)


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


def del_noise(df, mode):
    data = df.copy()
    
    if mode == 'mimic':
        stay_id_id = 'stay_id'
    elif mode == 'eicu':
        stay_id_id = 'patientunitstayid'
    
    print('step1 모든 관측치가 amb ,0, 0인 stay 제거')
    #모든 관측치가 amb , 0, 0인 경우 제외
    sub_view = data[[stay_id_id, 'Time_since_ICU_admission', 'Annotation', 'Shock_next_12h', 'classes']]
    
    filtered_df = sub_view[(sub_view['Annotation'] == 'ambiguous') & 
                        (sub_view['Shock_next_12h'] == 0) & 
                        (sub_view['classes'] == 0)]

    matching_stay_ids = []

    for stay_id in filtered_df[stay_id_id].unique():
        if filtered_df[filtered_df[stay_id_id] == stay_id].shape[0] == sub_view[sub_view[stay_id_id] == stay_id].shape[0]:
            matching_stay_ids.append(stay_id)
    
    data = data[~(data[stay_id_id].isin(matching_stay_ids))]
    print('step1 완료')
    
    print('step2 forward fill 잔재 제거(특정 시점부터 마지막까지 모두 amb-0-0인 경우)')
    # 조건을 벡터화하여 계산
    condition_met = (data['Annotation'] == 'ambiguous') & \
                    (data['Shock_next_12h'] == 0) & \
                    (data['classes'] == 0)

    # 각 stay_id에 대해 첫 조건 불만족 지점 찾기
    # Use groupby with stay_id_id and then apply a lambda function to find the min index for each group
    first_not_met_index = data[~condition_met].groupby(stay_id_id).apply(lambda x: x.index.min())
    
    sub_view = data[[stay_id_id, 'Time_since_ICU_admission', 'Annotation', 'Shock_next_12h', 'classes']]
    
    # Assuming data and first_not_met_index are already defined

    # Initialize a flag column with all True (meaning keep all rows initially)
    sub_view['keep_row'] = True

    # Iterate over first_not_met_index to update the flag column
    for stay_id, index in first_not_met_index.items():
        condition = (sub_view[stay_id_id] == stay_id) & (sub_view.index >= index) & condition_met
        sub_view.loc[condition, 'keep_row'] = False

    # Filter the sub_viewFrame based on the flag column
    sub_view = sub_view[sub_view['keep_row']]

    # Drop the flag column if it's no longer needed
    sub_view.drop(columns=['keep_row'], inplace=True)
    
    data = data[data.index.isin(sub_view.index)].reset_index(drop=True)
    
    print('step2 완료')
    
    # print('step3 한번이라도 circulatory failure event가 발생하지 않은 stay 제거')
    # sub_view = data[[stay_id_id, 'Time_since_ICU_admission', 'MAP', 'Lactate', 'vasoactive/inotropic','Annotation', 'Shock_next_12h', 'classes']]
    
    # filtered_stay_ids = sub_view.groupby(stay_id_id).filter(lambda x: (x['Annotation'] == 'no_circ').all())

    # unique_stay_ids = filtered_stay_ids[stay_id_id].unique()
    
    # filtered_stay_ids_circ = sub_view.groupby(stay_id_id).filter(lambda x: ~x['Annotation'].str.contains(r'\bcirc\b', regex=True).any())
    
    # unique_stay_ids_circ = filtered_stay_ids_circ[stay_id_id].unique()
    
    # del_stayid = list(unique_stay_ids) + list(unique_stay_ids_circ)
    
    # print('step3 완료, 제거 stay 수',len(set(del_stayid)))

    # data = data[~(data[stay_id_id].isin(set(del_stayid)))].reset_index(drop=True)
    
    return data


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
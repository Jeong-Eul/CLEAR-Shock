import time
import numpy as np
import pandas as pd

import Case
from imp import reload
reload(Case)
from tqdm import tqdm

mimic_path = '/Users/DAHS/Desktop/ECP_CONT/ECP_SCL/Make Derived Variable/mimic_df_cp.csv.gz'
eicu_path = '/Users/DAHS/Desktop/ECP_CONT/ECP_SCL/Make Derived Variable/eicu_df_cp.csv.gz'

def number(df, mode):
    
    if mode == 'mimic':
        print('number of subject :', len(df.drop_duplicates(subset=["subject_id"])))
        print('number of stay :', len(df.drop_duplicates(subset=["stay_id"])))
        
    else:
        print('number of subject :', len(df.drop_duplicates(subset=["uniquepid"])))
        print('number of stay :', len(df.drop_duplicates(subset=["patientunitstayid"])))
        

mimic = pd.read_csv(mimic_path, compression = 'gzip')
eicu = pd.read_csv(eicu_path, compression = 'gzip')

# Annotation and Modifying Preprocessing error
mimic_annoted = Case.annotation(mimic, 'mimic')
mimic_annoted = mimic_annoted.reset_index(drop=True)

analysis_cohort_for_mimic = []

idx = mimic_annoted[(mimic_annoted['ABPd']==75.00)|(mimic_annoted['ABPs']==125.00)].index

mimic_annoted['MAP_fillna'].loc[idx]=1
mimic_annoted['ABPd_fillna'].loc[idx]=1
mimic_annoted['ABPs_fillna'].loc[idx]=1

lactic_idx = mimic_annoted[mimic_annoted['Lactate']==1.1].index
mimic_annoted['Lactate_fillna'].loc[lactic_idx]=1
mimic_labeled = Case.optimized_shock_labeler(mimic_annoted, 'mimic')


eicu_annoted = Case.annotation(eicu, 'eicu')
eicu_annoted = eicu_annoted.reset_index(drop=True)

analysis_cohort_for_eicu = []

idx = eicu_annoted[(eicu_annoted['ABPd']==75.00)|(eicu_annoted['ABPs']==125.00)].index

eicu_annoted['MAP_fillna'].loc[idx]=1
eicu_annoted['ABPd_fillna'].loc[idx]=1
eicu_annoted['ABPs_fillna'].loc[idx]=1

lactic_idx = eicu_annoted[eicu_annoted['Lactate']==1.1].index
eicu_annoted['Lactate_fillna'].loc[lactic_idx]=1
eicu_labeled = Case.optimized_shock_labeler(eicu_annoted, 'eicu')

mimic_labeled['PaO2/FiO2'] = mimic_labeled['PaO2'] / ((mimic_labeled['FIO2 (%)']+ 0.00000001)/100)
eicu_labeled['PaO2/FiO2'] = eicu_labeled['PaO2'] / ((eicu_labeled['FIO2 (%)'] + 0.00000001)/100)


# Noise removal
invalid_columns_mimic, _ = Case.find_invalid_columns(mimic_labeled)
invalid_columns_eicu, _ = Case.find_invalid_columns(eicu_labeled)

m_labeled = Case.replace_inf_with_previous(mimic_labeled, invalid_columns_mimic)
e_labeled = Case.replace_inf_with_previous(eicu_labeled, invalid_columns_eicu)


# Case labeling
m_case1_case2, m_case3_case4 = Case.Case_definetion(m_labeled, 'mimic')
e_case1_case2, e_case3_case4 = Case.Case_definetion(e_labeled, 'eicu')

m_case1_case2['INDEX'] = 'CASE1_CASE2_DF'
m_case3_case4['INDEX'] = 'CASE3_CASE4_DF'

e_case1_case2['INDEX'] = 'CASE1_CASE2_DF'
e_case3_case4['INDEX'] = 'CASE3_CASE4_DF'

# before onset of shock, remove stay which are not measured ABPd and ABPs
# This process applyed only for event set
mode = 'mimic'
if mode == 'mimic':
        print('|Start MIMIC-IV process|')
        stay_id_id = 'stay_id'
elif mode == 'eicu':
    print('|Start eICU-CRD process|')
    stay_id_id = 'patientunitstayid'

mimic_event = m_case1_case2[m_case1_case2['Case']=='event']
drop_stay = []
print('event가 있는 코호트 수:')
number(mimic_event, 'mimic') 
for stay in mimic_event[stay_id_id].unique():
    view = m_case1_case2[m_case1_case2[stay_id_id]==stay]
    view = view[~(view['Case']=='event')]
    if all(view['ABPd_fillna']==1) | all(view['ABPs_fillna']==1) | all(view['Lactate_fillna']==1):
        drop_stay.append(stay)

filtered = m_case1_case2[~(m_case1_case2['stay_id'].isin(drop_stay))]
target_m_case1_case2 = filtered[filtered.stay_id.isin(mimic_event[stay_id_id].unique())]

mode = 'eicu'

eicu_event = e_case1_case2[e_case1_case2['Case']=='event']
drop_stay = []
print('event가 있는 코호트 수:', len(eicu_event[stay_id_id].unique()))
number(eicu_event, 'eicu') 
for stay in eicu_event[stay_id_id].unique():
    view = e_case1_case2[e_case1_case2[stay_id_id]==stay]
    view = view[~(view['Case']=='event')]
    if all(view['ABPd_fillna']==1) | all(view['ABPs_fillna']==1) | all(view['Lactate_fillna']==1):
        drop_stay.append(stay)

filtered = e_case1_case2[~(e_case1_case2[stay_id_id].isin(drop_stay))]
target_e_case1_case2 = filtered[filtered[stay_id_id].isin(eicu_event[stay_id_id].unique())]

mimic_analysis = pd.concat([target_m_case1_case2, m_case3_case4[m_case3_case4.stay_id.isin(target_m_case1_case2.stay_id.unique())]], axis = 0).reset_index(drop=True)
eicu_analysis = pd.concat([target_e_case1_case2, e_case3_case4[e_case3_case4.patientunitstayid.isin(target_e_case1_case2.patientunitstayid.unique())]], axis = 0).reset_index(drop=True)


# ethnicity merging(MIMIC-IV, eICU)
mimic_ethnicity = {'AMERICAN INDIAN/ALASKA NATIVE': 0, 'ASIAN': 1,
 'ASIAN - ASIAN INDIAN': 2, 'ASIAN - CHINESE': 3,
 'ASIAN - KOREAN': 4, 'ASIAN - SOUTH EAST ASIAN': 5,
 'BLACK/AFRICAN': 6, 'BLACK/AFRICAN AMERICAN': 7,
 'BLACK/CAPE VERDEAN': 8, 'BLACK/CARIBBEAN ISLAND': 9,
 'HISPANIC OR LATINO': 10, 'HISPANIC/LATINO - CENTRAL AMERICAN': 11,
 'HISPANIC/LATINO - COLUMBIAN': 12, 'HISPANIC/LATINO - CUBAN': 13,
 'HISPANIC/LATINO - DOMINICAN': 14, 'HISPANIC/LATINO - GUATEMALAN': 15,
 'HISPANIC/LATINO - HONDURAN': 16, 'HISPANIC/LATINO - MEXICAN': 17,
 'HISPANIC/LATINO - PUERTO RICAN': 18, 'HISPANIC/LATINO - SALVADORAN': 19, 
 'MULTIPLE RACE/ETHNICITY': 20, 'NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER': 21, 
 'OTHER': 22, 'PATIENT DECLINED TO ANSWER': 23, 'PORTUGUESE': 24, 'SOUTH AMERICAN': 25,
 'UNABLE TO OBTAIN': 26, 'UNKNOWN': 27, 'WHITE': 28, 'WHITE - BRAZILIAN': 29,
 'WHITE - EASTERN EUROPEAN': 30, 'WHITE - OTHER EUROPEAN': 31, 'WHITE - RUSSIAN': 32}

eicu_ethnicity = {'0': 0, 'African American': 1, 'Asian': 2, 
                  'Caucasian': 3, 'Hispanic': 4, 'Native American': 5, 'Other/Unknown': 6}

new_mimic_ethnicity = {v:k for k,v in mimic_ethnicity.items()}
new_eicu_ethnicity = {v:k for k,v in eicu_ethnicity.items()}

mimic_analysis['ethnicity'] = mimic_analysis['ethnicity'].map(new_mimic_ethnicity)
eicu_analysis['ethnicity'] = eicu_analysis['ethnicity'].map(new_eicu_ethnicity)

A = {'AMERICAN INDIAN/ALASKA NATIVE': 1, 'ASIAN': 2, 'ASIAN - ASIAN INDIAN': 2, 
 'ASIAN - CHINESE': 2, 'ASIAN - KOREAN': 2, 'ASIAN - SOUTH EAST ASIAN': 2, 
 'BLACK/AFRICAN': 5, 'BLACK/AFRICAN AMERICAN': 5, 'BLACK/CAPE VERDEAN': 6, 
 'BLACK/CARIBBEAN ISLAND': 6, 'HISPANIC OR LATINO': 4, 'HISPANIC/LATINO - CENTRAL AMERICAN': 4, 
 'HISPANIC/LATINO - COLUMBIAN': 4, 'HISPANIC/LATINO - CUBAN': 4, 'HISPANIC/LATINO - DOMINICAN': 4, 
 'HISPANIC/LATINO - GUATEMALAN': 4, 'HISPANIC/LATINO - HONDURAN': 4, 'HISPANIC/LATINO - MEXICAN': 4, 
 'HISPANIC/LATINO - PUERTO RICAN': 4, 'HISPANIC/LATINO - SALVADORAN': 4, 'MULTIPLE RACE/ETHNICITY': 6, 
 'NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER': 6, 'OTHER': 6, 'PATIENT DECLINED TO ANSWER': 6, 'PORTUGUESE': 4, 
 'SOUTH AMERICAN': 5, 'UNABLE TO OBTAIN': 6, 'UNKNOWN': 6, 'WHITE': 3, 'WHITE - BRAZILIAN': 3, 
 'WHITE - EASTERN EUROPEAN': 3, 'WHITE - OTHER EUROPEAN': 3, 'WHITE - RUSSIAN': 3}

B = {'0': 6, 'African American': 1, 'Asian': 2, 'Caucasian': 3, 'Hispanic': 4, 'Native American': 5, 'Other/Unknown': 6}

mimic_analysis['ethnicity'] = mimic_analysis['ethnicity'].map(A)
eicu_analysis['ethnicity'] = eicu_analysis['ethnicity'].map(B)

# Derived Variable part 2

mimic_analysis['over_lactic'] = (mimic_analysis['Lactate'] >= 2).astype(int)
eicu_analysis['over_lactic'] = (eicu_analysis['Lactate'] >= 2).astype(int)

mimic_result  = pd.DataFrame()

for stay in mimic_analysis['stay_id'].unique():
    view = mimic_analysis[mimic_analysis['stay_id']==stay]

    view.set_index('Time_since_ICU_admission', inplace=True)
    view['MAP_3hr_avg'] = view['MAP'].rolling(window=4, min_periods=1).mean().shift(1) 

    view['MAP_increase'] = (view['MAP'] > view['MAP_3hr_avg']).astype(int)
    view['MAP_decrease'] = (view['MAP'] < view['MAP_3hr_avg']).astype(int)
    view = view.reset_index()

    view = view.fillna(0)
    
    mimic_result = pd.concat([mimic_result, view], axis = 0)
    
eicu_result  = pd.DataFrame()

for stay in eicu_analysis['patientunitstayid'].unique():
    view = eicu_analysis[eicu_analysis['patientunitstayid']==stay]

    view.set_index('Time_since_ICU_admission', inplace=True)
    view['MAP_3hr_avg'] = view['MAP'].rolling(window=4, min_periods=1).mean().shift(1) 

    view['MAP_increase'] = (view['MAP'] > view['MAP_3hr_avg']).astype(int)
    view['MAP_decrease'] = (view['MAP'] < view['MAP_3hr_avg']).astype(int)
    view = view.reset_index()

    view = view.fillna(0)
    
    eicu_result = pd.concat([eicu_result, view], axis = 0)
    

mimic_result_lac = Case.Lactate_up(mimic_result, 'mimic')
eicu_result_lac = Case.Lactate_up(eicu_result, 'eicu')


# Continuation or Improving state definition

mimic_result_lac_3_4 = Case.recov_Annotation(mimic_result_lac, 'mimic')
eicu_result_lac_3_4 = Case.recov_Annotation(eicu_result_lac, 'eicu')


def Case3_Case4_labeler(parts, mode):
    targ = parts.copy().reset_index(drop=True)
    if mode == 'mimic':
        stay_id_id = 'stay_id'
    else:
        stay_id_id = 'patientunitstayid'
    for stayid in tqdm(targ[stay_id_id].unique()):
        interest = targ[targ[stay_id_id]==stayid]
        interest['endpoint_window'] = interest['Time_since_ICU_admission'] + 1
        
        for idx, row in interest.iterrows():
            if row['INDEX']!='CASE1_CASE2_DF':
                current_time = row['Time_since_ICU_admission']
                endpoint_window = row['endpoint_window']

                future_rows = interest[(interest['Time_since_ICU_admission'] > current_time) & (interest['Time_since_ICU_admission'] <= endpoint_window)]

                if any(future_rows['after_shock_annotation'] == 'not_recov'):
                    targ.loc[idx, 'Case'] = 4
                else:
                    targ.loc[idx, 'Case'] = 3

    return targ.reset_index(drop=True)

m_modified = Case3_Case4_labeler(mimic_result_lac_3_4, 'mimic')
e_modified = Case3_Case4_labeler(eicu_result_lac_3_4, 'eicu')


# Not event set preprocessing

case1_m = []
for stay_id, group in m_labeled.groupby('stay_id'):
    if all(group['Annotation'] == 'no_circ'):
        case1_m.append(stay_id)
no_event_mimic = m_labeled[m_labeled['stay_id'].isin(case1_m)].reset_index(drop=True)

case1_e = []
for stay_id, group in e_labeled.groupby('patientunitstayid'):
    if all(group['Annotation'] == 'no_circ'):
        case1_e.append(stay_id)
no_event_eicu = e_labeled[e_labeled['patientunitstayid'].isin(case1_e)].reset_index(drop=True)

no_event_mimic['ethnicity'] = no_event_mimic['ethnicity'].map(new_mimic_ethnicity)
no_event_eicu['ethnicity'] = no_event_eicu['ethnicity'].map(new_eicu_ethnicity)

no_event_mimic['ethnicity'] = no_event_mimic['ethnicity'].map(A)
no_event_eicu['ethnicity'] = no_event_eicu['ethnicity'].map(B)

# Derived variable part 2(for not event set)

no_event_mimic['over_lactic'] = (no_event_mimic['Lactate'] >= 2).astype(int)
no_event_eicu['over_lactic'] = (no_event_eicu['Lactate'] >= 2).astype(int)

mimic_result_no_event  = pd.DataFrame()

for stay in no_event_mimic['stay_id'].unique():
    view = no_event_mimic[no_event_mimic['stay_id']==stay]

    view.set_index('Time_since_ICU_admission', inplace=True)
    view['MAP_3hr_avg'] = view['MAP'].rolling(window=4, min_periods=1).mean().shift(1) 

    view['MAP_increase'] = (view['MAP'] > view['MAP_3hr_avg']).astype(int)
    view['MAP_decrease'] = (view['MAP'] < view['MAP_3hr_avg']).astype(int)
    view = view.reset_index()

    view = view.fillna(0)
    
    mimic_result_no_event = pd.concat([mimic_result_no_event, view], axis = 0)
    
    
eicu_result_no_event  = pd.DataFrame()

for stay in no_event_eicu['patientunitstayid'].unique():
    view = no_event_eicu[no_event_eicu['patientunitstayid']==stay]

    view.set_index('Time_since_ICU_admission', inplace=True)
    view['MAP_3hr_avg'] = view['MAP'].rolling(window=4, min_periods=1).mean().shift(1) 

    view['MAP_increase'] = (view['MAP'] > view['MAP_3hr_avg']).astype(int)
    view['MAP_decrease'] = (view['MAP'] < view['MAP_3hr_avg']).astype(int)
    view = view.reset_index()

    view = view.fillna(0)
    
    eicu_result_no_event = pd.concat([eicu_result_no_event, view], axis = 0)
    
    
mimic_result_no_event['progress'] = 0
eicu_result_no_event['progress'] = 0

m_no_event = Case.Lactate_up(mimic_result_no_event, 'mimic')
e_no_event = Case.Lactate_up(eicu_result_no_event, 'eicu')

m_no_event['Case'] = 1
e_no_event['Case'] = 1

mimic_final = pd.concat([m_no_event[m_modified.columns], m_modified], axis = 0)
eicu_final = pd.concat([e_no_event[e_modified.columns], e_modified], axis = 0)

mimic_final.to_csv('mimic_analysis.csv.gz', compression = 'gzip', index=False)
eicu_final.to_csv('eicu_analysis.csv.gz', compression = 'gzip', index=False)
import pandas as pd
import numpy as np
import time
import sys
from tqdm import tqdm
pd.set_option('mode.chained_assignment', None)

mimic_path = '/Users/DAHS/Desktop/ECP_CONT/ECP_SCL/Case Labeling/mimic_analysis.csv.gz'
eicu_path = '/Users/DAHS/Desktop/ECP_CONT/ECP_SCL/Case Labeling/eicu_analysis.csv.gz'

mimic = pd.read_csv(mimic_path, compression = 'gzip')
eicu = pd.read_csv(eicu_path, compression = 'gzip')


def main():
    print('Creating Mortality next 24h Task...')
    mortality_24h_prediction(mimic, eicu)
    print('Finish!')
    print('--------------------------------------')
    print('Creating Remain Length of stay Task...')
    remain_ratio_prediction(mimic, eicu)
    print('Finish!')
    print('--------------------------------------')
    print('Creating ARDS next 4, 8h Task...')
    ARDS_prediction(mimic, eicu)
    print('Finish!')
    print('--------------------------------------')
    
    

def ARDS_prediction(mimic, eicu):
    m_ards_df = mimic[['subject_id', 'stay_id', 'Time_since_ICU_admission', 'PEEP', 'PaO2', 'FIO2 (%)']].copy()
    e_ards_df = eicu[['uniquepid', 'patientunitstayid', 'Time_since_ICU_admission', 'PEEP', 'PaO2', 'FIO2 (%)']].copy()
    
    m_ards_df['PaO2/FiO2'] = m_ards_df['PaO2'] / (m_ards_df['FIO2 (%)']/100)
    e_ards_df['PaO2/FiO2'] = e_ards_df['PaO2'] / (e_ards_df['FIO2 (%)']/100)

    m_ards_df = Annotation_ARDS(m_ards_df, 'mimic')
    e_ards_df = Annotation_ARDS(e_ards_df, 'eicu')
    
    m_ards_4h_df = ARDS_4h_labeler(m_ards_df, mode = 'mimic')
    e_ards_4h_df = ARDS_4h_labeler(e_ards_df, mode = 'eicu')

    m_ards_8h_df = ARDS_8h_labeler(m_ards_df, mode = 'mimic')
    e_ards_8h_df = ARDS_8h_labeler(e_ards_df, mode = 'eicu')
    
    """
    patient_id | stay id | Time_since_ICU_admission | 'PEEP' | 'PaO2' | 'FIO2 (%)' | PaO2/FiO2 | Annotation_ARDS | label
    """

    m_ards_4h_df.to_csv('/Users/DAHS/Desktop/ECP_CONT/ECP_SCL/Experiment(Supervised Learning)/sub task prediction/sub_task_dataset/mimic_ards_4h.csv.gz')
    e_ards_4h_df.to_csv('/Users/DAHS/Desktop/ECP_CONT/ECP_SCL/Experiment(Supervised Learning)/sub task prediction/sub_task_dataset/eicu_ards_4h.csv.gz')

    m_ards_8h_df.to_csv('/Users/DAHS/Desktop/ECP_CONT/ECP_SCL/Experiment(Supervised Learning)/sub task prediction/sub_task_dataset/mimic_ards_8h.csv.gz')
    e_ards_8h_df.to_csv('/Users/DAHS/Desktop/ECP_CONT/ECP_SCL/Experiment(Supervised Learning)/sub task prediction/sub_task_dataset/eicu_ards_8h.csv.gz')

def remain_ratio_prediction(mimic, eicu):
    m_los_df = mimic[['subject_id', 'stay_id', 'Time_since_ICU_admission']].copy()
    e_los_df = eicu[['uniquepid', 'patientunitstayid', 'Time_since_ICU_admission']].copy()
    
    m_los_df['remain_los'] = np.nan

    for stay in m_los_df.stay_id.unique():

        interest = m_los_df[m_los_df['stay_id']==stay]
        remain_los = interest['Time_since_ICU_admission'].values[::-1]

        m_los_df.loc[interest.index, 'remain_los'] = remain_los
        
    m_los_df = m_los_df.copy()
    m_los_df.loc[m_los_df['remain_los'] < 24, 'remain_los'] = 0
    m_los_df.loc[(m_los_df['remain_los'] >= 24) & (m_los_df['remain_los'] < 48), 'remain_los'] = 1
    m_los_df.loc[(m_los_df['remain_los'] >= 48) & (m_los_df['remain_los'] < 72), 'remain_los'] = 2
    m_los_df.loc[(m_los_df['remain_los'] >= 72) & (m_los_df['remain_los'] < 96), 'remain_los'] = 3
    m_los_df.loc[(m_los_df['remain_los'] >= 96) & (m_los_df['remain_los'] < 120), 'remain_los'] = 4
    m_los_df.loc[m_los_df['remain_los'] >= 120, 'remain_los'] = 5
    
    e_los_df['remain_los'] = np.nan

    for stay in e_los_df.patientunitstayid.unique():

        interest = e_los_df[e_los_df['patientunitstayid']==stay]
        remain_los = interest['Time_since_ICU_admission'].values[::-1]

        e_los_df.loc[interest.index, 'remain_los'] = remain_los
        
    e_los_df = e_los_df.copy()
    e_los_df.loc[e_los_df['remain_los'] < 24, 'remain_los'] = 0
    e_los_df.loc[(e_los_df['remain_los'] >= 24) & (e_los_df['remain_los'] < 48), 'remain_los'] = 1
    e_los_df.loc[(e_los_df['remain_los'] >= 48) & (e_los_df['remain_los'] < 72), 'remain_los'] = 2
    e_los_df.loc[(e_los_df['remain_los'] >= 72) & (e_los_df['remain_los'] < 96), 'remain_los'] = 3
    e_los_df.loc[(e_los_df['remain_los'] >= 96) & (e_los_df['remain_los'] < 120), 'remain_los'] = 4
    e_los_df.loc[e_los_df['remain_los'] >= 120, 'remain_los'] = 5
    
    """
    patient_id | stay id | Time_since_ICU_admission | remain_los(label)
    """

    m_los_df.to_csv('/Users/DAHS/Desktop/ECP_CONT/ECP_SCL/Experiment(Supervised Learning)/sub task prediction/sub_task_dataset/mimic_remain_los.csv.gz')
    e_los_df.to_csv('/Users/DAHS/Desktop/ECP_CONT/ECP_SCL/Experiment(Supervised Learning)/sub task prediction/sub_task_dataset/eicu_remain_los.csv.gz')
        

def mortality_24h_prediction(mimic, eicu):

    ### mortality early prediction (24h)
    """

    향후 24시간 이내 사망 예측
    건강이 악화되고 있는지 미리 예측하는 Task

    """

    mimic_mort_path = '/Users/DAHS/Desktop/MIMICIV2.2_PREPROC/data/cohort/cohort_icu_mortality_0_.csv.gz'
    eicu_mort_path = '/Users/DAHS/Desktop/early_prediction_of_circ_scl/eicu-crd/preprocessing_data/cohort/cohort_.csv.gz'


    mimic_mort = pd.read_csv(mimic_mort_path, compression = 'gzip')
    eicu_mort = pd.read_csv(eicu_mort_path, compression = 'gzip')

    m_mort_df = mimic[['subject_id', 'stay_id', 'Time_since_ICU_admission']].copy()
    e_mort_df = eicu[['uniquepid', 'patientunitstayid', 'Time_since_ICU_admission']].copy()

    m_mort_df['death'] = np.nan
    death_stay_ids = mimic_mort[mimic_mort['label']==1].stay_id.unique()

    for stay in death_stay_ids:
        
        if stay in m_mort_df.stay_id.unique():
        
            interest = m_mort_df[m_mort_df['stay_id']==stay]
            death_idx = interest.index[-1]
            
            current_time = interest[interest['stay_id']==stay].iloc[-1, :]['Time_since_ICU_admission']
            idx = interest[(interest['Time_since_ICU_admission'] >= current_time-24) & (interest['Time_since_ICU_admission'] < current_time)].index
            
            m_mort_df.loc[death_idx,'death'] = 'event'
            m_mort_df.loc[idx, 'death'] = 1
        
    m_mort_df = m_mort_df.fillna(0)

    e_mort_df['death'] = np.nan
    death_stay_ids_e = eicu_mort[eicu_mort['label']==1].patientunitstayid.unique()

    for stay in death_stay_ids_e:
        
        if stay in e_mort_df.patientunitstayid.unique():
        
            interest = e_mort_df[e_mort_df['patientunitstayid']==stay]
            death_idx = interest.index[-1]
            
            current_time = interest[interest['patientunitstayid']==stay].iloc[-1, :]['Time_since_ICU_admission']
            idx = interest[(interest['Time_since_ICU_admission'] >= current_time-24) & (interest['Time_since_ICU_admission'] < current_time)].index
            
            e_mort_df.loc[death_idx,'death'] = 'event'
            e_mort_df.loc[idx, 'death'] = 1
        
    e_mort_df = e_mort_df.fillna(0)

    """
    patient_id | stay id | Time_since_ICU_admission | death(label)
    """

    m_mort_df.to_csv('/Users/DAHS/Desktop/ECP_CONT/ECP_SCL/Experiment(Supervised Learning)/sub task prediction/sub_task_dataset/mimic_mort_24h.csv.gz')
    e_mort_df.to_csv('/Users/DAHS/Desktop/ECP_CONT/ECP_SCL/Experiment(Supervised Learning)/sub task prediction/sub_task_dataset/eicu_mort_24h.csv.gz')
    

def Annotation_ARDS(df, mode):
    
    if mode == 'mimic':
        stay_id_id = 'stay_id'
    else:
        stay_id_id = 'patientunitstayid'
    
    targ = df.copy()

    targ['Annotation_ARDS'] = np.nan

    for stay_id in tqdm(targ[stay_id_id].unique()):
        stay_df = targ[targ[stay_id_id] == stay_id].sort_values(by='Time_since_ICU_admission')

        idx_ards = stay_df[(stay_df['PEEP'] >= 5.0) & (stay_df['PaO2/FiO2'] <=300)].index
        idx_no_ards = stay_df.index.difference(idx_ards)
        
        if not idx_ards.empty:
            targ.loc[idx_ards, 'Annotation_ARDS'] = 'ARDS'
        if not idx_no_ards.empty:
            targ.loc[idx_no_ards, 'Annotation_ARDS'] = 'no_ARDS'

    return targ


def ARDS_4h_labeler(df, mode):
    
    if mode == 'mimic':
        stay_id_id = 'stay_id'
    else:
        stay_id_id = 'patientunitstayid'

    targ = df.copy()
    targ['ARDS_next_12h'] = np.nan
  
    for stay_id in tqdm(targ[stay_id_id].unique()):
        stay_df = targ[targ[stay_id_id] == stay_id].sort_values(by='Time_since_ICU_admission')
        stay_df['endpoint_window'] = stay_df['Time_since_ICU_admission'] + 4

        for idx, row in stay_df.iterrows():
            current_time = row['Time_since_ICU_admission']
            endpoint_window = row['endpoint_window']

            future_rows = stay_df[(stay_df['Time_since_ICU_admission'] > current_time) & (stay_df['Time_since_ICU_admission'] <= endpoint_window)]

            if any(future_rows['Annotation_ARDS'] == 'ARDS'):
                targ.loc[idx, 'ARDS_next_4h'] = 1
            else:
                targ.loc[idx, 'ARDS_next_4h'] = 0

    return targ

def ARDS_8h_labeler(df, mode):
    
    if mode == 'mimic':
        stay_id_id = 'stay_id'
    else:
        stay_id_id = 'patientunitstayid'    

    targ = df.copy()
    targ['ARDS_next_12h'] = np.nan
  
    for stay_id in tqdm(targ[stay_id_id].unique()):
        stay_df = targ[targ[stay_id_id] == stay_id].sort_values(by='Time_since_ICU_admission')
        stay_df['endpoint_window'] = stay_df['Time_since_ICU_admission'] + 8

        for idx, row in stay_df.iterrows():
            current_time = row['Time_since_ICU_admission']
            endpoint_window = row['endpoint_window']

            future_rows = stay_df[(stay_df['Time_since_ICU_admission'] > current_time) & (stay_df['Time_since_ICU_admission'] <= endpoint_window)]

            if any(future_rows['Annotation_ARDS'] == 'ARDS'):
                targ.loc[idx, 'ARDS_next_8h'] = 1
            else:
                targ.loc[idx, 'ARDS_next_8h'] = 0

    return targ
import pandas as pd
import numpy as np


def mortality_prediction_DATA(mimic_train, mimic_valid, eicu_test):
    
    mort_mimic_path = '/Users/DAHS/Desktop/ECP_CONT/ECP_SCL/Experiment(Supervised Learning)/sub task prediction/sub_task_dataset/mimic_mort_24h.csv.gz'
    mort_eicu_path = '/Users/DAHS/Desktop/ECP_CONT/ECP_SCL/Experiment(Supervised Learning)/sub task prediction/sub_task_dataset/eicu_mort_24h.csv.gz'

    mort_mimic = pd.read_csv(mort_mimic_path, compression = 'gzip', index_col=0)
    mort_eicu = pd.read_csv(mort_eicu_path, compression = 'gzip', index_col=0)
    
    mort_mimic_train = pd.merge(mimic_train.drop(['Annotation', 'Case', 'ethnicity', 'Shock_next_8h', 'INDEX'], axis = 1), mort_mimic, how = 'left', on = ['subject_id', 'stay_id', 'Time_since_ICU_admission'])
    mort_mimic_valid = pd.merge(mimic_valid.drop(['Annotation', 'Case', 'ethnicity', 'Shock_next_8h', 'INDEX'], axis = 1), mort_mimic, how = 'left', on = ['subject_id', 'stay_id', 'Time_since_ICU_admission'])
    mort_eicu_test = pd.merge(eicu_test.drop(['Annotation', 'Case', 'ethnicity', 'Shock_next_8h', 'INDEX'], axis = 1), mort_eicu, how = 'left', on = ['uniquepid', 'patientunitstayid', 'Time_since_ICU_admission'])
    
    drop_stayid = []
    for stay_id, group in mort_mimic_train.groupby('stay_id'):
        if any(group['death'] == 'event'):
            drop_stayid.append(stay_id)
            
    for stay_id, group in mort_mimic_valid.groupby('stay_id'):
        if any(group['death'] == 'event'):
            drop_stayid.append(stay_id)
            
    for stay_id, group in mort_eicu_test.groupby('patientunitstayid'):
        if any(group['death'] == 'event'):
            drop_stayid.append(stay_id)

    mort_mimic_train = mort_mimic_train[(mort_mimic_train['stay_id'].isin(drop_stayid))].reset_index(drop=True)
    mort_mimic_valid = mort_mimic_valid[(mort_mimic_valid['stay_id'].isin(drop_stayid))].reset_index(drop=True)
    mort_eicu_test = mort_eicu_test[(mort_eicu_test['patientunitstayid'].isin(drop_stayid))].reset_index(drop=True)
    
    return mort_mimic_train, mort_mimic_valid, mort_eicu_test

def LOS_prediction_DATA(mimic_train, mimic_valid, eicu_test):
    
    los_mimic_path = '/Users/DAHS/Desktop/ECP_CONT/ECP_SCL/Experiment(Supervised Learning)/sub task prediction/sub_task_dataset/mimic_remain_los.csv.gz'
    los_eicu_path = '/Users/DAHS/Desktop/ECP_CONT/ECP_SCL/Experiment(Supervised Learning)/sub task prediction/sub_task_dataset/eicu_remain_los.csv.gz'

    los_mimic = pd.read_csv(los_mimic_path, compression = 'gzip', index_col=0)
    los_eicu = pd.read_csv(los_eicu_path, compression = 'gzip', index_col=0)
    
    los_mimic_train = pd.merge(mimic_train.drop(['Annotation', 'Case', 'ethnicity', 'Shock_next_8h', 'INDEX'], axis = 1), los_mimic, how = 'left', on = ['subject_id', 'stay_id', 'Time_since_ICU_admission'])
    los_mimic_valid = pd.merge(mimic_valid.drop(['Annotation', 'Case', 'ethnicity', 'Shock_next_8h', 'INDEX'], axis = 1), los_mimic, how = 'left', on = ['subject_id', 'stay_id', 'Time_since_ICU_admission'])
    los_eicu_test = pd.merge(eicu_test.drop(['Annotation', 'Case', 'ethnicity', 'Shock_next_8h', 'INDEX'], axis = 1), los_eicu, how = 'left', on = ['uniquepid', 'patientunitstayid', 'Time_since_ICU_admission'])
    
    return los_mimic_train, los_mimic_valid, los_eicu_test

def ARDS4_prediction_DATA(mimic_train, mimic_valid, eicu_test):
    
    ards4h_mimic_path = '/Users/DAHS/Desktop/ECP_CONT/ECP_SCL/Experiment(Supervised Learning)/sub task prediction/sub_task_dataset/mimic_ards_4h.csv.gz'
    ards4h_eicu_path = '/Users/DAHS/Desktop/ECP_CONT/ECP_SCL/Experiment(Supervised Learning)/sub task prediction/sub_task_dataset/eicu_ards_4h.csv.gz'

    ards4h_mimic = pd.read_csv(ards4h_mimic_path, compression = 'gzip', index_col=0)
    ards4h_eicu = pd.read_csv(ards4h_eicu_path, compression = 'gzip', index_col=0)
    
    ards4h_mimic.replace([np.inf, -np.inf], np.nan, inplace=True)
    ards4h_mimic.fillna(0, inplace=True) 
    
    ards4h_eicu.replace([np.inf, -np.inf], np.nan, inplace=True)
    ards4h_eicu.fillna(0, inplace=True) 
    
    ards4h_mimic_train = pd.merge(mimic_train.drop(['Annotation', 'Case', 'ethnicity', 'Shock_next_8h', 'INDEX', 'PaO2/FiO2'], axis = 1), ards4h_mimic.drop(['PEEP','PaO2', 'FIO2 (%)'], axis = 1), how = 'left', on = ['subject_id', 'stay_id', 'Time_since_ICU_admission'])
    ards4h_mimic_valid = pd.merge(mimic_valid.drop(['Annotation', 'Case', 'ethnicity', 'Shock_next_8h', 'INDEX', 'PaO2/FiO2'], axis = 1), ards4h_mimic.drop(['PEEP','PaO2', 'FIO2 (%)'], axis = 1), how = 'left', on = ['subject_id', 'stay_id', 'Time_since_ICU_admission'])
    ards4h_eicu_test = pd.merge(eicu_test.drop(['Annotation', 'Case', 'ethnicity', 'Shock_next_8h', 'INDEX', 'PaO2/FiO2'], axis = 1), ards4h_eicu.drop(['PEEP','PaO2', 'FIO2 (%)'], axis = 1), how = 'left', on = ['uniquepid', 'patientunitstayid', 'Time_since_ICU_admission'])
    
    
    drop_stayid = []
    for stay_id, group in ards4h_mimic_train.groupby('stay_id'):
        if any(group['Annotation_ARDS'] == 'ARDS'):
            drop_stayid.append(stay_id)
            
    for stay_id, group in ards4h_mimic_valid.groupby('stay_id'):
        if any(group['Annotation_ARDS'] == 'ARDS'):
            drop_stayid.append(stay_id)
            
    for stay_id, group in ards4h_eicu_test.groupby('patientunitstayid'):
        if any(group['Annotation_ARDS'] == 'ARDS'):
            drop_stayid.append(stay_id)
            
    ards4h_mimic_train = ards4h_mimic_train[(ards4h_mimic_train['stay_id'].isin(drop_stayid))].reset_index(drop=True)
    ards4h_mimic_valid = ards4h_mimic_valid[(ards4h_mimic_valid['stay_id'].isin(drop_stayid))].reset_index(drop=True)
    ards4h_eicu_test = ards4h_eicu_test[(ards4h_eicu_test['patientunitstayid'].isin(drop_stayid))].reset_index(drop=True)
    
    
    ards4h_mimic_train = get_ARDS(ards4h_mimic_train, event='ARDS', mode = 'mimic')
    ards4h_mimic_valid = get_ARDS(ards4h_mimic_valid, event='ARDS', mode = 'mimic')
    ards4h_eicu_test = get_ARDS(ards4h_eicu_test, event='ARDS', mode = 'eicu')
    
    
    return ards4h_mimic_train, ards4h_mimic_valid, ards4h_eicu_test

def ARDS8_prediction_DATA(mimic_train, mimic_valid, eicu_test):
    
    ards8h_mimic_path = '/Users/DAHS/Desktop/ECP_CONT/ECP_SCL/Experiment(Supervised Learning)/sub task prediction/sub_task_dataset/mimic_ards_8h.csv.gz'
    ards8h_eicu_path = '/Users/DAHS/Desktop/ECP_CONT/ECP_SCL/Experiment(Supervised Learning)/sub task prediction/sub_task_dataset/eicu_ards_8h.csv.gz'

    ards8h_mimic = pd.read_csv(ards8h_mimic_path, compression = 'gzip', index_col=0)
    ards8h_eicu = pd.read_csv(ards8h_eicu_path, compression = 'gzip', index_col=0)
    
    
    ards8h_mimic.replace([np.inf, -np.inf], np.nan, inplace=True)
    ards8h_mimic.fillna(0, inplace=True) 
    
    ards8h_eicu.replace([np.inf, -np.inf], np.nan, inplace=True)
    ards8h_eicu.fillna(0, inplace=True) 
    
    
    ards8h_mimic_train = pd.merge(mimic_train.drop(['Annotation', 'Case', 'ethnicity', 'Shock_next_8h', 'INDEX', 'PaO2/FiO2'], axis = 1), ards8h_mimic.drop(['PEEP','PaO2', 'FIO2 (%)'], axis = 1), how = 'left', on = ['subject_id', 'stay_id', 'Time_since_ICU_admission'])
    ards8h_mimic_valid = pd.merge(mimic_valid.drop(['Annotation', 'Case', 'ethnicity', 'Shock_next_8h', 'INDEX', 'PaO2/FiO2'], axis = 1), ards8h_mimic.drop(['PEEP','PaO2', 'FIO2 (%)'], axis = 1), how = 'left', on = ['subject_id', 'stay_id', 'Time_since_ICU_admission'])
    ards8h_eicu_test = pd.merge(eicu_test.drop(['Annotation', 'Case', 'ethnicity', 'Shock_next_8h', 'INDEX', 'PaO2/FiO2'], axis = 1), ards8h_eicu.drop(['PEEP','PaO2', 'FIO2 (%)'], axis = 1), how = 'left', on = ['uniquepid', 'patientunitstayid', 'Time_since_ICU_admission'])
    
    drop_stayid = []
    for stay_id, group in ards8h_mimic_train.groupby('stay_id'):
        if any(group['Annotation_ARDS'] == 'ARDS'):
            drop_stayid.append(stay_id)
            
    for stay_id, group in ards8h_mimic_valid.groupby('stay_id'):
        if any(group['Annotation_ARDS'] == 'ARDS'):
            drop_stayid.append(stay_id)
            
    for stay_id, group in ards8h_eicu_test.groupby('patientunitstayid'):
        if any(group['Annotation_ARDS'] == 'ARDS'):
            drop_stayid.append(stay_id)
            
    ards8h_mimic_train = ards8h_mimic_train[(ards8h_mimic_train['stay_id'].isin(drop_stayid))].reset_index(drop=True)
    ards8h_mimic_valid = ards8h_mimic_valid[(ards8h_mimic_valid['stay_id'].isin(drop_stayid))].reset_index(drop=True)
    ards8h_eicu_test = ards8h_eicu_test[(ards8h_eicu_test['patientunitstayid'].isin(drop_stayid))].reset_index(drop=True)
    
    
    ards8h_mimic_train = get_ARDS(ards8h_mimic_train, event='ARDS', mode = 'mimic')
    ards8h_mimic_valid = get_ARDS(ards8h_mimic_valid, event='ARDS', mode = 'mimic')
    ards8h_eicu_test = get_ARDS(ards8h_eicu_test, event='ARDS', mode = 'eicu')
    
    return ards8h_mimic_train, ards8h_mimic_valid, ards8h_eicu_test


def SIC4_prediction_DATA(mimic_train, mimic_valid, eicu_test):
    
    sics4h_mimic_path = '/Users/DAHS/Desktop/ECP_CONT/ECP_SCL/Experiment(Supervised Learning)/sub task prediction/sub_task_dataset/mimic_sic_4h.csv.gz'
    sics4h_eicu_path = '/Users/DAHS/Desktop/ECP_CONT/ECP_SCL/Experiment(Supervised Learning)/sub task prediction/sub_task_dataset/eicu_sic_4h.csv.gz'

    sics4h_mimic = pd.read_csv(sics4h_mimic_path, compression = 'gzip', index_col=0)
    sics4h_eicu = pd.read_csv(sics4h_eicu_path, compression = 'gzip', index_col=0)
    
    sics4h_mimic_train = pd.merge(mimic_train.drop(['Annotation', 'Case', 'ethnicity', 'Shock_next_8h', 'INDEX'], axis = 1), sics4h_mimic, how = 'inner', on = ['subject_id', 'stay_id', 'Time_since_ICU_admission'])
    sics4h_mimic_valid = pd.merge(mimic_valid.drop(['Annotation', 'Case', 'ethnicity', 'Shock_next_8h', 'INDEX'], axis = 1), sics4h_mimic, how = 'inner', on = ['subject_id', 'stay_id', 'Time_since_ICU_admission'])
    sics4h_eicu_test = pd.merge(eicu_test.drop(['Annotation', 'Case', 'ethnicity', 'Shock_next_8h', 'INDEX'], axis = 1), sics4h_eicu, how = 'inner', on = ['uniquepid', 'patientunitstayid', 'Time_since_ICU_admission'])
    
    drop_stayid = []
    for stay_id, group in sics4h_mimic_train.groupby('stay_id'):
        if any(group['Annotation_SIC'] == 'SIC'):
            drop_stayid.append(stay_id)
            
    for stay_id, group in sics4h_mimic_valid.groupby('stay_id'):
        if any(group['Annotation_SIC'] == 'SIC'):
            drop_stayid.append(stay_id)
            
    for stay_id, group in sics4h_eicu_test.groupby('patientunitstayid'):
        if any(group['Annotation_SIC'] == 'SIC'):
            drop_stayid.append(stay_id)
            
    sics4h_mimic_train = sics4h_mimic_train[(sics4h_mimic_train['stay_id'].isin(drop_stayid))].reset_index(drop=True)
    sics4h_mimic_valid = sics4h_mimic_valid[(sics4h_mimic_valid['stay_id'].isin(drop_stayid))].reset_index(drop=True)
    sics4h_eicu_test = sics4h_eicu_test[(sics4h_eicu_test['patientunitstayid'].isin(drop_stayid))].reset_index(drop=True)
    
    
    sics4h_mimic_train = get_SIC(sics4h_mimic_train, event='SIC', mode = 'mimic')
    sics4h_mimic_valid = get_SIC(sics4h_mimic_valid, event='SIC', mode = 'mimic')
    sics4h_eicu_test = get_SIC(sics4h_eicu_test, event='SIC', mode = 'eicu')
    
    
    return sics4h_mimic_train, sics4h_mimic_valid, sics4h_eicu_test


def SIC8_prediction_DATA(mimic_train, mimic_valid, eicu_test):
    
    sics8h_mimic_path = '/Users/DAHS/Desktop/ECP_CONT/ECP_SCL/Experiment(Supervised Learning)/sub task prediction/sub_task_dataset/mimic_sic_8h.csv.gz'
    sics8h_eicu_path = '/Users/DAHS/Desktop/ECP_CONT/ECP_SCL/Experiment(Supervised Learning)/sub task prediction/sub_task_dataset/eicu_sic_8h.csv.gz'

    sics8h_mimic = pd.read_csv(sics8h_mimic_path, compression = 'gzip', index_col=0)
    sics8h_eicu = pd.read_csv(sics8h_eicu_path, compression = 'gzip', index_col=0)
    
    sics8h_mimic_train = pd.merge(mimic_train.drop(['Annotation', 'Case', 'ethnicity', 'Shock_next_8h', 'INDEX'], axis = 1), sics8h_mimic, how = 'inner', on = ['subject_id', 'stay_id', 'Time_since_ICU_admission'])
    sics8h_mimic_valid = pd.merge(mimic_valid.drop(['Annotation', 'Case', 'ethnicity', 'Shock_next_8h', 'INDEX'], axis = 1), sics8h_mimic, how = 'inner', on = ['subject_id', 'stay_id', 'Time_since_ICU_admission'])
    sics8h_eicu_test = pd.merge(eicu_test.drop(['Annotation', 'Case', 'ethnicity', 'Shock_next_8h', 'INDEX'], axis = 1), sics8h_eicu, how = 'inner', on = ['uniquepid', 'patientunitstayid', 'Time_since_ICU_admission'])
    
    drop_stayid = []
    for stay_id, group in sics8h_mimic_train.groupby('stay_id'):
        if any(group['Annotation_SIC'] == 'SIC'):
            drop_stayid.append(stay_id)
            
    for stay_id, group in sics8h_mimic_valid.groupby('stay_id'):
        if any(group['Annotation_SIC'] == 'SIC'):
            drop_stayid.append(stay_id)
            
    for stay_id, group in sics8h_eicu_test.groupby('patientunitstayid'):
        if any(group['Annotation_SIC'] == 'SIC'):
            drop_stayid.append(stay_id)
            
    sics8h_mimic_train = sics8h_mimic_train[(sics8h_mimic_train['stay_id'].isin(drop_stayid))].reset_index(drop=True)
    sics8h_mimic_valid = sics8h_mimic_valid[(sics8h_mimic_valid['stay_id'].isin(drop_stayid))].reset_index(drop=True)
    sics8h_eicu_test = sics8h_eicu_test[(sics8h_eicu_test['patientunitstayid'].isin(drop_stayid))].reset_index(drop=True)
    
    
    sics8h_mimic_train = get_SIC(sics8h_mimic_train, event='SIC', mode = 'mimic')
    sics8h_mimic_valid = get_SIC(sics8h_mimic_valid, event='SIC', mode = 'mimic')
    sics8h_eicu_test = get_SIC(sics8h_eicu_test, event='SIC', mode = 'eicu')
    
    
    return sics8h_mimic_train, sics8h_mimic_valid, sics8h_eicu_test


def get_ARDS(target, event='ARDS', mode = 'mimic'):
    
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
                if row['Annotation_ARDS']==event:
                    split_data.append(pd.DataFrame(current_part))
                    event_occurred = True
                    current_part = []
            
        if current_part:
            split_data.append(pd.DataFrame(current_part))
            current_part = []

    return pd.concat(split_data).reset_index(drop=True)


def get_SIC(target, event='SIC', mode = 'mimic'):
    
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
                if row['Annotation_SIC']==event:
                    split_data.append(pd.DataFrame(current_part))
                    event_occurred = True
                    current_part = []
            
        if current_part:
            split_data.append(pd.DataFrame(current_part))
            current_part = []

    return pd.concat(split_data).reset_index(drop=True)

def number(df):
    print('number of subject :', len(df.drop_duplicates(subset=["subject_id"])))
    print('number of hadm :', len(df.drop_duplicates(subset=["hadm_id"])))
    print('number of stay :', len(df.drop_duplicates(subset=["stay_id"])))
    print(df.iloc[:,-1].value_counts(normalize=True))
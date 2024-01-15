import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
import warnings
from tqdm import tqdm
import random
import time
import gc

warnings.filterwarnings('ignore')

def check_class_ratio(dataset):
    class_ratio = round(np.mean(dataset.classes), 4)
    return class_ratio

def data_split(df, seed, train_ratio, Threshold, n_trial, down_sample, case1, case2):
    data = df.copy()
    
    
    #모든 관측치가 amb , 0, 0인 경우 제외
    sub_view = data[['stay_id', 'Time_since_ICU_admission', 'Annotation', 'Shock_next_12h', 'classes']]
    sub_view[(sub_view['Annotation']=='ambiguous')&(sub_view['Shock_next_12h']==0)&(sub_view['classes']==0)]


    filtered_df = sub_view[(sub_view['Annotation'] == 'ambiguous') & 
                        (sub_view['Shock_next_12h'] == 0) & 
                        (sub_view['classes'] == 0)]


    matching_stay_ids = []

    for stay_id in filtered_df['stay_id'].unique():
        if filtered_df[filtered_df['stay_id'] == stay_id].shape[0] == sub_view[sub_view['stay_id'] == stay_id].shape[0]:
            matching_stay_ids.append(stay_id)
    
    data = data[~(data['stay_id'].isin(matching_stay_ids))]
    
    # 조건을 벡터화하여 계산
    condition_met = (data['Annotation'] == 'ambiguous') & \
                    (data['Shock_next_12h'] == 0) & \
                    (data['classes'] == 0)

    # 각 stay_id에 대해 첫 조건 불만족 지점 찾기
    # Use groupby with 'stay_id' and then apply a lambda function to find the min index for each group
    first_not_met_index = data[~condition_met].groupby('stay_id').apply(lambda x: x.index.min())
    
    sub_view = data[['stay_id', 'Time_since_ICU_admission', 'Annotation', 'Shock_next_12h', 'classes']]
    
    # Assuming data and first_not_met_index are already defined

    # Initialize a flag column with all True (meaning keep all rows initially)
    sub_view['keep_row'] = True

    # Iterate over first_not_met_index to update the flag column
    for stay_id, index in first_not_met_index.items():
        condition = (sub_view['stay_id'] == stay_id) & (sub_view.index >= index) & condition_met
        sub_view.loc[condition, 'keep_row'] = False

    # Filter the sub_viewFrame based on the flag column
    sub_view = sub_view[sub_view['keep_row']]

    # Drop the flag column if it's no longer needed
    sub_view.drop(columns=['keep_row'], inplace=True)
    
    data = data[data.index.isin(sub_view.index)].reset_index(drop=True)
    
    sub_view = data[['stay_id', 'Time_since_ICU_admission', 'MAP', 'Lactate', 'vasoactive/inotropic','Annotation', 'Shock_next_12h', 'classes']]
    
    filtered_stay_ids = sub_view.groupby('stay_id').filter(lambda x: (x['Annotation'] == 'no_circ').all())

    unique_stay_ids = filtered_stay_ids['stay_id'].unique()
    
    filtered_stay_ids_circ = sub_view.groupby('stay_id').filter(lambda x: ~x['Annotation'].str.contains(r'\bcirc\b', regex=True).any())
    
    unique_stay_ids_circ = filtered_stay_ids_circ['stay_id'].unique()
    print('한번이라도 circulatory failure event가 발생하지 않은 stay 제거',len(unique_stay_ids.tolist()+unique_stay_ids_circ.tolist()))

    data = data[~(data['stay_id'].isin(unique_stay_ids.tolist()+unique_stay_ids_circ.tolist()))].reset_index(drop=True)
    
    gc.collect()
    
    search_time = time.time()
     
    for T in range(n_trial):
        array = data.subject_id.unique()
        
        # seed = np.random.randint(0, 10000, 1)
        # seed = 6379
        np.random.seed(seed) 
        np.random.shuffle(array)


        split_point = int(train_ratio * len(array))
        stay_for_train, stay_for_test = np.split(array, [split_point])

        
        condition_train = data.subject_id.isin(stay_for_train)
        holdout_train = data[condition_train]

        condition_test = data.subject_id.isin(stay_for_test)
        holdout_test = data[condition_test]

        train_class_ratio  = check_class_ratio(holdout_train)
        test_class_ratio  = check_class_ratio(holdout_test)
                
        if abs(train_class_ratio - test_class_ratio) <= Threshold:
            
            break  # 클래스 비율의 차이가 threshold 이하일 경우 반복문 종료
        
        if T % 100 == 0:
            print('Trial: ', T)
            
        if T % 10000 == 0:
        
            Threshold = Threshold + 0.05
            print('Threshold 조정 + 0.05, 현재 한계값: {}'.format(Threshold))
        
        if T == 9999:
            print('최대 Trial 달성, 분할 불가')
    
    train = holdout_train.copy()
    test = holdout_test.copy()
    search_time_end = time.time()
    
    trn_class1 = train.classes.value_counts()[0]
    trn_class2 = train.classes.value_counts()[1]
    trn_class3 = train.classes.value_counts()[2]
    trn_class4 = train.classes.value_counts()[3]
    
    tes_class1 = test.classes.value_counts()[0]
    tes_class2 = test.classes.value_counts()[1]
    tes_class3 = test.classes.value_counts()[2]
    tes_class4 = test.classes.value_counts()[3]
    
    
    # train_no_circ = train[(train['classes'] == 0)|(train['classes'] == 1)]
    # train_circ = train[(train['classes'] == 2)|(train['classes'] == 3)]
    
    # test_no_circ = test[(test['classes'] == 0)|(test['classes'] == 1)]
    # test_circ = test[(test['classes'] == 2)|(test['classes'] == 3)]
                         
    print('test set : test set = {} : {}'.format(train_ratio, 1-train_ratio))
    print('Train set class: ', train.classes.value_counts().sort_index())
    print('Test set class: ', test.classes.value_counts().sort_index())
    print('-'*20)
    print('Train class ratio: {}:{}:{}:{}'.format((trn_class1)/(trn_class1+trn_class2+trn_class3+trn_class4), (trn_class2)/(trn_class1+trn_class2+trn_class3+trn_class4), (trn_class3)/(trn_class1+trn_class2+trn_class3+trn_class4), (trn_class4)/(trn_class1+trn_class2+trn_class3+trn_class4)))
    print('Test class ratio: {}:{}:{}:{}'.format((tes_class1)/(tes_class1+tes_class2+tes_class3+tes_class4), (tes_class2)/(tes_class1+tes_class2+tes_class3+tes_class4), (tes_class3)/(tes_class1+tes_class2+tes_class3+tes_class4), (tes_class4)/(tes_class1+tes_class2+tes_class3+tes_class4)))
    print('-'*20)
    print('Number of trainset patient:', len(train.subject_id.unique()))
    print('Number of testset patient:', len(test.subject_id.unique()))
    print('Number of trainset stay:', len(train.stay_id.unique()))
    print('Number of testset stay:', len(test.stay_id.unique()))
    print('-'*20)
    print('Split seed: ',seed)
    print('train ratio:', train_ratio)
    print('Threshold:', Threshold)
    print('-'*20)
    print('총 소요 시간(초):{}'.format(search_time_end - search_time))
    print('시도한 trial 수: ', T)
    

    return train.reset_index(drop=True), test.reset_index(drop=True)

class TableDataset(Dataset):
    def __init__(self,data_path,data_type,mode,seed, sampling, case1, case2):
        self.data_path = data_path
        self.data_type = data_type # eicu or mimic
        self.mode = mode # train / valid / test
        self.target = 'classes'
        self.seed = seed
        self.sampling = sampling
        self.case1 = case1
        self.case2 = case2
        self.df_num, self.df_cat, self.y = self.__prepare_data__()

    def __prepare_data__(self):
        df_raw = pd.read_csv(self.data_path, compression='gzip')
        
        df_raw.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_raw.fillna(0, inplace=True)
        
        self.num_features = ['HR', 'Temperature', 'MAP', 'ABPs', 'ABPd', 'Respiratory Rate', 'O2 Sat (%)', 'SVO2', 'SpO2',
                             'PaO2','FIO2 (%)', 'EtCO2', 'Cardiac Output', 'GCS_score', 'Lactate', 'Lactate_clearance_1h', 'Lactate_clearance_3h', 'Lactate_clearance_5h', 'Lactate_clearance_7h', 'Lactate_clearance_9h', 'Lactate_clearance_11h',
                             'BUN','Total Bilirubin', 'ALT', 'Troponin-T', 'Creatinine','RedBloodCell', 'pH', 'Hemoglobin', 'Hematocrit','classes', 'stay_id', 'hadm_id', 'Annotation', 'ethnicity']
        self.cat_features = ['vasoactive/inotropic', 'Mechanical_circ_support', 'Shock_next_12h']    
        
        # for col in df_raw.columns:
        #     if df_raw[col].nunique() == 2 and all(df_raw[col].unique() == [0, 1]):
        #         self.cat_features.append(col)
        #     else:
        #         self.num_features.append(col)
                
        scaler = MinMaxScaler()
        df_train, df_valid = data_split(df_raw, self.seed, 0.7, 0.05, 1, self.sampling, self.case1, self.case2)
        
        # if dataset is eicu
        if self.data_type == 'mimic':
            if self.mode == "train":
                X_num = df_train[self.num_features].drop(['classes', 'stay_id', 'hadm_id', 'Annotation', 'ethnicity'], axis = 1)
                X_num_scaled = scaler.fit_transform(X_num)
                
                X_num = pd.DataFrame(X_num_scaled,columns = X_num.columns)
                X_cat = df_train[self.cat_features].drop(['Shock_next_12h'], axis = 1)
                y = df_train[self.target]
                return X_num, X_cat, y
            
            else:
                X_num_standard = df_train[self.num_features].drop(['classes', 'stay_id', 'hadm_id', 'Annotation', 'ethnicity'], axis = 1)
                scaler.fit(X_num_standard)
                
                X_num = df_valid[self.num_features].drop(['classes', 'stay_id', 'hadm_id', 'Annotation', 'ethnicity'], axis = 1)
                X_num_scaled = scaler.transform(X_num)
                
                X_num = pd.DataFrame(X_num_scaled,columns = X_num.columns)
                X_cat = df_valid[self.cat_features].drop(['Shock_next_12h'], axis = 1)
                y = df_valid[self.target]
                return X_num, X_cat, y

        # if dataset is eicu
        else:
            ## scaler fitting을 위한 과정
            df_scaling = pd.read_csv("/Users/DAHS/Desktop/ECP_CONT/ECP_SCL/Case Labeling/eICU.csv.gz", compression='gzip')
            df_train, _ = data_split(df_scaling,self.seed, 0.7, 0.05, 1, self.sampling, self.case1, self.case2)
            X_num_standard = df_train[self.num_features].drop(['classes', 'patientunitstayid', 'uniquepid', 'Annotation', 'ethnicity'], axis = 1)
            scaler.fit(X_num_standard)

            X_num = df_raw[self.num_features].drop(['classes', 'patientunitstayid', 'uniquepid', 'Annotation', 'ethnicity'], axis = 1)
            X_num_scaled = scaler.transform(X_num)
            X_num = pd.DataFrame(X_num_scaled,columns = X_num.columns)
            X_cat = df_raw[self.cat_features].drop(['Shock_next_12h'], axis = 1)
            y = df_raw[self.target]
            return X_num, X_cat, y

    def __getitem__(self,index):
      
        X_num_features = torch.tensor(self.df_num.iloc[index,:].values,dtype=torch.float32)
        X_cat_features = torch.tensor(self.df_cat.iloc[index,:].values,dtype=torch.float32).long()
        label = torch.tensor(int(self.y.iloc[index]),dtype=torch.float32)

        return X_num_features, X_cat_features, label
    
    def __len__(self):
        return self.y.shape[0]
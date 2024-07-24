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
from sklearn.utils import resample
import time
import gc
import sys

import eicu_year_process
module_path='/Users/DAHS/Desktop/ECP_CONT/ECP_SCL/Experiment/evaluation/'
if module_path not in sys.path:
    sys.path.append(module_path)

import get_hospital_eicu
warnings.filterwarnings('ignore')

def check_class_ratio(dataset):
    class_ratio = round(np.mean(dataset.Case), 4)
    return class_ratio

def data_split(df, seed, train_ratio, Threshold, n_trial, mode):
    
    if mode == 'eicu':
        patient_id = 'uniquepid'
        stay_id = 'patientunitstayid'
        seed = seed
        
        class1 = df.Case.value_counts()[1]
        class2 = df.Case.value_counts()[2]
        class3 = df.Case.value_counts()[3]
        class4 = df.Case.value_counts()[4]
        
        
        print("========== 데이터셋 분할 정보 ==========")
        print("학습셋 클래스 비율: {}".format(df.Case.value_counts().sort_index()))
        print("--------------------------------------")

        print("========== 클래스 비율 ==========")
        print("학습셋 클래스 비율: {:.2f}:{:.2f}:{:.2f}:{:.2f}".format(
            class1/(class1+class2+class3+class4),
            class2/(class1+class2+class3+class4),
            class3/(class1+class2+class3+class4),
            class4/(class1+class2+class3+class4)))
        print("--------------------------------------")

        print("========== 환자 및 체류 정보 ==========")
        print("학습셋 환자 수: {}".format(len(df[patient_id].unique())))
        print("학습셋 체류 수: {}".format(len(df[stay_id].unique())))
        print("--------------------------------------")
        
        return df
        
    else: 
        patient_id = 'subject_id'
        stay_id = 'stay_id'
        seed = seed

        
        data = df.copy()
        gc.collect()
        
        search_time = time.time()
        
        for T in range(n_trial):
            array = data[stay_id].unique()
            
            # seed = np.random.randint(0, 10000, 1)
            seed = 9040
            np.random.seed(seed) 
            np.random.shuffle(array)


            split_point = int(train_ratio * len(array))
            stay_for_train, stay_for_test = np.split(array, [split_point])

            
            condition_train = data[stay_id].isin(stay_for_train)
            holdout_train = data[condition_train]

            condition_test = data[stay_id].isin(stay_for_test)
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
        
        trn_class1 = train.Case.value_counts()[1]
        trn_class2 = train.Case.value_counts()[2]
        trn_class3 = train.Case.value_counts()[3]
        trn_class4 = train.Case.value_counts()[4]
        
        tes_class1 = test.Case.value_counts()[1]
        tes_class2 = test.Case.value_counts()[2]
        tes_class3 = test.Case.value_counts()[3]
        tes_class4 = test.Case.value_counts()[4]
        
        
        print("========== 데이터셋 분할 정보 ==========")
        print("데이터셋 비율: 학습 = {:.2f}, 테스트 = {:.2f}".format(train_ratio, 1-train_ratio))
        print("학습셋 클래스 비율: {}".format(train.Case.value_counts().sort_index()))
        print("테스트셋 클래스 비율: {}".format(test.Case.value_counts().sort_index()))
        print("--------------------------------------")

        print("========== 클래스 비율 ==========")
        print("학습셋 클래스 비율: {:.2f}:{:.2f}:{:.2f}:{:.2f}".format(
            trn_class1/(trn_class1+trn_class2+trn_class3+trn_class4),
            trn_class2/(trn_class1+trn_class2+trn_class3+trn_class4),
            trn_class3/(trn_class1+trn_class2+trn_class3+trn_class4),
            trn_class4/(trn_class1+trn_class2+trn_class3+trn_class4)))
        print("테스트셋 클래스 비율: {:.2f}:{:.2f}:{:.2f}:{:.2f}".format(
            tes_class1/(tes_class1+tes_class2+tes_class3+tes_class4),
            tes_class2/(tes_class1+tes_class2+tes_class3+tes_class4),
            tes_class3/(tes_class1+tes_class2+tes_class3+tes_class4),
            tes_class4/(tes_class1+tes_class2+tes_class3+tes_class4)))
        print("--------------------------------------")

        print("========== 환자 및 체류 정보 ==========")
        print("학습셋 환자 수: {}".format(len(train[patient_id].unique())))
        print("테스트셋 환자 수: {}".format(len(test[patient_id].unique())))
        print("학습셋 체류 수: {}".format(len(train[stay_id].unique())))
        print("테스트셋 체류 수: {}".format(len(test[stay_id].unique())))
        print("--------------------------------------")

        print("========== 실험 설정 ==========")
        print("분할 시드: {}".format(seed))
        print("학습 비율: {}".format(train_ratio))
        print("임계값: {}".format(Threshold))
        print("--------------------------------------")

        print("========== 실행 결과 ==========")
        print("총 소요 시간(초): {:.2f}".format(search_time_end - search_time))
        print("시도한 시행 횟수: {}".format(T))

        return train.reset_index(drop=True), test.reset_index(drop=True)

class TrainingDataset(Dataset):
    def __init__(self,data_path,data_type,mode,seed):
        self.data_path = data_path
        self.data_type = data_type # eicu or mimic
        self.mode = mode # train / valid / test
        self.target = 'Case'
        self.seed = seed
        self.df_num, self.df_cat, self.y = self.__prepare_data__()

    def __prepare_data__(self):
        df_raw = pd.read_csv(self.data_path, compression='gzip')
        df_raw = df_raw.drop(['Shock_next_12h', 'after_shock_annotation', 'Unnamed: 0'], axis = 1)
        df_raw = df_raw[~(df_raw['gender']==2)].reset_index(drop=True)

        df_raw.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_raw.fillna(0, inplace=True) 
        
        df_raw = df_raw[~(df_raw['Case']=='event')]
        df_raw = df_raw[~((df_raw['INDEX']=='CASE3_CASE4_DF')&(df_raw['Annotation']=='no_circ'))]
        df_raw['Case'] = pd.to_numeric(df_raw['Case'], errors='coerce')
        # df_raw = df_raw[~(df_raw['Annotation']=='ambiguous')]
        
        if self.data_type == 'mimic':
            stay = 'stay_id'
        elif self.data_type == 'eicu':
            stay = 'patientunitstayid'
        
        scaler = MinMaxScaler()
        
        # if dataset is eicu
        if self.data_type == 'mimic':
            
            self.cat_features = []
            self.num_features = []
        
            for col in df_raw.columns:
                if df_raw[col].nunique() == 2:
                    self.cat_features.append(col)
                else:
                    self.num_features.append(col)
                    
            df_train, df_valid = data_split(df_raw, self.seed, 0.9, 0.05, 1, mode = self.data_type)
            
            if self.mode == "train":
                X_num = df_train[self.num_features].drop(['Case', 'stay_id', 'subject_id','hadm_id','Annotation'], axis = 1)
                X_num_scaled = scaler.fit_transform(X_num)
                
                X_num = pd.DataFrame(X_num_scaled,columns = X_num.columns)
                X_cat = df_train[self.cat_features].drop(['Shock_next_8h','INDEX'], axis = 1)
                y = df_train[self.target]
                return X_num, X_cat, y
            
            else:
                X_num_standard = df_train[self.num_features].drop(['Case', 'stay_id', 'subject_id','hadm_id', 'Annotation'], axis = 1)
                scaler.fit(X_num_standard)
                
                X_num = df_valid[self.num_features].drop(['Case', 'stay_id', 'subject_id','hadm_id', 'Annotation'], axis = 1)
                X_num_scaled = scaler.transform(X_num)
                
                X_num = pd.DataFrame(X_num_scaled,columns = X_num.columns)
                X_cat = df_valid[self.cat_features].drop(['Shock_next_8h','INDEX'], axis = 1)
                y = df_valid[self.target]
                return X_num, X_cat, y

        # if dataset is eicu
        else:
            
            mimic = pd.read_csv('/Users/DAHS/Desktop/ECP_CONT/ECP_SCL/Case Labeling/mimic_analysis.csv.gz', compression='gzip')
            
            self.cat_features = []
            self.num_features = []
        
            for col in mimic.columns:
                if mimic[col].nunique() == 2:
                    self.cat_features.append(col)
                else:
                    self.num_features.append(col)
                    
                
            X_num_standard = mimic[self.num_features].drop(['Case', 'stay_id', 'subject_id','hadm_id','Annotation'], axis = 1)
            scaler.fit(X_num_standard)
            
            to_remove = ['stay_id', 'subject_id', 'hadm_id', 'Case', 'Annotation']

            for item in to_remove:
                if item in self.num_features:
                    self.num_features.remove(item)
            
            X_num_standard = df_raw[self.num_features]
            X_num_scaled = scaler.transform(X_num_standard)
            X_num = pd.DataFrame(X_num_scaled,columns = X_num_standard.columns)
            X_cat = df_raw[self.cat_features].drop(['Shock_next_8h','INDEX'], axis = 1)
            y = df_raw[self.target]
            return X_num, X_cat, y

    def __getitem__(self,index):
      
        X_num_features = torch.tensor(self.df_num.iloc[index,:].values,dtype=torch.float32)
        X_cat_features = torch.tensor(self.df_cat.iloc[index,:].values,dtype=torch.float32).long()
        label = torch.tensor(int(self.y.iloc[index]),dtype=torch.float32)

        return X_num_features, X_cat_features, label
    
    def __len__(self):
        return self.y.shape[0]
    

class MLPDataset(Dataset):
    def __init__(self,data_path,data_type,mode,seed):
        self.data_path = data_path
        self.data_type = data_type # eicu or mimic
        self.mode = mode # train / valid / test
        self.target = 'Case'
        self.seed = seed
        self.X, self.y = self.__prepare_data__()

    def __prepare_data__(self):
        df_raw = pd.read_csv(self.data_path, compression='gzip')
        df_raw = df_raw.drop(['Shock_next_12h', 'after_shock_annotation', 'Unnamed: 0'], axis = 1)
        df_raw = df_raw[~(df_raw['gender']==2)].reset_index(drop=True)

        df_raw.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_raw.fillna(0, inplace=True) 
        
        df_raw = df_raw[~(df_raw['Case']=='event')]
        df_raw = df_raw[~((df_raw['INDEX']=='CASE3_CASE4_DF')&(df_raw['Annotation']=='no_circ'))]
        df_raw['Case'] = pd.to_numeric(df_raw['Case'], errors='coerce')
        # df_raw = df_raw[~(df_raw['Annotation']=='ambiguous')]

        
        if self.data_type == 'mimic':
            stay = 'stay_id'
        elif self.data_type == 'eicu':
            stay = 'patientunitstayid'
            df_raw = eicu_year_process.matching_patient(df_raw)
            _, df_raw = get_hospital_eicu.hospital(df_raw)
        
        scaler = MinMaxScaler()
        
        # if dataset is eicu
        if self.data_type == 'mimic':
            
            self.cat_features = []
            self.num_features = []
        
            for col in df_raw.columns:
                if df_raw[col].nunique() == 2:
                    self.cat_features.append(col)
                else:
                    self.num_features.append(col)
                    
            df_train, df_valid = data_split(df_raw, self.seed, 0.9, 0.05, 1, mode = self.data_type)
            
            if self.mode == "train":
                X_num = df_train[self.num_features].drop(['Case', 'stay_id', 'subject_id','hadm_id','Annotation'], axis = 1)
                X_num_scaled = scaler.fit_transform(X_num)
                
                X_num = pd.DataFrame(X_num_scaled,columns = X_num.columns)
                X_cat = df_train[self.cat_features].drop(['Shock_next_8h','INDEX'], axis = 1)
                X = pd.concat([X_num, X_cat], axis = 1)
                y = df_train[self.target]
                return X, y
            
            else:
                X_num_standard = df_train[self.num_features].drop(['Case', 'stay_id', 'subject_id','hadm_id', 'Annotation'], axis = 1)
                scaler.fit(X_num_standard)
                
                X_num = df_valid[self.num_features].drop(['Case', 'stay_id', 'subject_id','hadm_id', 'Annotation'], axis = 1)
                X_num_scaled = scaler.transform(X_num)
                
                X_num = pd.DataFrame(X_num_scaled,columns = X_num.columns)
                X_cat = df_valid[self.cat_features].drop(['Shock_next_8h','INDEX'], axis = 1)
                X = pd.concat([X_num, X_cat], axis = 1)
                y = df_valid[self.target]
                return X, y
               

        # if dataset is eicu
        else:
            
            mimic = pd.read_csv('/Users/DAHS/Desktop/ECP_CONT/ECP_SCL/Case Labeling/mimic_analysis.csv.gz', compression='gzip')
            mimic = mimic.drop(['Shock_next_12h', 'after_shock_annotation', 'Unnamed: 0'], axis = 1)
            self.cat_features = []
            self.num_features = []
        
            for col in mimic.columns:
                if mimic[col].nunique() == 2:
                    self.cat_features.append(col)
                else:
                    self.num_features.append(col)
                    
                
            X_num_standard = mimic[self.num_features].drop(['Case', 'stay_id', 'subject_id','hadm_id','Annotation'], axis = 1)
            scaler.fit(X_num_standard)
            
            to_remove = ['stay_id', 'subject_id', 'hadm_id', 'Case', 'Annotation']

            for item in to_remove:
                if item in self.num_features:
                    self.num_features.remove(item)
            
            X_num_standard = df_raw[self.num_features]
            X_num_scaled = scaler.transform(X_num_standard)
            X_num = pd.DataFrame(X_num_scaled,columns = X_num_standard.columns)
            X_cat = df_raw[self.cat_features].drop(['Shock_next_8h','INDEX'], axis = 1)
            X = pd.concat([X_num, X_cat], axis = 1)
            y = df_raw[self.target]
            return X, y

    def __getitem__(self,index):
      
        X_features = torch.tensor(self.X.iloc[index,:].values,dtype=torch.float32)
        label = torch.tensor(int(self.y.iloc[index]),dtype=torch.float32)

        return X_features, label
    
    def __len__(self):
        return self.y.shape[0]
    
    
class TableDataset(Dataset):
    def __init__(self,data_path,data_type,mode,seed):
        self.data_path = data_path
        self.data_type = data_type # eicu or mimic
        self.mode = mode # train / valid / test / fine
        self.target = 'Case'
        self.seed = seed
        self.df_num, self.df_cat, self.y = self.__prepare_data__()

    def __prepare_data__(self):
        df_raw = pd.read_csv(self.data_path, compression='gzip')
        df_raw = df_raw[~(df_raw['gender']==2)].reset_index(drop=True)
        df_raw = df_raw.drop(['Shock_next_12h', 'after_shock_annotation', 'Unnamed: 0'], axis = 1)

        df_raw.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_raw.fillna(0, inplace=True) 
        
        df_raw = df_raw[~(df_raw['Case']=='event')]
        df_raw = df_raw[~((df_raw['INDEX']=='CASE3_CASE4_DF')&(df_raw['Annotation']=='no_circ'))]
        df_raw['Case'] = pd.to_numeric(df_raw['Case'], errors='coerce')
        # df_raw = df_raw[~(df_raw['Annotation']=='ambiguous')]
        
        if self.data_type == 'mimic':
            stay = 'stay_id'
        elif self.data_type == 'eicu':
            stay = 'patientunitstayid'
        
        scaler = MinMaxScaler()
        
        if self.data_type == 'mimic':
            
            self.cat_features = []
            self.num_features = []
        
            for col in df_raw.columns:
                if df_raw[col].nunique() == 2:
                    self.cat_features.append(col)
                else:
                    self.num_features.append(col)
            if self.mode != 'noevent':         
                df_train, df_valid = data_split(df_raw, self.seed, 0.9, 0.05, 1, mode = self.data_type)
            if self.mode == "train":
                X_num = df_train[self.num_features].drop(['Case', 'stay_id', 'subject_id','hadm_id','Annotation'], axis = 1)
                X_num_scaled = scaler.fit_transform(X_num)
                
                X_num = pd.DataFrame(X_num_scaled,columns = X_num.columns)
                X_cat = df_train[self.cat_features].drop(['Shock_next_8h','INDEX'], axis = 1)
                y = df_train[self.target]
                return X_num, X_cat, y
            
            elif self.mode == "valid":
                X_num_standard = df_train[self.num_features].drop(['Case', 'stay_id', 'subject_id','hadm_id', 'Annotation'], axis = 1)
                scaler.fit(X_num_standard)
                
                X_num = df_valid[self.num_features].drop(['Case', 'stay_id', 'subject_id','hadm_id', 'Annotation'], axis = 1)
                X_num_scaled = scaler.transform(X_num)
                
                X_num = pd.DataFrame(X_num_scaled,columns = X_num.columns)
                X_cat = df_valid[self.cat_features].drop(['Shock_next_8h','INDEX'], axis = 1)
                y = df_valid[self.target]
                return X_num, X_cat, y
            
            else:
                mimic = pd.read_csv('/Users/DAHS/Desktop/ECP_CONT/ECP_SCL/Case Labeling/mimic_analysis.csv.gz', compression='gzip')
                mimic = mimic.drop(['Shock_next_12h', 'after_shock_annotation', 'Unnamed: 0'], axis = 1)
                mimic = mimic[~(mimic['gender']==2)].reset_index(drop=True)
                mimic = mimic[~(mimic['Case']=='event')]
                mimic = mimic[~((mimic['INDEX']=='CASE3_CASE4_DF')&(mimic['Annotation']=='no_circ'))]
                mimic['Case'] = pd.to_numeric(mimic['Case'], errors='coerce')
                
                self.cat_features = []
                self.num_features = []
            
                for col in mimic.columns:
                    if mimic[col].nunique() == 2:
                        self.cat_features.append(col)
                    else:
                        self.num_features.append(col)
                        
                X_num_standard = mimic[self.num_features].drop(['Case', 'stay_id', 'subject_id','hadm_id', 'Annotation'], axis = 1)
                scaler.fit(X_num_standard)
                
                to_remove = ['stay_id', 'subject_id', 'hadm_id', 'Case', 'Annotation']

                for item in to_remove:
                    if item in self.num_features:
                        self.num_features.remove(item)
                
                X_num_standard = df_raw[self.num_features]
                X_num_scaled = scaler.transform(X_num_standard)
                X_num = pd.DataFrame(X_num_scaled,columns = X_num_standard.columns)
                X_cat = df_raw[self.cat_features].drop(['Shock_next_8h','INDEX'], axis = 1)
                y = df_raw[self.target]
                return X_num, X_cat, y

        # if dataset is eicu
        else:
            
            # df_raw = eicu_year_process.matching_patient(df_raw)
            
            if self.mode == 'all':
                # df_raw = df_raw[df_raw['hospitaldischargeyear']==2014]
                
                mimic = pd.read_csv('/Users/DAHS/Desktop/ECP_CONT/ECP_SCL/Case Labeling/mimic_analysis.csv.gz', compression='gzip')
                mimic = mimic.drop(['Shock_next_12h', 'after_shock_annotation', 'Unnamed: 0'], axis = 1)
                mimic = mimic[~(mimic['gender']==2)].reset_index(drop=True)
                mimic = mimic[~(mimic['Case']=='event')]
                mimic = mimic[~((mimic['INDEX']=='CASE3_CASE4_DF')&(mimic['Annotation']=='no_circ'))]
                mimic['Case'] = pd.to_numeric(mimic['Case'], errors='coerce')
                
                df_train, _ = data_split(mimic, self.seed, 0.9, 0.05, 1, mode = 'mimic')
                
                self.cat_features = []
                self.num_features = []
            
                for col in df_train.columns:
                    if df_train[col].nunique() == 2:
                        self.cat_features.append(col)
                    else:
                        self.num_features.append(col)
                        
                X_num_standard = df_train[self.num_features].drop(['Case', 'stay_id', 'subject_id','hadm_id', 'Annotation'], axis = 1)
                scaler.fit(X_num_standard)
                
                to_remove = ['stay_id', 'subject_id', 'hadm_id', 'Case', 'Annotation']

                for item in to_remove:
                    if item in self.num_features:
                        self.num_features.remove(item)
                
                X_num_standard = df_raw[self.num_features]
                X_num_scaled = scaler.transform(X_num_standard)
                X_num = pd.DataFrame(X_num_scaled,columns = X_num_standard.columns)
                X_cat = df_raw[self.cat_features].drop(['Shock_next_8h','INDEX'], axis = 1)
                y = df_raw[self.target]
                return X_num, X_cat, y
            
            else:

                df_raw = df_raw[df_raw['hospitaldischargeyear']==2015]
                
                mimic = pd.read_csv('/Users/DAHS/Desktop/ECP_CONT/ECP_SCL/Case Labeling/mimic_analysis.csv.gz', compression='gzip')
                mimic = mimic.drop('Shock_next_12h', axis = 1)
                self.cat_features = []
                self.num_features = []
            
                for col in mimic.columns:
                    if mimic[col].nunique() == 2:
                        self.cat_features.append(col)
                    else:
                        self.num_features.append(col)
                        
                        
                X_num_standard = mimic[self.num_features].drop(['Case', 'stay_id', 'subject_id','hadm_id', 'Annotation'], axis = 1)
                scaler.fit(X_num_standard)
                
                to_remove = ['stay_id', 'subject_id', 'hadm_id', 'Case', 'Annotation']

                for item in to_remove:
                    if item in self.num_features:
                        self.num_features.remove(item)
                
                X_num_standard = df_raw[self.num_features]
                X_num_scaled = scaler.transform(X_num_standard)
                X_num = pd.DataFrame(X_num_scaled,columns = X_num_standard.columns)
                X_cat = df_raw[self.cat_features].drop(['Shock_next_8h','INDEX'], axis = 1)
                y = df_raw[self.target]
                return X_num, X_cat, y

    def __getitem__(self,index):
      
        X_num_features = torch.tensor(self.df_num.iloc[index,:].values,dtype=torch.float32)
        X_cat_features = torch.tensor(self.df_cat.iloc[index,:].values,dtype=torch.float32).long()
        label = torch.tensor(int(self.y.iloc[index]),dtype=torch.float32)

        return X_num_features, X_cat_features, label
    
    def __len__(self):
        return self.y.shape[0]
    

class Positive_Case2(Dataset):
    def __init__(self,data_path,data_type,mode,seed):
        self.data_path = data_path
        self.data_type = data_type # eicu or mimic
        self.mode = mode # train / valid / test / fine
        self.target = 'Case'
        self.seed = seed
        self.df_num, self.df_cat, self.y = self.__prepare_data__()

    def __prepare_data__(self):
        df_raw = pd.read_csv(self.data_path, compression='gzip')
        df_raw = df_raw[~(df_raw['gender']==2)].reset_index(drop=True)
        df_raw = df_raw.drop(['Shock_next_12h', 'after_shock_annotation', 'Unnamed: 0'], axis = 1)

        df_raw.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_raw.fillna(0, inplace=True) 
        
        class_sample_sizes = {1: 10, 2: 1884, 3: 10, 4: 10}
        df_raw['Case'] = pd.to_numeric(df_raw['Case'], errors='coerce')

        sampled_dfs = []
        for class_id, sample_size in class_sample_sizes.items():
            sampled_df = df_raw[df_raw['Case'] == class_id].sample(n=sample_size, random_state=76)
            sampled_dfs.append(sampled_df)

        df_raw = pd.concat(sampled_dfs)
        # df_raw = df_raw[~(df_raw['Annotation']=='ambiguous')]
        
        if self.data_type == 'mimic':
            stay = 'stay_id'
        elif self.data_type == 'eicu':
            stay = 'patientunitstayid'
        
        scaler = MinMaxScaler()
        
        positive_sample = np.load('/Users/DAHS/Desktop/ECP_CONT/ECP_SCL/Cohort_selection/Train/result/positive_attention.npy')
        df_raw = df_raw[df_raw['stay_id'].isin(positive_sample)]
        
        if self.data_type == 'mimic':
            
            mimic = pd.read_csv('/Users/DAHS/Desktop/ECP_CONT/ECP_SCL/Case Labeling/mimic_analysis(new_version0313).csv.gz', compression='gzip')
            mimic = mimic.drop(['Shock_next_12h', 'after_shock_annotation', 'Unnamed: 0'], axis = 1)
            mimic = mimic[~(mimic['gender']==2)].reset_index(drop=True)
            mimic = mimic[~(mimic['Case']=='event')]
            mimic = mimic[~((mimic['INDEX']=='CASE3_CASE4_DF')&(mimic['Annotation']=='no_circ'))]
            mimic['Case'] = pd.to_numeric(mimic['Case'], errors='coerce')
            self.cat_features = []
            self.num_features = []
        
            for col in mimic.columns:
                if mimic[col].nunique() == 2:
                    self.cat_features.append(col)
                else:
                    self.num_features.append(col)
                    
            if self.mode == "train":
                X_num = df_raw[self.num_features].drop(['Case', 'stay_id', 'subject_id','hadm_id','Annotation'], axis = 1)
                X_num_scaled = scaler.fit_transform(X_num)
                
                X_num = pd.DataFrame(X_num_scaled,columns = X_num.columns)
                X_cat = df_raw[self.cat_features].drop(['Shock_next_8h','INDEX'], axis = 1)
                y = df_raw[self.target]
                return X_num, X_cat, y
            
            else:
                X_num_standard = df_raw[self.num_features].drop(['Case', 'stay_id', 'subject_id','hadm_id', 'Annotation'], axis = 1)
                scaler.fit(X_num_standard)
                
                X_num = df_raw[self.num_features].drop(['Case', 'stay_id', 'subject_id','hadm_id', 'Annotation'], axis = 1)
                X_num_scaled = scaler.transform(X_num)
                
                X_num = pd.DataFrame(X_num_scaled,columns = X_num.columns)
                X_cat = df_raw[self.cat_features].drop(['Shock_next_8h','INDEX'], axis = 1)
                y = df_raw[self.target]
                return X_num, X_cat, y


    def __getitem__(self,index):
      
        X_num_features = torch.tensor(self.df_num.iloc[index,:].values,dtype=torch.float32)
        X_cat_features = torch.tensor(self.df_cat.iloc[index,:].values,dtype=torch.float32).long()
        label = torch.tensor(int(self.y.iloc[index]),dtype=torch.float32)

        return X_num_features, X_cat_features, label
    
    def __len__(self):
        return self.y.shape[0]

class Positive_Case4(Dataset):
    def __init__(self,data_path,data_type,mode,seed):
        self.data_path = data_path
        self.data_type = data_type # eicu or mimic
        self.mode = mode # train / valid / test / fine
        self.target = 'Case'
        self.seed = seed
        self.df_num, self.df_cat, self.y = self.__prepare_data__()

    def __prepare_data__(self):
        df_raw = pd.read_csv(self.data_path, compression='gzip')
        df_raw = df_raw[~(df_raw['gender']==2)].reset_index(drop=True)
        df_raw = df_raw.drop(['Shock_next_12h', 'after_shock_annotation', 'Unnamed: 0'], axis = 1)

        df_raw.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_raw.fillna(0, inplace=True) 
        
        class_sample_sizes = {1: 10, 2: 10, 3: 10, 4: 7599}
        df_raw['Case'] = pd.to_numeric(df_raw['Case'], errors='coerce')

        sampled_dfs = []
        for class_id, sample_size in class_sample_sizes.items():
            sampled_df = df_raw[df_raw['Case'] == class_id].sample(n=sample_size, random_state=76)
            sampled_dfs.append(sampled_df)

        df_raw = pd.concat(sampled_dfs)
        # df_raw = df_raw[~(df_raw['Annotation']=='ambiguous')]
        
        if self.data_type == 'mimic':
            stay = 'stay_id'
        elif self.data_type == 'eicu':
            stay = 'patientunitstayid'
        
        scaler = MinMaxScaler()
        
        positive_sample = np.load('/Users/DAHS/Desktop/ECP_CONT/ECP_SCL/Cohort_selection/Train/result/positive_attention.npy')
        df_raw = df_raw[df_raw['stay_id'].isin(positive_sample)]
        
        if self.data_type == 'mimic':
            
            mimic = pd.read_csv('/Users/DAHS/Desktop/ECP_CONT/ECP_SCL/Case Labeling/mimic_analysis(new_version0313).csv.gz', compression='gzip')
            mimic = mimic.drop(['Shock_next_12h', 'after_shock_annotation', 'Unnamed: 0'], axis = 1)
            mimic = mimic[~(mimic['gender']==2)].reset_index(drop=True)
            mimic = mimic[~(mimic['Case']=='event')]
            mimic = mimic[~((mimic['INDEX']=='CASE3_CASE4_DF')&(mimic['Annotation']=='no_circ'))]
            mimic['Case'] = pd.to_numeric(mimic['Case'], errors='coerce')
            self.cat_features = []
            self.num_features = []
        
            for col in mimic.columns:
                if mimic[col].nunique() == 2:
                    self.cat_features.append(col)
                else:
                    self.num_features.append(col)
                    
            if self.mode == "train":
                X_num = df_raw[self.num_features].drop(['Case', 'stay_id', 'subject_id','hadm_id','Annotation'], axis = 1)
                X_num_scaled = scaler.fit_transform(X_num)
                
                X_num = pd.DataFrame(X_num_scaled,columns = X_num.columns)
                X_cat = df_raw[self.cat_features].drop(['Shock_next_8h','INDEX'], axis = 1)
                y = df_raw[self.target]
                return X_num, X_cat, y
            
            else:
                X_num_standard = df_raw[self.num_features].drop(['Case', 'stay_id', 'subject_id','hadm_id', 'Annotation'], axis = 1)
                scaler.fit(X_num_standard)
                
                X_num = df_raw[self.num_features].drop(['Case', 'stay_id', 'subject_id','hadm_id', 'Annotation'], axis = 1)
                X_num_scaled = scaler.transform(X_num)
                
                X_num = pd.DataFrame(X_num_scaled,columns = X_num.columns)
                X_cat = df_raw[self.cat_features].drop(['Shock_next_8h','INDEX'], axis = 1)
                y = df_raw[self.target]
                return X_num, X_cat, y


    def __getitem__(self,index):
      
        X_num_features = torch.tensor(self.df_num.iloc[index,:].values,dtype=torch.float32)
        X_cat_features = torch.tensor(self.df_cat.iloc[index,:].values,dtype=torch.float32).long()
        label = torch.tensor(int(self.y.iloc[index]),dtype=torch.float32)

        return X_num_features, X_cat_features, label
    
    def __len__(self):
        return self.y.shape[0]
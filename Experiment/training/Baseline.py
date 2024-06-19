import pandas as pd
import numpy as np
import time
import sys

from sklearn.metrics import recall_score

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.stats import sem, t

module_path='/Users/DAHS/Desktop/ECP_CONT/ECP_SCL/Cohort_selection/'
if module_path not in sys.path:
    sys.path.append(module_path)

from cohort_loader import *
import eicu_year_process

module_path='/Users/DAHS/Desktop/ECP_CONT/ECP_SCL/Experiment(Supervised Learning)/evaluation/'
if module_path not in sys.path:
    sys.path.append(module_path)

import get_hospital_eicu

import split
from imp import reload
reload(split)

import Multiclass_evaluation
reload(Multiclass_evaluation)


mimic_path = '/Users/DAHS/Desktop/ECP_CONT/ECP_SCL/Case Labeling/mimic_analysis(new_version0313).csv.gz'
eicu_path = '/Users/DAHS/Desktop/ECP_CONT/ECP_SCL/Case Labeling/eicu_analysis(new_version0313).csv.gz'

mimic = pd.read_csv(mimic_path, compression = 'gzip')
mimic = mimic.drop('Shock_next_12h', axis = 1)

eicu = pd.read_csv(eicu_path, compression = 'gzip')
eicu = eicu.drop('Shock_next_12h', axis = 1)

dataset_mimic = mimic[~(mimic['gender']==2)].reset_index(drop=True)

dataset_mimic.replace([np.inf, -np.inf], np.nan, inplace=True)
dataset_mimic.fillna(0, inplace=True) 
dataset_mimic = dataset_mimic[~(dataset_mimic['Case']=='event')]
dataset_mimic = dataset_mimic[~((dataset_mimic['INDEX']=='CASE3_CASE4_DF')&(dataset_mimic['Annotation']=='no_circ'))]

eventset = mimic[(mimic['Case']=='event')].reset_index(drop=True)
dataset_mimic['Case'] = pd.to_numeric(dataset_mimic['Case'], errors='coerce')

mimic_train, mimic_valid = data_split(dataset_mimic, 9040, 0.9, Threshold=0.05, n_trial=1, mode = 'mimic')

eicu_t = eicu_year_process.matching_patient(eicu)
dataset_eicu = eicu_t[~(eicu_t['gender']==2)].reset_index(drop=True)
dataset_eicu.replace([np.inf, -np.inf], np.nan, inplace=True)
dataset_eicu.fillna(0, inplace=True) 

eicu_test = dataset_eicu[~(dataset_eicu['Case']=='event')]
event_eicu = dataset_eicu[(dataset_eicu['Case']=='event')]
eicu_test = eicu_test[~((eicu_test['INDEX']=='CASE3_CASE4_DF')&(eicu_test['Annotation']=='no_circ'))]
eicu_test['Case'] = pd.to_numeric(eicu_test['Case'], errors='coerce')
hosp_id, eicu_test = get_hospital_eicu.hospital(eicu_test)

eicu_type = get_hospital_eicu.eicu_subgroup(eicu_test)
unitadmitsource, unittype, unitstaytype = get_hospital_eicu.make_eicu_dataset(eicu_type)

X_train, y_train, _ = split.split_X_Y(mimic_train, mode = 'mimic')
X_valid, y_valid, valid_output = split.split_X_Y(mimic_valid, mode = 'mimic')

models, result = Multiclass_evaluation.create_analysis(eventset, X_train, y_train, X_valid, valid_output, mode = 'base')
external_validation = Multiclass_evaluation.external_validation_total_models(models, eicu_test, event_eicu)


def Caculate_alarm_time(inference_output, event, mode):
    global stay, scaled_pw
    alarm_time = 0
    alpha_ea_term = 0 
    
    if mode == 'mimic':
        icu_stay = 'stay_id'
    else:
        icu_stay = 'patientunitstayid'

    usecol = [icu_stay,'Time_since_ICU_admission', 'Annotation', 'progress', 'Case', 'prediction_label']

    event_set = event[usecol[:-1]].copy()
    event_set['prediction_label'] = 'label'
    output_set = inference_output[inference_output['INDEX']=='CASE1_CASE2_DF'][usecol]
    output_set['prediction_label'] = output_set['prediction_label']

    event_set = event_set[event_set[icu_stay].isin(output_set[icu_stay].unique())]

    event_set_reset = event_set.reset_index(drop=True)
    output_set_reset = output_set.reset_index(drop=True)

    total_set = pd.concat([event_set_reset, output_set_reset], axis = 0, ignore_index=True).reset_index(drop=True).sort_values(by='Time_since_ICU_admission')
    n_stay = total_set[icu_stay].nunique()
    for stay in total_set[icu_stay].unique():
        
        testing = total_set[total_set[icu_stay]==stay][[icu_stay, 'Time_since_ICU_admission', 'Case', 'prediction_label']]
        
        testing['Case'] = testing['Case'].replace({1: 0, 2:1})
        testing['prediction_label'] = testing['prediction_label'].replace({1: 0, 2:1})
        start_time = testing['Time_since_ICU_admission'].iloc[0]
        event_time = testing['Time_since_ICU_admission'].iloc[-1]

        time_window = event_time - 8

        prediction_window = testing[(testing['Time_since_ICU_admission'] >= time_window) & (testing['Time_since_ICU_admission'] < event_time)]
        scaled_pw = prediction_window.reset_index(drop=True)

        try:

            alarm_time += scaled_pw[scaled_pw['prediction_label']==1].index[0]
          
        except:
            alarm_time += 8
    print(alarm_time/n_stay)
    return alarm_time/n_stay


#ACID
#valid_all

ACID_result_valid_all = pd.DataFrame()
model_names = ['xgb', 'lgbm', 'catboost', 'rf', 'dt', 'svm-ovr', 'lr', 'naivebayes', 'knn']

for idx, model in enumerate(models):
    output = valid_output.copy()
    valid_preds = model.predict(X_valid)
    output['prediction_label'] = valid_preds if idx != 0 else valid_preds + 1
    output['prediction_prob'] = model.predict_proba(X_valid)[:, 1]
    
    cosine, _ = Multiclass_evaluation.ACID(output, eventset, 'mimic')
    
    # Ensure cosine is in a list or array if it's a scalar
    cosine = [cosine] if not isinstance(cosine, (list, np.ndarray)) else cosine
    
    # Convert to DataFrame
    cosine_df = pd.DataFrame({'cosine': cosine}, index=[model_names[idx]])
    ACID_result_valid_all = pd.concat([ACID_result_valid_all, cosine_df])
    
#ACID    
#Test
X_test, y_test, test_output = split.split_X_Y(eicu_test, mode = 'eicu')
ACID_result_all = pd.DataFrame()
model_names = ['xgb', 'lgbm', 'catboost', 'rf', 'dt', 'svm-ovr', 'lr', 'naivebayes', 'knn']

sample_test = test_output[test_output['patientunitstayid'].isin(event_eicu.patientunitstayid.unique())]
hat = sample_test[sample_test['after_shock_annotation']=='before_experience_shock'].groupby('patientunitstayid').count()['Time_since_ICU_admission']

for idx, model in enumerate(models):
    output = test_output.copy()
    test_preds = model.predict(X_test)
    output['prediction_label'] = test_preds if idx != 0 else test_preds + 1
    output['prediction_prob'] = model.predict_proba(X_test)[:, 1]
    
    cosine, _ = Multiclass_evaluation.ACID(output, event_eicu, 'eicu')
    time = Caculate_alarm_time(output[output.patientunitstayid.isin(hat[hat > 8].index)], event_eicu, 'eicu')
    time = [time] if not isinstance(time, (list, np.ndarray)) else time
    # Ensure cosine is in a list or array if it's a scalar
    cosine = [cosine] if not isinstance(cosine, (list, np.ndarray)) else cosine
    
    # Convert to DataFrame
    cosine_df = pd.DataFrame({'cosine': cosine, 'time': time}, index=[model_names[idx]])
    ACID_result_all = pd.concat([ACID_result_all, cosine_df])
    
    
    
#NSA

def NSA(recover, no_recover, mode):
    global no_recover_sid, recover_sid, recover_set, no_recover_set, consist
    recover_df = recover.copy()
    no_recover_df = no_recover.copy()
    
    consist = []
    no_consist = []
    if mode == 'mimic':
        stay_id_id = 'stay_id'
    else:
        stay_id_id ='patientunitstayid'
        
    recover_set = []
    recover_sid = []

    for idx, stay in enumerate(recover_df[stay_id_id].unique()):
        interest = recover_df[recover_df[stay_id_id]==stay]
        
        if interest.Case.nunique()>=3:
        
            before_shock = interest[interest['after_shock_annotation']=='before_experience_shock']
            before_shock['Case'] = before_shock['Case'].replace({1:0, 2:1})
            before_shock['prediction_label'] = before_shock['prediction_label'].replace({1:0, 2:1})

            if before_shock[before_shock['Case']==1]['prediction_label'].sum() >= 1:
                before_shock_score = 1
            else:
                before_shock_score = 0

            #---------------------------------------------------------------------------------------------
            interest = recover_df[recover_df[stay_id_id]==stay]
            after_shock = interest.iloc[[-1]]
            after_shock['Case'] = after_shock['Case'].replace({3:1, 4:0})
            after_shock['prediction_label'] = after_shock['prediction_label'].replace({3:1, 4:0})

            after_shock_score = recall_score(after_shock['Case'], after_shock['prediction_label'])
            #---------------------------------------------------------------------------------------------

            nsa = (0.7*np.array(before_shock_score).mean() + 0.3*np.array(after_shock_score).mean())
            consist.append(after_shock_score)
            
            recover_set.append(nsa)
            recover_sid.append(stay)
        
    no_recover_set = []
    no_recover_sid = []
    
    for stay in no_recover_df[stay_id_id].unique():
        
        interest = no_recover_df[no_recover_df[stay_id_id]==stay]
        
        if interest.Case.nunique()>=3:
            before_shock = interest[interest['after_shock_annotation']=='before_experience_shock']
            before_shock['Case'] = before_shock['Case'].replace({1:0, 2:1})
            before_shock['prediction_label'] = before_shock['prediction_label'].replace({1:0, 2:1})

            if before_shock[before_shock['Case']==1]['prediction_label'].sum() >= 1:
                before_shock_score = 1
            else:
                before_shock_score = 0

            #---------------------------------------------------------------------------------------------
            interest = no_recover_df[no_recover_df[stay_id_id]==stay]
            after_shock = interest.iloc[[-1]]
            after_shock['Case'] = after_shock['Case'].replace({3:0, 4:1})
            after_shock['prediction_label'] = after_shock['prediction_label'].replace({3:0, 4:1})

            after_shock_score = recall_score(after_shock['Case'], after_shock['prediction_label'])
            #---------------------------------------------------------------------------------------------

            nsa = (0.7*np.array(before_shock_score).mean() + 0.3*np.array(after_shock_score).mean())
            no_recover_set.append(nsa)
            no_recover_sid.append(stay)
            no_consist.append(after_shock_score)
            
    nsa = (np.array(recover_set).mean() + np.array(no_recover_set).mean())/2
    nsa_latter_term = (np.array(consist).mean())
    
    return nsa, nsa_latter_term

def NSA_Curve(recover, no_recover, mode):
    
    recover_df = recover.copy()
    no_recover_df = no_recover.copy()
    
    thresholds = np.linspace(0, 1, 21)
    threshold_by_score = []
    
    if mode == 'mimic':
        stay_id_id = 'stay_id'
    else:
        stay_id_id ='patientunitstayid'
    
    for th in thresholds:    
        recover_set = []

        for idx, stay in enumerate(recover_df[stay_id_id].unique()):
            interest = recover_df[recover_df[stay_id_id]==stay]
            before_shock = interest[interest['after_shock_annotation']=='before_experience_shock']
            before_shock['Case'] = before_shock['Case'].replace({1:0, 2:1})
            before_shock['prediction_label'] = before_shock['prediction_label'].replace({1:0, 2:1, 3:0, 4:0})
            
            before_shock['prediction_label'] = np.where(before_shock['prediction_prob_case2'] >= th, before_shock['Case'], 0)

            if before_shock[before_shock['Case']==1]['prediction_label'].sum() >= 1:
                before_shock_score = 1
            else:
                before_shock_score = 0

            #---------------------------------------------------------------------------------------------
            interest = recover_df[recover_df[stay_id_id]==stay]
            after_shock = interest.iloc[[-1]]
            after_shock['Case'] = after_shock['Case'].replace({3:1, 4:0, 0:1, 1:0, 2:0})
            after_shock['prediction_label'] = after_shock['prediction_label'].replace({3:1, 4:0, 1:0, 2:0})

            before_shock['prediction_label'] = np.where(before_shock['prediction_prob_case3'] >= th, before_shock['Case'], 0)
            
            after_shock_score = recall_score(after_shock['Case'], after_shock['prediction_label'])
            #---------------------------------------------------------------------------------------------

            nsa = (0.7*np.array(before_shock_score).mean() + 0.3*np.array(after_shock_score).mean())
            recover_set.append(nsa)
            
        no_recover_set = []

        for stay in no_recover_df[stay_id_id].unique():
            
            interest = no_recover_df[no_recover_df[stay_id_id]==stay]
            before_shock = interest[interest['after_shock_annotation']=='before_experience_shock']
            before_shock['Case'] = before_shock['Case'].replace({1:0, 2:1})
            before_shock['prediction_label'] = before_shock['prediction_label'].replace({1:0, 2:1, 3:0, 4:0})

            before_shock['prediction_label'] = np.where(before_shock['prediction_prob_case2'] >= th, before_shock['Case'], 0)
            
            if before_shock[before_shock['Case']==1]['prediction_label'].sum() >= 1:
                before_shock_score = 1
            else:
                before_shock_score = 0

            #---------------------------------------------------------------------------------------------
            interest = no_recover_df[no_recover_df[stay_id_id]==stay]
            after_shock = interest.iloc[[-1]]
            after_shock['Case'] = after_shock['Case'].replace({3:0, 4:1, 0:1, 1:0, 2:0})
            after_shock['prediction_label'] = after_shock['prediction_label'].replace({3:0, 4:1, 1:0, 2:0})
            before_shock['prediction_label'] = np.where(before_shock['prediction_prob_case4'] >= th, before_shock['Case'], 0)
            after_shock_score = recall_score(after_shock['Case'], after_shock['prediction_label'])
            #---------------------------------------------------------------------------------------------

            # nsa = np.round((before_shock_score + after_shock_score + 0.00000001)/(2+0.00000001), 4)
            nsa = (0.7*np.array(before_shock_score).mean() + 0.3*np.array(after_shock_score).mean())
            no_recover_set.append(nsa)
        nsa = (np.array(recover_set).mean() + np.array(no_recover_set).mean())/2
        threshold_by_score.append(nsa)
    return threshold_by_score


X_viz, y_train_viz, _ = split.split_X_Y_for_VIZ(mimic_train, mode = 'mimic')
X_valid_viz, y_train_valid_viz, output_valid_viz = split.split_X_Y_for_VIZ(mimic_valid, mode = 'mimic')
X_test_viz, y_test_viz, test_output_viz = split.split_X_Y_for_VIZ(eicu_test, mode = 'eicu')

mimic_valid_nsas_all = []
eicu_test_nsas_all = []

mimic_origin = pd.read_csv(mimic_path, compression = 'gzip')
eicu_origin = pd.read_csv(eicu_path, compression='gzip')

for idx, model in enumerate(models):

    # model.fit(X_viz[col], y_train_viz)

    preds = model.predict(X_valid_viz)
    output_valid_viz['prediction_label'] = preds if idx != 0 else preds + 1
    output_valid_viz['prediction_prob_case2'] = model.predict_proba(X_valid_viz)[:, 1]
    output_valid_viz['prediction_prob_case3'] = model.predict_proba(X_valid_viz)[:, 2]
    output_valid_viz['prediction_prob_case4'] = model.predict_proba(X_valid_viz)[:, 3]

    test_preds = model.predict(X_test_viz)
    test_output_viz['prediction_label'] = test_preds if idx != 0 else test_preds + 1
    test_output_viz['prediction_prob_case2'] = model.predict_proba(X_test_viz)[:, 1]
    test_output_viz['prediction_prob_case3'] = model.predict_proba(X_test_viz)[:, 2]
    test_output_viz['prediction_prob_case4'] = model.predict_proba(X_test_viz)[:, 3]
    
    event_cohort_eval = output_valid_viz[output_valid_viz.stay_id.isin(eventset.stay_id.unique())]
    event_cohort_test = test_output_viz[test_output_viz.patientunitstayid.isin(event_eicu.patientunitstayid.unique())]

    
    recovery = mimic_origin[(mimic_origin['INDEX']=='CASE3_CASE4_DF')&(mimic_origin['Annotation']=='no_circ')]
    recovery_eval = event_cohort_eval[event_cohort_eval['stay_id'].isin(recovery.stay_id.unique())].sort_values(by=['stay_id', 'Time_since_ICU_admission'])
    no_recovery_eval = event_cohort_eval[~event_cohort_eval['stay_id'].isin(recovery.stay_id.unique())].sort_values(by=['stay_id', 'Time_since_ICU_admission'])

    recovery_e = eicu_origin[(eicu_origin['INDEX']=='CASE3_CASE4_DF')&(eicu_origin['Annotation']=='no_circ')]
    recovery_test = event_cohort_test[event_cohort_test['patientunitstayid'].isin(recovery_e.patientunitstayid.unique())].sort_values(by=['patientunitstayid', 'Time_since_ICU_admission'])
    no_recovery_test = event_cohort_test[~event_cohort_test['patientunitstayid'].isin(recovery_e.patientunitstayid.unique())].sort_values(by=['patientunitstayid', 'Time_since_ICU_admission'])

    threshold_by_score_val = NSA_Curve(recovery_eval, no_recovery_eval, 'mimic')
    threshold_by_score_test = NSA_Curve(recovery_test, no_recovery_test, 'eicu')
    
    mimic_valid_nsas_all.append([threshold_by_score_val])
    eicu_test_nsas_all.append([threshold_by_score_test])
    
    
    
#CLEAR-A-NSA

# 글꼴 설정
rc('font', family='Times New Roman')

# 각 임계값에 대한 NSA score 데이터
thresholds = np.linspace(0, 1, 21)

# 데이터 평균 및 표준오차 계산
mimic_mean = np.mean(mimic_valid_nsas_all, axis=0).squeeze()
eicu_mean = np.mean(eicu_test_nsas_all, axis=0).squeeze()

# 표준오차를 사용하여 95% 신뢰 구간 계산
t_critical = t.ppf(0.975, df=4)  # 95% 신뢰구간, 자유도 8 (n-1 for 9 samples)
mimic_se = sem(mimic_valid_nsas_all, axis=0).squeeze()
eicu_se = sem(eicu_test_nsas_all, axis=0).squeeze()
mimic_ci = t_critical * mimic_se
eicu_ci = t_critical * eicu_se

# 그래프 그리기
plt.figure(figsize=(10, 8))
plt.plot(thresholds, mimic_mean, label='Validation (MIMIC-IV)', color='darkred', linestyle='dashed')
plt.fill_between(thresholds, mimic_mean - mimic_ci, mimic_mean + mimic_ci, color='orangered', alpha=0.08, hatch='//')
plt.plot(thresholds, eicu_mean, label='Test (eICU)', color='cornflowerblue')
plt.fill_between(thresholds, eicu_mean - eicu_ci, eicu_mean + eicu_ci, color='navy', alpha=0.08, hatch='\\\\')

cumulative_std_area = np.trapz(np.abs(mimic_mean - eicu_mean) * np.cumsum(mimic_se + eicu_se), thresholds)

plt.fill_between(thresholds, mimic_mean, eicu_mean, color='gainsboro', alpha=0.2, label='Discrepancy: {:.4f}'.format(cumulative_std_area))

plt.xlabel('Threshold')
plt.ylabel('NSA Score')
plt.ylim(0,1)
plt.grid(True)
plt.legend(loc='best', fontsize='x-large')
plt.savefig('CLEAR-A-NSA.png')
plt.close()
    

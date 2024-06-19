import pandas as pd
import numpy as np
import time
import sys

from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.stats import sem, t
from sklearn.metrics import classification_report
from scipy.interpolate import make_interp_spline
module_path='/Users/DAHS/Desktop/ECP_CONT/ECP_SCL/Experiment/evaluation/'
if module_path not in sys.path:
    sys.path.append(module_path)
    
from integrating_embedding import *

import integrating_embedding
from imp import reload
reload(integrating_embedding)

import split
from imp import reload
reload(split)

import Multiclass_evaluation
reload(Multiclass_evaluation)

import get_hospital_eicu

emb_path_trn_mimic = '/Users/DAHS/Desktop/ECP_CONT/ECP_SCL/Cohort_selection/Train/result/emb_train_new_version(0313).npy'
emb_path_vld_mimic = '/Users/DAHS/Desktop/ECP_CONT/ECP_SCL/Cohort_selection/Train/result/emb_valid_new_version(0313).npy'

emb_path_trn_eicu = '/Users/DAHS/Desktop/ECP_CONT/ECP_SCL/Cohort_selection/Train/result/emb_eicu_new_version(0313).npy'

mimic_path = '/Users/DAHS/Desktop/ECP_CONT/ECP_SCL/Case Labeling/mimic_analysis(new_version0313).csv.gz'
eicu_path = '/Users/DAHS/Desktop/ECP_CONT/ECP_SCL/Case Labeling/eicu_analysis(new_version0313).csv.gz'

mimic_train_emb, mimic_valid_emb, event = integrating_embedding.integrating(mimic_path, emb_path_trn_mimic, emb_path_vld_mimic, _, 'mimic')
eicu_test_emb, event_eicu, hosp_id = integrating_embedding.integrating(eicu_path, emb_path_trn_eicu, _, _, 'eicu')
eicu_type = get_hospital_eicu.eicu_subgroup(eicu_test_emb)
unitadmitsource, unittype, unitstaytype = get_hospital_eicu.make_eicu_dataset(eicu_type)


X_train, y_train, output = split.split_X_Y(mimic_train_emb, mode = 'mimic')
X_valid, y_valid, valid_output = split.split_X_Y(mimic_valid_emb, mode = 'mimic')

models_all, result = Multiclass_evaluation.create_analysis(event, X_train, y_train, X_valid, valid_output, mode = 'emb')
external_validation_all = Multiclass_evaluation.external_validation_total_models(models_all, eicu_test_emb, X_train,event_eicu)


col = list(X_train.columns[226:])

add_col = ['MAP', 'MAP_3hr_avg', 'Time_since_ICU_admission', 'cum_use_vaso', 'Lactate_clearance_1h', 'vasoactive/inotropic', 'Sodium_fillna', 'over_lactic', 'ethnicity', 'gender','Age',
           'Fluids(ml)', 'Lactate_clearance_7h', 'MAP_change_1h', 'Respiratory Rate_fillna', 'lactate_down', 'Lactate_clearance_9h', 'MAP_change_5h', 'pH_change_5h', 'suspected_infection', 'HR', 'Glucose', 'Lactate']

for adding in add_col:
    col.append(adding)
    
models_reduced, result = Multiclass_evaluation.create_analysis(event, X_train[col], y_train, X_valid[col], valid_output, mode = 'emb')

test_col=col.copy()

for element in ['uniquepid', 'Unnamed: 0', 'patientunitstayid', 'Annotation', 'Case', 'after_shock_annotation', 'Shock_next_8h', 'INDEX', 'hospitalid', 'hospitaldischargeyear', 'progress']:
    test_col.append(element)
    
external_validation_all = Multiclass_evaluation.external_validation_total_models(models_all, eicu_test_emb, X_train, event_eicu)
external_validation_reduced = Multiclass_evaluation.external_validation_total_models(models_reduced, eicu_test_emb[test_col], X_train[col], event_eicu)

#NSA
reload(split)
X_test, y_test, test_output = split.split_X_Y(eicu_test_emb, mode = 'eicu')
#valid_all

ACID_result_valid_all = pd.DataFrame()
model_names = ['xgb', 'lgbm', 'catboost', 'rf', 'dt', 'svm-ovr', 'lr', 'naivebayes', 'knn']

for idx, model in enumerate(models_all):
    output = valid_output.copy()
    valid_preds = model.predict(X_valid)
    output['prediction_label'] = valid_preds if idx != 0 else valid_preds + 1
    output['prediction_prob'] = model.predict_proba(X_valid)[:, 1]
    
    cosine, _ = Multiclass_evaluation.ACID(output, event, 'mimic')
    
    # Ensure cosine is in a list or array if it's a scalar
    cosine = [cosine] if not isinstance(cosine, (list, np.ndarray)) else cosine
    
    # Convert to DataFrame
    cosine_df = pd.DataFrame({'cosine': cosine}, index=[model_names[idx]])
    ACID_result_valid_all = pd.concat([ACID_result_valid_all, cosine_df])
    

#NSA

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

ACID_result_all = pd.DataFrame()
model_names = ['xgb', 'lgbm', 'catboost', 'rf', 'dt', 'svm-ovr', 'lr', 'naivebayes', 'knn']

sample_test = test_output[test_output['patientunitstayid'].isin(event_eicu.patientunitstayid.unique())]
hat = sample_test[sample_test['after_shock_annotation']=='before_experience_shock'].groupby('patientunitstayid').count()['Time_since_ICU_admission']

#CLEAR-A
for idx, model in enumerate(models_all):
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
    

ACID_result = pd.DataFrame()
model_names = ['xgb', 'lgbm', 'catboost', 'rf', 'dt', 'svm-ovr', 'lr', 'naivebayes', 'knn']

#CLEAR-R
for idx, model in enumerate(models_reduced):
    output = test_output.copy()
    test_preds = model.predict(X_test[col])
    output['prediction_label'] = test_preds if idx != 0 else test_preds + 1
    output['prediction_prob'] = model.predict_proba(X_test[col])[:, 1]
    
    cosine, _ = Multiclass_evaluation.ACID(output, event_eicu, 'eicu')
    time = Caculate_alarm_time(output[output.patientunitstayid.isin(hat[hat > 8].index)], event_eicu, 'eicu')
    time = [time] if not isinstance(time, (list, np.ndarray)) else time
    # Ensure cosine is in a list or array if it's a scalar
    cosine = [cosine] if not isinstance(cosine, (list, np.ndarray)) else cosine
    
    # Convert to DataFrame
    cosine_df = pd.DataFrame({'cosine': cosine, 'time': time}, index=[model_names[idx]])
    ACID_result = pd.concat([ACID_result, cosine_df])
    
    
mimic_origin = pd.read_csv(mimic_path, compression = 'gzip')
eicu_origin = pd.read_csv(eicu_path, compression='gzip')


### Naive System Recall & Tracking system

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
            before_shock['prediction_label'] = before_shock['prediction_label'].replace({1:0, 2:1})
            
            before_shock['prediction_label'] = np.where(before_shock['prediction_prob_case2'] >= th, before_shock['Case'], 0)

            if before_shock[before_shock['Case']==1]['prediction_label'].sum() >= 1:
                before_shock_score = 1
            else:
                before_shock_score = 0

            #---------------------------------------------------------------------------------------------
            interest = recover_df[recover_df[stay_id_id]==stay]
            after_shock = interest.iloc[[-1]]
            after_shock['Case'] = after_shock['Case'].replace({3:1, 4:0, 0:1, 1:0, 2:0})
            after_shock['prediction_label'] = after_shock['prediction_label'].replace({3:1, 4:0, 0:1, 1:0, 2:0})

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
            before_shock['prediction_label'] = before_shock['prediction_label'].replace({1:0, 2:1})

            before_shock['prediction_label'] = np.where(before_shock['prediction_prob_case2'] >= th, before_shock['Case'], 0)
            
            if before_shock[before_shock['Case']==1]['prediction_label'].sum() >= 1:
                before_shock_score = 1
            else:
                before_shock_score = 0

            #---------------------------------------------------------------------------------------------
            interest = no_recover_df[no_recover_df[stay_id_id]==stay]
            after_shock = interest.iloc[[-1]]
            after_shock['Case'] = after_shock['Case'].replace({3:0, 4:1, 0:1, 1:0, 2:0})
            after_shock['prediction_label'] = after_shock['prediction_label'].replace({3:0, 4:1, 0:1, 1:0, 2:0})
            before_shock['prediction_label'] = np.where(before_shock['prediction_prob_case4'] >= th, before_shock['Case'], 0)
            after_shock_score = recall_score(after_shock['Case'], after_shock['prediction_label'])
            #---------------------------------------------------------------------------------------------

            nsa = (0.7*np.array(before_shock_score).mean() + 0.3*np.array(after_shock_score).mean())
            no_recover_set.append(nsa)
        nsa = (np.array(recover_set).mean() + np.array(no_recover_set).mean())/2
        threshold_by_score.append(nsa)
    return threshold_by_score

X_viz, y_train_viz, _ = split.split_X_Y_for_VIZ(mimic_train_emb, mode = 'mimic')
X_valid_viz, y_train_valid_viz, output_valid_viz = split.split_X_Y_for_VIZ(mimic_valid_emb, mode = 'mimic')
X_test_viz, y_test_viz, test_output_viz = split.split_X_Y_for_VIZ(eicu_test_emb, mode = 'eicu')

mimic_valid_nsas = []
eicu_test_nsas = []

for idx, model in enumerate(models_reduced):

    # model.fit(X_viz[col], y_train_viz)

    preds = model.predict(X_valid_viz[col])
    output_valid_viz['prediction_label'] = preds if idx != 0 else preds + 1
    output_valid_viz['prediction_prob_case2'] = model.predict_proba(X_valid_viz[col])[:, 1]
    output_valid_viz['prediction_prob_case3'] = model.predict_proba(X_valid_viz[col])[:, 2]
    output_valid_viz['prediction_prob_case4'] = model.predict_proba(X_valid_viz[col])[:, 3]

    test_preds = model.predict(X_test_viz[col])
    test_output_viz['prediction_label'] = test_preds if idx != 0 else test_preds + 1
    test_output_viz['prediction_prob_case2'] = model.predict_proba(X_test_viz[col])[:, 1]
    test_output_viz['prediction_prob_case3'] = model.predict_proba(X_test_viz[col])[:, 2]
    test_output_viz['prediction_prob_case4'] = model.predict_proba(X_test_viz[col])[:, 3]
    
    event_cohort_eval = output_valid_viz[output_valid_viz.stay_id.isin(event.stay_id.unique())]
    event_cohort_test = test_output_viz[test_output_viz.patientunitstayid.isin(event_eicu.patientunitstayid.unique())]

    
    recovery = mimic_origin[(mimic_origin['INDEX']=='CASE3_CASE4_DF')&(mimic_origin['Annotation']=='no_circ')]
    recovery_eval = event_cohort_eval[event_cohort_eval['stay_id'].isin(recovery.stay_id.unique())].sort_values(by=['stay_id', 'Time_since_ICU_admission'])
    no_recovery_eval = event_cohort_eval[~event_cohort_eval['stay_id'].isin(recovery.stay_id.unique())].sort_values(by=['stay_id', 'Time_since_ICU_admission'])

    recovery_e = eicu_origin[(eicu_origin['INDEX']=='CASE3_CASE4_DF')&(eicu_origin['Annotation']=='no_circ')]
    recovery_test = event_cohort_test[event_cohort_test['patientunitstayid'].isin(recovery_e.patientunitstayid.unique())].sort_values(by=['patientunitstayid', 'Time_since_ICU_admission'])
    no_recovery_test = event_cohort_test[~event_cohort_test['patientunitstayid'].isin(recovery_e.patientunitstayid.unique())].sort_values(by=['patientunitstayid', 'Time_since_ICU_admission'])

    threshold_by_score_val = NSA_Curve(recovery_eval, no_recovery_eval, 'mimic')
    threshold_by_score_test = NSA_Curve(recovery_test, no_recovery_test, 'eicu')
    
    mimic_valid_nsas.append([threshold_by_score_val])
    eicu_test_nsas.append([threshold_by_score_test])
    
    


# 글꼴 설정
rc('font', family='Times New Roman')

thresholds = np.linspace(0, 1, 21)

mimic_mean = np.mean(mimic_valid_nsas, axis=0).squeeze()
eicu_mean = np.mean(eicu_test_nsas, axis=0).squeeze()

t_critical = t.ppf(0.975, df=4) 
mimic_se = sem(mimic_valid_nsas, axis=0).squeeze()
eicu_se = sem(eicu_test_nsas, axis=0).squeeze()
mimic_ci = t_critical * mimic_se
eicu_ci = t_critical * eicu_se

plt.figure(figsize=(10, 8))
plt.plot(thresholds, mimic_mean, label='Validation (MIMIC-IV)', color='darkred', linestyle='dashed')
plt.fill_between(thresholds, mimic_mean - mimic_ci, mimic_mean + mimic_ci, color='orangered', alpha=0.08, hatch='//')
plt.plot(thresholds, eicu_mean, label='Test (eICU)', color='cornflowerblue')
plt.fill_between(thresholds, eicu_mean - eicu_ci, eicu_mean + eicu_ci, color='navy', alpha=0.08, hatch='\\\\')

cumulative_std_area = np.trapz(np.abs(mimic_mean - eicu_mean) * np.cumsum(mimic_se + eicu_se), thresholds)


plt.fill_between(thresholds, mimic_mean, eicu_mean, color='silver', alpha=0.2, label='Discrepancy: {:.4f}'.format(cumulative_std_area))

plt.xlabel('Threshold')
plt.ylabel('NSA Score')
plt.ylim(0,1)
plt.grid(True)
plt.legend(loc='best', fontsize='x-large')

plt.savefig('CLEAR-R-NSA.png')
plt.close()


#CLEAR-A
mimic_valid_nsas_all = []
eicu_test_nsas_all = []

for idx, model in enumerate(models_all):

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
    
    event_cohort_eval = output_valid_viz[output_valid_viz.stay_id.isin(event.stay_id.unique())]
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

import numpy as np
import matplotlib.pyplot as plt

rc('font', family='Times New Roman')

thresholds = np.linspace(0, 1, 21)

mimic_mean = np.mean(mimic_valid_nsas_all, axis=0).squeeze()
eicu_mean = np.mean(eicu_test_nsas_all, axis=0).squeeze()

t_critical = t.ppf(0.975, df=4)
mimic_se = sem(mimic_valid_nsas_all, axis=0).squeeze()
eicu_se = sem(eicu_test_nsas_all, axis=0).squeeze()
mimic_ci = t_critical * mimic_se
eicu_ci = t_critical * eicu_se


plt.figure(figsize=(10, 8))
plt.plot(thresholds, mimic_mean, label='Validation (MIMIC-IV)', color='darkred', linestyle='dashed')
plt.fill_between(thresholds, mimic_mean - mimic_ci, mimic_mean + mimic_ci, color='orangered', alpha=0.08, hatch='//')
plt.plot(thresholds, eicu_mean, label='Test (eICU)', color='cornflowerblue')
plt.fill_between(thresholds, eicu_mean - eicu_ci, eicu_mean + eicu_ci, color='navy', alpha=0.08, hatch='\\\\')

cumulative_std_area = np.trapz(np.abs(mimic_mean - eicu_mean) * np.cumsum(mimic_se + eicu_se), thresholds)

plt.fill_between(thresholds, mimic_mean, eicu_mean, color='silver', alpha=0.2, label='Discrepancy: {:.4f}'.format(cumulative_std_area))

plt.xlabel('Threshold')
plt.ylabel('NSA Score')
plt.ylim(0,1)
plt.grid(True)
plt.legend(loc='best', fontsize='x-large')
plt.savefig('CLEAR-A-NSA.png')
plt.close()

#Prognostic monitorinig system visulization

def min_max_scaling(column, min_value, max_value):
    
    if (min_value == 'none') & (max_value == 'none'):
        min_value = column.min()
        max_value = column.max()
        scaled_column = (column - min_value) / (max_value - min_value + 0.0000000001)
        return scaled_column, min_value, max_value
    
    else:
        scaled_column = (column - min_value) / (max_value - min_value + 0.0000000001)
        return scaled_column
    
    
eicu_path_pre = '/Users/DAHS/Desktop/ECP_CONT/ECP_SCL/Make Derived Variable/eicu_df_cp.csv.gz'
eicu_pre = pd.read_csv(eicu_path_pre, compression = 'gzip')
mimic_path_pre = '/Users/DAHS/Desktop/ECP_CONT/ECP_SCL/Make Derived Variable/mimic_df_cp.csv.gz'
mimic_pre = pd.read_csv(mimic_path_pre, compression = 'gzip')

X_viz, y_train_viz, output_viz = split.split_X_Y_for_VIZ(mimic_train_emb, mode = 'mimic')
recovery_set = output_viz[output_viz['stay_id'].isin(recovery.stay_id.unique())].sort_values(by=['stay_id', 'Time_since_ICU_admission'])
no_recovery_set = output_viz[~output_viz['stay_id'].isin(recovery.stay_id.unique())].sort_values(by=['stay_id', 'Time_since_ICU_admission'])


import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
# 4, "6(33965130)", "13(38132467)"
sec = 13

interest = recovery_set[recovery_set['stay_id'] == viz_id[sec]].reset_index(drop=True)

interest['Fluids(ml)_scaled'], fluids_min, fluids_max = min_max_scaling(interest['Fluids(ml)'], 'none', 'none')
interest['Lactate_scaled'], lactate_min, lactate_max = min_max_scaling(interest['Lactate'], 'none', 'none')
interest['MAP_scaled'], map_min, map_max = min_max_scaling(interest['MAP'], 'none', 'none')

event_index = interest[interest['after_shock_annotation']=='before_experience_shock'].index[-1]

Case1_2 = interest.iloc[:event_index+1]
Case3_4 = interest.iloc[event_index+1:]
Case3_4 = Case3_4.drop_duplicates(subset=['Time_since_ICU_admission'])

onset_of_shock_time = Case3_4['Time_since_ICU_admission'].iloc[0]
end_of_shock_time = Case3_4['Time_since_ICU_admission'].iloc[-1]

out_of_analysis = mimic_pre[mimic_pre['stay_id'] == viz_id[sec]][['Time_since_ICU_admission', 'MAP', 'Lactate', 'vasoactive/inotropic', 'Fluids(ml)']]
view = out_of_analysis[(out_of_analysis['Time_since_ICU_admission'] > end_of_shock_time) & (out_of_analysis['Time_since_ICU_admission'] < end_of_shock_time+5)]

view['Fluids(ml)_scaled'] = min_max_scaling(view['Fluids(ml)'], fluids_min, fluids_max)
view['Lactate_scaled'] = min_max_scaling(view['Lactate'],lactate_min, lactate_max)
view['MAP_scaled'] = min_max_scaling(view['MAP'],  map_min, map_max)

fig, ax1 = plt.subplots(3, 1, figsize=(15, 9)) 

# Case1_2
ax1[0].plot(Case1_2['Time_since_ICU_admission'], Case1_2['prediction_prob_case2'], linestyle='-', color='darkmagenta', label='Shock within 8h ris')
# Case3_4
ax1[0].plot(Case3_4['Time_since_ICU_admission'], Case3_4['prediction_prob_case4'], linestyle='-', color='red', label='continuous shock risk')

ax1[0].axvspan(onset_of_shock_time-1, onset_of_shock_time, color='lightcoral')
ax1[0].axvspan(onset_of_shock_time, end_of_shock_time, color='mistyrose', alpha=0.6)
ax1[0].axvspan(end_of_shock_time, end_of_shock_time+5, color='lightgreen', alpha=0.4, label = 'Recovery')
# Onset of shock
ax1[0].text(onset_of_shock_time - 0.5 , 0.5 * (Case1_2['prediction_prob_case2'].iloc[-1] + Case3_4['prediction_prob_case4'].iloc[0]), 'Onset of \nShock', ha='center', va='center', fontsize=12, color='black', fontweight='bold')

ymin, ymax = 0, 1 
ax1[0].set_yticks(np.arange(ymin, ymax + 0.1, 0.2))

ax1[0].set_ylabel('Risk Score')
ax1[0].grid(True)

# Intervention ----------------------------------------------------------------------------------------------------


vital_lab = pd.concat([interest[view.columns], view.reset_index(drop=True)], axis = 0, ignore_index=True).drop_duplicates(subset=['Time_since_ICU_admission']).sort_values(by=['Time_since_ICU_admission'])

# Case1_2
ax1[1].plot(vital_lab['Time_since_ICU_admission'], (vital_lab['vasoactive/inotropic']/2), linestyle='-', color='darkred', label='Use of Vasopressor')
# Case3_4
ax1[1].plot(vital_lab['Time_since_ICU_admission'], vital_lab['Fluids(ml)_scaled'], linestyle='-', color='limegreen', label='Fluids(ml)')


ax1[1].axvspan(onset_of_shock_time-1, onset_of_shock_time, color='lightcoral')
ax1[1].axvspan(onset_of_shock_time, end_of_shock_time, color='mistyrose', alpha=0.6)
ax1[1].axvspan(end_of_shock_time, end_of_shock_time+5, color='lightgreen', alpha=0.4, label = 'Recovery')

ax1[1].set_ylabel('Intervention')
# ax1[1].set_xlabel('Time Since ICU Admission (Hours)')
ax1[1].grid(True)
# ax1[1].legend(loc = 'upper left', fontsize = 10)

# Vital sign & Lab test--------------------------------------------------------------------------------------------

# fig, ax1 = plt.subplots(3, 1, figsize=(18, 15))
vital_lab = pd.concat([interest[view.columns], view.reset_index(drop=True)], axis = 0, ignore_index=True).drop_duplicates(subset=['Time_since_ICU_admission']).sort_values(by=['Time_since_ICU_admission'])

MAP_before_shock = np.linspace(vital_lab['Time_since_ICU_admission'].min(), vital_lab['Time_since_ICU_admission'].max(), 300) 
spl = make_interp_spline(vital_lab['Time_since_ICU_admission'].values, vital_lab['MAP'].values, k=3) 
power_smooth_map_bs = spl(MAP_before_shock)

Lactate_before_shock = np.linspace(vital_lab['Time_since_ICU_admission'].min(), vital_lab['Time_since_ICU_admission'].max(), 300) 
spl = make_interp_spline(vital_lab['Time_since_ICU_admission'].values, vital_lab['Lactate'].values, k=3) 
power_smooth_lactate_bs = spl(Lactate_before_shock)

ax2 = ax1[2].twinx() 

# MAP 데이터 플로팅
ax1[2].plot(MAP_before_shock, power_smooth_map_bs, linestyle='-', color='gray', label='MAP', alpha=0.5)
ax1[2].set_ylabel('MAP', color='Black')
ax1[2].tick_params(axis='y', labelcolor='Black')

# Lactate 데이터 플로팅
ax2.plot(Lactate_before_shock, power_smooth_lactate_bs, linestyle='-', color='darkblue', label='Lactate', alpha=0.5)
ax2.set_ylabel('Lactate', color='Black')
ax2.tick_params(axis='y', labelcolor='Black') 

# 충격과 회복 기간 표시
ax1[2].axvspan(onset_of_shock_time-1, onset_of_shock_time, color='lightcoral')
ax1[2].axvspan(onset_of_shock_time, end_of_shock_time, color='mistyrose', alpha=0.6)
ax1[2].axvspan(end_of_shock_time, end_of_shock_time+5, color='lightgreen', alpha=0.4, label='Recovery')

ax1[2].grid(True)

fig.tight_layout()
plt.savefig('PMV.png')



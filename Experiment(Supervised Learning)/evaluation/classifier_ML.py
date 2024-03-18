from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression  
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import os
from sklearn.calibration import CalibratedClassifierCV

from sklearn.ensemble import AdaBoostClassifier
from catboost import CatBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# dll_directory = r'C:\Users\DAHS\anaconda3\envs\DL\Lib\site-packages\thundersvm'
# original_path = os.environ['PATH']

# os.environ['PATH'] = dll_directory + ';' + original_path
# try:
#     from thundersvm import SVC
# finally:
#     os.environ['PATH'] = original_path

from sklearn.svm import LinearSVC

from sklearn.pipeline import make_pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_recall_curve, auc

from lightgbm import plot_importance
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

import pandas as pd
import numpy as np


def classifier(algorithm, x, y, x_valid, valid_output, mode = 'emb'):
    
    if mode == 'emb':
 
        if algorithm == 'lgbm':
            # lgbm_wrapper = LGBMClassifier(n_estimators = 10, random_state = 42, extra_trees = True, verbose=-1)
            lgbm_wrapper = LGBMClassifier(random_state = 42, verbose=-1)
            
            lgbm_wrapper.fit(x, y)
            
            valid_preds = lgbm_wrapper.predict(x_valid)
            valid_output['prediction_label'] = valid_preds
            valid_output['prediction_prob'] = lgbm_wrapper.predict_proba(x_valid)[:, 1]
        
            return lgbm_wrapper, valid_output
        
        elif algorithm == 'xgb':
            xgbm = XGBClassifier(random_state=42)
            xgbm.fit(x, y-1)
            
            valid_preds = xgbm.predict(x_valid)
            valid_output['prediction_label'] = valid_preds
            valid_output['prediction_prob'] = xgbm.predict_proba(x_valid)[:, 1]
            valid_output['prediction_label'] = valid_output['prediction_label']+1
            
            return xgbm, valid_output
        
        elif algorithm == 'rf':
            rf = RandomForestClassifier(random_state = 42)
            rf.fit(x, y)
            
            valid_preds = rf.predict(x_valid)
            valid_output['prediction_label'] = valid_preds
            valid_output['prediction_prob'] = rf.predict_proba(x_valid)[:, 1]

            return rf, valid_output
        
        elif algorithm == 'dt':
        
            tree = DecisionTreeClassifier(random_state=42).fit(x, y)
            valid_preds = tree.predict(x_valid)
            valid_output['prediction_label'] = valid_preds
            valid_output['prediction_prob'] = tree.predict_proba(x_valid)[:, 1]
            
            return tree, valid_output
        
        elif algorithm == 'lr':
            
            mMscaler = MinMaxScaler()
            mMscaler.fit(x)
            x_scaled = mMscaler.transform(x)
            x_valid_scaled = mMscaler.transform(x_valid)
            
            logit_regression = LogisticRegression(random_state = 42) 
            logit_regression.fit(x_scaled, y)

            valid_preds = logit_regression.predict(x_valid_scaled)
            valid_output['prediction_label'] = valid_preds
            valid_output['prediction_prob'] = logit_regression.predict_proba(x_valid_scaled)[:, 1]
            
            return logit_regression, valid_output
        
        # elif algorithm == 'svm-ovo':
            
        #     mMscaler = MinMaxScaler()
        #     mMscaler.fit(x)
        #     x_scaled = mMscaler.transform(x)
        #     x_valid_scaled = mMscaler.transform(x_valid)
            
        #     svc_ovo = Smake_pipeline(LinearSVC(dual='auto', random_state=42, tol=1e-05, multi_class = 'crammer_single'))
        #     svc_ovo.fit(x_scaled, y)

        #     valid_preds = svc_ovo.predict(x_valid_scaled)
        #     valid_output['prediction_label'] = valid_preds
        #     valid_output['prediction_prob'] = svc_ovo.predict_proba(x_valid_scaled)[:, 1]
      
        #     return svc_ovo, valid_output
        
        elif algorithm == 'svm-ovr':
            
            mMscaler = MinMaxScaler()
            mMscaler.fit(x)
            x_scaled = mMscaler.transform(x)
            x_valid_scaled = mMscaler.transform(x_valid)
    
            svc_ovr = LinearSVC(dual='auto', random_state=42, tol=1e-05, multi_class = 'ovr')
            calibrated_svm = CalibratedClassifierCV(svc_ovr)
            
            calibrated_svm.fit(x_scaled, y)

            valid_preds = calibrated_svm.predict(x_valid_scaled)
            valid_output['prediction_label'] = valid_preds
            valid_output['prediction_prob'] = calibrated_svm.predict_proba(x_valid_scaled)[:, 1]
      
            return calibrated_svm, valid_output
        
        
        elif algorithm == 'catboost':
            cat = CatBoostClassifier(random_state=42, has_time = True, verbose=False)
            cat.fit(x, y)
            
            valid_preds = cat.predict(x_valid)
            valid_output['prediction_label'] = valid_preds
            valid_output['prediction_prob'] = cat.predict_proba(x_valid)[:, 1]
        
            return cat, valid_output
        
        elif algorithm == 'naivebayes':
            
            mMscaler = MinMaxScaler()
            mMscaler.fit(x)
            x_scaled = mMscaler.transform(x)
            x_valid_scaled = mMscaler.transform(x_valid)
            
            nb = GaussianNB()
            nb.fit(x, y)
            
            valid_preds = nb.predict(x_valid)
            valid_output['prediction_label'] = valid_preds
            valid_output['prediction_prob'] = nb.predict_proba(x_valid)[:, 1]
        
            return nb, valid_output
        
        elif algorithm == 'knn':
            
            mMscaler = MinMaxScaler()
            mMscaler.fit(x)
            x_scaled = mMscaler.transform(x)
            x_valid_scaled = mMscaler.transform(x_valid)
            
            knn = KNeighborsClassifier(weights = 'distance')
            knn.fit(x, y)
            
            valid_preds = knn.predict(x_valid)
            valid_output['prediction_label'] = valid_preds
            valid_output['prediction_prob'] = knn.predict_proba(x_valid)[:, 1]
        
            return knn, valid_output
        
    else:
        if algorithm == 'lgbm':
            # lgbm_wrapper = LGBMClassifier(n_estimators = 10, random_state = 42, extra_trees = True, verbose=-1)
            lgbm_wrapper = LGBMClassifier(random_state = 42, verbose=-1)
            lgbm_wrapper.fit(x, y)
            
            valid_preds = lgbm_wrapper.predict(x_valid)
            valid_output['prediction_label'] = valid_preds
            valid_output['prediction_prob'] = lgbm_wrapper.predict_proba(x_valid)[:, 1]
        
            return lgbm_wrapper, valid_output
        
        elif algorithm == 'xgb':
            xgbm = XGBClassifier(random_state=42)
            xgbm.fit(x, y-1)
            
            valid_preds = xgbm.predict(x_valid)
            valid_output['prediction_label'] = valid_preds
            valid_output['prediction_prob'] = xgbm.predict_proba(x_valid)[:, 1]
            valid_output['prediction_label'] = valid_output['prediction_label']+1
            
            return xgbm, valid_output
        
        elif algorithm == 'rf':
            rf = RandomForestClassifier(random_state=42)
            rf.fit(x, y)
            
            valid_preds = rf.predict(x_valid)
            valid_output['prediction_label'] = valid_preds
            valid_output['prediction_prob'] = rf.predict_proba(x_valid)[:, 1]

            return rf, valid_output
        
        elif algorithm == 'dt':
        
            tree = DecisionTreeClassifier(random_state=42).fit(x, y)
            valid_preds = tree.predict(x_valid)
            valid_output['prediction_label'] = valid_preds
            valid_output['prediction_prob'] = tree.predict_proba(x_valid)[:, 1]
            
            return tree, valid_output
        
        elif algorithm == 'lr':
            
            mMscaler = MinMaxScaler()
            mMscaler.fit(x)
            x_scaled = mMscaler.transform(x)
            x_valid_scaled = mMscaler.transform(x_valid)
            
            logit_regression = LogisticRegression(random_state = 42) 
            logit_regression.fit(x_scaled, y)

            valid_preds = logit_regression.predict(x_valid_scaled)
            valid_output['prediction_label'] = valid_preds
            valid_output['prediction_prob'] = logit_regression.predict_proba(x_valid_scaled)[:, 1]
            
            return logit_regression, valid_output
        
        # elif algorithm == 'svm-ovo':
            
        #     mMscaler = MinMaxScaler()
        #     mMscaler.fit(x)
        #     x_scaled = mMscaler.transform(x)
        #     x_valid_scaled = mMscaler.transform(x_valid)
            
        #     svc_ovo = SVC(kernel='linear', probability=True, decision_function_shape='ovo')
        #     svc_ovo.fit(x_scaled, y)

        #     valid_preds = svc_ovo.predict(x_valid_scaled)
        #     valid_output['prediction_label'] = valid_preds
        #     valid_output['prediction_prob'] = svc_ovo.predict_proba(x_valid_scaled)[:, 1]
      
        #     return svc_ovo, valid_output
        
        elif algorithm == 'svm-ovr':
            
            mMscaler = MinMaxScaler()
            mMscaler.fit(x)
            x_scaled = mMscaler.transform(x)
            x_valid_scaled = mMscaler.transform(x_valid)
    
            svc_ovr = LinearSVC(dual='auto', random_state=42, tol=1e-05, multi_class = 'ovr')
            calibrated_svm = CalibratedClassifierCV(svc_ovr)
            
            calibrated_svm.fit(x_scaled, y)

            valid_preds = calibrated_svm.predict(x_valid_scaled)
            valid_output['prediction_label'] = valid_preds
            valid_output['prediction_prob'] = calibrated_svm.predict_proba(x_valid_scaled)[:, 1]
      
            return calibrated_svm, valid_output
        
        
        
        elif algorithm == 'catboost':
            cat = CatBoostClassifier(random_state=42, has_time = True, verbose=False)
            cat.fit(x, y)
            
            valid_preds = cat.predict(x_valid)
            valid_output['prediction_label'] = valid_preds
            valid_output['prediction_prob'] = cat.predict_proba(x_valid)[:, 1]
        
            return cat, valid_output
        
        elif algorithm == 'naivebayes':
            
            mMscaler = MinMaxScaler()
            mMscaler.fit(x)
            x_scaled = mMscaler.transform(x)
            x_valid_scaled = mMscaler.transform(x_valid)
            
            nb = GaussianNB()
            nb.fit(x, y)
            
            valid_preds = nb.predict(x_valid)
            valid_output['prediction_label'] = valid_preds
            valid_output['prediction_prob'] = nb.predict_proba(x_valid)[:, 1]
        
            return nb, valid_output
        
        elif algorithm == 'knn':
            
            mMscaler = MinMaxScaler()
            mMscaler.fit(x)
            x_scaled = mMscaler.transform(x)
            x_valid_scaled = mMscaler.transform(x_valid)
            
            knn = KNeighborsClassifier(weights = 'distance')
            knn.fit(x, y)
            
            valid_preds = knn.predict(x_valid)
            valid_output['prediction_label'] = valid_preds
            valid_output['prediction_prob'] = knn.predict_proba(x_valid)[:, 1]
        
            return knn, valid_output


def event_metric(event,inference_output,mode, model_name):
   
    if mode == 'mimic':
        icu_stay = 'stay_id'
    else:
        icu_stay = 'patientunitstayid'

    usecol = [icu_stay,'Time_since_ICU_admission', 'Annotation', 'progress', 'Case', 'prediction_label']

    event_set = event[usecol[:-1]].copy()
    event_set['prediction_label'] = 'label'
    output_set = inference_output[usecol]
    output_set['prediction_label'] = output_set['prediction_label']

    event_set = event_set[event_set[icu_stay].isin(output_set[icu_stay].unique())]
    
    event_set_reset = event_set.reset_index(drop=True)
    output_set_reset = output_set.reset_index(drop=True)

    total_set = pd.concat([event_set_reset, output_set_reset], axis = 0, ignore_index=True).reset_index(drop=True).sort_values(by='Time_since_ICU_admission')

    
    #### amb는 평가에서 제외해보자
    total_set = total_set[~(total_set['Annotation']=='ambiguous')]
    ###
    total_event = event_set.Case.value_counts()['event']
    captured_event = 0

    captured_trajectory = []
    non_captured_trajectory = []

    True_positive = 0
    True__False_positive = 0

    for stayid in total_set[icu_stay].unique():
        
        interest = total_set[total_set[icu_stay]==stayid]
        
        if any(interest['Case']=='event'):
            
            total_event += 1
            event_time = interest['Time_since_ICU_admission'].iloc[-1]
            time_window = event_time - 8
            capture_before_8h = interest[(interest['Time_since_ICU_admission'] >= time_window) & (interest['Time_since_ICU_admission'] < event_time)]
            
            ### amb는 평가에서 제외해보자
            if len(capture_before_8h) >= 1:
            ###
                if capture_before_8h['prediction_label'].sum() >= 1:
                    captured_event += 1
                    captured_trajectory.append(stayid)
                
                else:
                    non_captured_trajectory.append(stayid)
                
                
    for stayid in total_set[icu_stay].unique():
        
        interest = total_set[total_set[icu_stay]==stayid]
        
        total_positive_model = interest[interest['prediction_label']==1]
        
        tp_fp_count = len(total_positive_model)
        tp_count = len(total_positive_model[total_positive_model['Case']==1])
        
        True_positive += tp_count
        True__False_positive += tp_fp_count
                
    try:
        event_recall = np.round((captured_event / (len(captured_trajectory)+len(non_captured_trajectory))), 4)
    except:
        print('ZeroDivisionError')
        event_recall = 0
    
    try:
        event_precision = np.round((True_positive / True__False_positive), 4)
    except:
        print('ZeroDivisionError')
        event_precision = 0
        
    beta = 1.5
    
    try:
    
        event_ioc = (1+(beta)**2)*(event_recall*event_precision)/((beta)**2 * event_precision + event_recall)
    except:
        print('ZeroDivisionError')
        event_ioc = 0
    
    accuracy = accuracy_score(inference_output.Case, inference_output.prediction_label)
    precision = precision_score(inference_output.Case, inference_output.prediction_label)

    evaluation_score = {'acc':[accuracy],'recall':[event_recall], 'precision':[event_precision], 'raw_precision':[precision], 'IoC(1.5)':[event_ioc]}
    
    
    return pd.DataFrame(evaluation_score, index=[model_name])


def event_metric_ARDS4h(event, inference_output, mode, model_name):
       
    if mode == 'mimic':
        icu_stay = 'stay_id'
    else:
        icu_stay = 'patientunitstayid'

    usecol = [icu_stay,'Time_since_ICU_admission', 'Annotation_ARDS', 'ARDS_next_4h', 'prediction_label']

    event_set = event[usecol[:-1]].copy()
    event_set['prediction_label'] = 'label'
    output_set = inference_output[usecol]
    output_set['prediction_label'] = output_set['prediction_label']

    event_set = event_set[event_set[icu_stay].isin(output_set[icu_stay].unique())]
    
    event_set_reset = event_set.reset_index(drop=True)
    output_set_reset = output_set.reset_index(drop=True)

    total_set = pd.concat([event_set_reset, output_set_reset], axis = 0, ignore_index=True).reset_index(drop=True).sort_values(by='Time_since_ICU_admission')

    total_event = event_set.ARDS_next_4h.value_counts()['event']
    captured_event = 0

    captured_trajectory = []
    non_captured_trajectory = []

    True_positive = 0
    True__False_positive = 0

    for stayid in total_set[icu_stay].unique():
        
        interest = total_set[total_set[icu_stay]==stayid]
        
        if any(interest['ARDS_next_4h']=='event'):
            
            total_event += 1
            event_time = interest['Time_since_ICU_admission'].iloc[-1]
            time_window = event_time - 4
            capture_before_8h = interest[(interest['Time_since_ICU_admission'] > time_window) & (interest['Time_since_ICU_admission'] < event_time)]
            
            if capture_before_8h['prediction_label'].sum() >= 1:
                captured_event += 1
                captured_trajectory.append(stayid)
            
            else:
                non_captured_trajectory.append(stayid)
                
                
    for stayid in total_set[icu_stay].unique():
        
        interest = total_set[total_set[icu_stay]==stayid]
        
        total_positive_model = interest[interest['prediction_label']==1]
        
        tp_fp_count = len(total_positive_model)
        tp_count = len(total_positive_model[total_positive_model['ARDS_next_4h']==1])
        
        True_positive += tp_count
        True__False_positive += tp_fp_count
                
    try:
        event_recall = np.round((captured_event / (len(captured_trajectory)+len(non_captured_trajectory))), 4)
    except:
        print('ZeroDivisionError')
        event_recall = 0
    
    try:
        event_precision = np.round((True_positive / True__False_positive), 4)
    except:
        print('ZeroDivisionError')
        event_precision = 0
        
    beta = 1.5
    try:
    
        event_ioc = (1+(beta)**2)*(event_recall*event_precision)/((beta)**2 * event_precision + event_recall)
    except:
        print('ZeroDivisionError')
        event_ioc = 0
    accuracy = accuracy_score(inference_output.ARDS_next_4h, inference_output.prediction_label)
    precision = precision_score(inference_output.ARDS_next_4h, inference_output.prediction_label)
    
    evaluation_score = {'acc':[accuracy],'recall':[event_recall], 'precision':[event_precision], 'raw_precision':[precision], 'IoC(1.5)':[event_ioc]}
    
    
    return pd.DataFrame(evaluation_score, index=[model_name])

def ARDS4h_AUPRC(event, inference_output, mode, model_name):
    
    if mode == 'mimic':
        icu_stay = 'stay_id'
    else:
        icu_stay = 'patientunitstayid'
        
    evaluation_interest = inference_output[inference_output[icu_stay].isin(event[icu_stay].unique())]
    
    # 예측 확률과 실제 레이블이 필요
    y_true = evaluation_interest.iloc[:, -3] # 실제 레이블
    y_scores =evaluation_interest.prediction_prob # 예측 확률

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    new_recalls = []
    
    for threshold in thresholds:
        evaluation_interest['prediction_label'] = evaluation_interest['prediction_prob'].apply(lambda x: 1 if x > threshold else 0).astype(int)
        result = event_metric_ARDS4h(event, evaluation_interest, mode, model_name)
        recall = result['recall'].values[0]
        new_recalls.append(recall)
    
    auprc = auc(np.array(new_recalls), precisions[:-1])
    
    return auprc

def event_metric_ARDS8h(event,inference_output,mode, model_name):
       
    if mode == 'mimic':
        icu_stay = 'stay_id'
    else:
        icu_stay = 'patientunitstayid'

    usecol = [icu_stay,'Time_since_ICU_admission', 'Annotation_ARDS', 'ARDS_next_8h', 'prediction_label']

    event_set = event[usecol[:-1]].copy()
    event_set['prediction_label'] = 'label'
    output_set = inference_output[usecol]
    output_set['prediction_label'] = output_set['prediction_label']

    event_set = event_set[event_set[icu_stay].isin(output_set[icu_stay].unique())]
    
    event_set_reset = event_set.reset_index(drop=True)
    output_set_reset = output_set.reset_index(drop=True)

    total_set = pd.concat([event_set_reset, output_set_reset], axis = 0, ignore_index=True).reset_index(drop=True).sort_values(by='Time_since_ICU_admission')

    total_event = event_set.ARDS_next_8h.value_counts()['event']
    captured_event = 0

    captured_trajectory = []
    non_captured_trajectory = []

    True_positive = 0
    True__False_positive = 0

    for stayid in total_set[icu_stay].unique():
        
        interest = total_set[total_set[icu_stay]==stayid]
        
        if any(interest['ARDS_next_8h']=='event'):
            
            total_event += 1
            event_time = interest['Time_since_ICU_admission'].iloc[-1]
            time_window = event_time - 8
            capture_before_8h = interest[(interest['Time_since_ICU_admission'] > time_window) & (interest['Time_since_ICU_admission'] < event_time)]
            
            if capture_before_8h['prediction_label'].sum() >= 1:
                captured_event += 1
                captured_trajectory.append(stayid)
            
            else:
                non_captured_trajectory.append(stayid)
                
                
    for stayid in total_set[icu_stay].unique():
        
        interest = total_set[total_set[icu_stay]==stayid]
        
        total_positive_model = interest[interest['prediction_label']==1]
        
        tp_fp_count = len(total_positive_model)
        tp_count = len(total_positive_model[total_positive_model['ARDS_next_8h']==1])
        
        True_positive += tp_count
        True__False_positive += tp_fp_count
                
    try:
        event_recall = np.round((captured_event / (len(captured_trajectory)+len(non_captured_trajectory))), 4)
    except:
        print('ZeroDivisionError')
        event_recall = 0
    
    try:
        event_precision = np.round((True_positive / True__False_positive), 4)
    except:
        print('ZeroDivisionError')
        event_precision = 0
        
    beta = 1.5
    try:
    
        event_ioc = (1+(beta)**2)*(event_recall*event_precision)/((beta)**2 * event_precision + event_recall)
    except:
        print('ZeroDivisionError')
        event_ioc = 0
    accuracy = accuracy_score(inference_output.ARDS_next_8h, inference_output.prediction_label)
    precision = precision_score(inference_output.ARDS_next_8h, inference_output.prediction_label)
    
    evaluation_score = {'acc':[accuracy],'recall':[event_recall], 'precision':[event_precision], 'raw_precision':[precision], 'IoC(1.5)':[event_ioc]}
    
    
    return pd.DataFrame(evaluation_score, index=[model_name])

def ARDS8h_AUPRC(event, inference_output, mode, model_name):
    
    if mode == 'mimic':
        icu_stay = 'stay_id'
    else:
        icu_stay = 'patientunitstayid'
        
    evaluation_interest = inference_output[inference_output[icu_stay].isin(event[icu_stay].unique())]
    
    # 예측 확률과 실제 레이블이 필요
    y_true = evaluation_interest.iloc[:, -3] # 실제 레이블
    y_scores =evaluation_interest.prediction_prob # 예측 확률

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    new_recalls = []
    
    for threshold in thresholds:
        evaluation_interest['prediction_label'] = evaluation_interest['prediction_prob'].apply(lambda x: 1 if x > threshold else 0).astype(int)
        result = event_metric_ARDS8h(event, evaluation_interest, mode, model_name)
        recall = result['recall'].values[0]
        new_recalls.append(recall)
    
    auprc = auc(np.array(new_recalls), precisions[:-1])
    
    return auprc


def event_metric_SIC4h(event,inference_output,mode, model_name):
       
    if mode == 'mimic':
        icu_stay = 'stay_id'
    else:
        icu_stay = 'patientunitstayid'

    usecol = [icu_stay,'Time_since_ICU_admission', 'Annotation_SIC', 'SIC_next_4h', 'prediction_label']

    event_set = event[usecol[:-1]].copy()
    event_set['prediction_label'] = 'label'
    output_set = inference_output[usecol]
    output_set['prediction_label'] = output_set['prediction_label']

    event_set = event_set[event_set[icu_stay].isin(output_set[icu_stay].unique())]
    
    event_set_reset = event_set.reset_index(drop=True)
    output_set_reset = output_set.reset_index(drop=True)

    total_set = pd.concat([event_set_reset, output_set_reset], axis = 0, ignore_index=True).reset_index(drop=True).sort_values(by='Time_since_ICU_admission')

    total_event = event_set.SIC_next_4h.value_counts()['event']
    captured_event = 0

    captured_trajectory = []
    non_captured_trajectory = []

    True_positive = 0
    True__False_positive = 0

    for stayid in total_set[icu_stay].unique():
        
        interest = total_set[total_set[icu_stay]==stayid]
        
        if any(interest['SIC_next_4h']=='event'):
            
            total_event += 1
            event_time = interest['Time_since_ICU_admission'].iloc[-1]
            time_window = event_time - 4
            capture_before_8h = interest[(interest['Time_since_ICU_admission'] > time_window) & (interest['Time_since_ICU_admission'] < event_time)]
            
            if capture_before_8h['prediction_label'].sum() >= 1:
                captured_event += 1
                captured_trajectory.append(stayid)
            
            else:
                non_captured_trajectory.append(stayid)
                
                
    for stayid in total_set[icu_stay].unique():
        
        interest = total_set[total_set[icu_stay]==stayid]
        
        total_positive_model = interest[interest['prediction_label']==1]
        
        tp_fp_count = len(total_positive_model)
        tp_count = len(total_positive_model[total_positive_model['SIC_next_4h']==1])
        
        True_positive += tp_count
        True__False_positive += tp_fp_count
                
    try:
        event_recall = (captured_event / (len(captured_trajectory)+len(non_captured_trajectory)))
    except:
        print('ZeroDivisionError')
        event_recall = 0
    
    try:
        event_precision = np.round((True_positive / True__False_positive), 4)
    except:
        print('ZeroDivisionError')
        event_precision = 0
        
    beta = 1.5
    try:
    
        event_ioc = (1+(beta)**2)*(event_recall*event_precision)/((beta)**2 * event_precision + event_recall)
    except:
        print('ZeroDivisionError')
        event_ioc = 0
        
    accuracy = accuracy_score(inference_output.SIC_next_4h, inference_output.prediction_label)
    precision = precision_score(inference_output.SIC_next_4h, inference_output.prediction_label)
    
    evaluation_score = {'acc':[accuracy],'recall':[event_recall], 'precision':[event_precision], 'raw_precision':[precision], 'IoC(1.5)':[event_ioc]}
    
    
    return pd.DataFrame(evaluation_score, index=[model_name])

def SIC4h_AUPRC(event, inference_output, mode, model_name):
    
    if mode == 'mimic':
        icu_stay = 'stay_id'
    else:
        icu_stay = 'patientunitstayid'
        
    evaluation_interest = inference_output[inference_output[icu_stay].isin(event[icu_stay].unique())]
    
    # 예측 확률과 실제 레이블이 필요
    y_true = evaluation_interest.iloc[:, -3] # 실제 레이블
    y_scores =evaluation_interest.prediction_prob # 예측 확률

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    new_recalls = []
    
    for threshold in thresholds:
        evaluation_interest['prediction_label'] = evaluation_interest['prediction_prob'].apply(lambda x: 1 if x > threshold else 0).astype(int)
        result = event_metric_SIC4h(event, evaluation_interest, mode, model_name)
        recall = result['recall'].values[0]
        new_recalls.append(recall)
    
    auprc = auc(np.array(new_recalls), precisions[:-1])
    
    return auprc

def event_metric_SIC8h(event,inference_output,mode, model_name):
       
    if mode == 'mimic':
        icu_stay = 'stay_id'
    else:
        icu_stay = 'patientunitstayid'

    usecol = [icu_stay,'Time_since_ICU_admission', 'Annotation_SIC', 'SIC_next_8h', 'prediction_label']

    event_set = event[usecol[:-1]].copy()
    event_set['prediction_label'] = 'label'
    output_set = inference_output[usecol]
    output_set['prediction_label'] = output_set['prediction_label']

    event_set = event_set[event_set[icu_stay].isin(output_set[icu_stay].unique())]
    
    event_set_reset = event_set.reset_index(drop=True)
    output_set_reset = output_set.reset_index(drop=True)

    total_set = pd.concat([event_set_reset, output_set_reset], axis = 0, ignore_index=True).reset_index(drop=True).sort_values(by='Time_since_ICU_admission')

    total_event = event_set.SIC_next_8h.value_counts()['event']
    captured_event = 0

    captured_trajectory = []
    non_captured_trajectory = []

    True_positive = 0
    True__False_positive = 0

    for stayid in total_set[icu_stay].unique():
        
        interest = total_set[total_set[icu_stay]==stayid]
        
        if any(interest['SIC_next_8h']=='event'):
            
            total_event += 1
            event_time = interest['Time_since_ICU_admission'].iloc[-1]
            time_window = event_time - 8
            capture_before_8h = interest[(interest['Time_since_ICU_admission'] > time_window) & (interest['Time_since_ICU_admission'] < event_time)]
            
            if capture_before_8h['prediction_label'].sum() >= 1:
                captured_event += 1
                captured_trajectory.append(stayid)
            
            else:
                non_captured_trajectory.append(stayid)
                
                
    for stayid in total_set[icu_stay].unique():
        
        interest = total_set[total_set[icu_stay]==stayid]
        
        total_positive_model = interest[interest['prediction_label']==1]
        
        tp_fp_count = len(total_positive_model)
        tp_count = len(total_positive_model[total_positive_model['SIC_next_8h']==1])
        
        True_positive += tp_count
        True__False_positive += tp_fp_count
                
    try:
        event_recall = (captured_event / (len(captured_trajectory)+len(non_captured_trajectory)))
    except:
        print('ZeroDivisionError')
        event_recall = 0
    
    try:
        event_precision = np.round((True_positive / True__False_positive), 4)
    except:
        print('ZeroDivisionError')
        event_precision = 0
        
    beta = 1.5
    try:
    
        event_ioc = (1+(beta)**2)*(event_recall*event_precision)/((beta)**2 * event_precision + event_recall)
    except:
        print('ZeroDivisionError')
        event_ioc = 0
        
    accuracy = accuracy_score(inference_output.SIC_next_8h, inference_output.prediction_label)
    precision = precision_score(inference_output.SIC_next_8h, inference_output.prediction_label)
    
    evaluation_score = {'acc':[accuracy],'recall':[event_recall], 'precision':[event_precision], 'raw_precision':[precision], 'IoC(1.5)':[event_ioc]}
    
    
    return pd.DataFrame(evaluation_score, index=[model_name])

def SIC8h_AUPRC(event, inference_output, mode, model_name):
    
    if mode == 'mimic':
        icu_stay = 'stay_id'
    else:
        icu_stay = 'patientunitstayid'
        
    evaluation_interest = inference_output[inference_output[icu_stay].isin(event[icu_stay].unique())]
    
    # 예측 확률과 실제 레이블이 필요
    y_true = evaluation_interest.iloc[:, -3] # 실제 레이블
    y_scores =evaluation_interest.prediction_prob # 예측 확률

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    new_recalls = []
    
    for threshold in thresholds:
        evaluation_interest['prediction_label'] = evaluation_interest['prediction_prob'].apply(lambda x: 1 if x > threshold else 0).astype(int)
        result = event_metric_SIC8h(event, evaluation_interest, mode, model_name)
        recall = result['recall'].values[0]
        new_recalls.append(recall)
    
    auprc = auc(np.array(new_recalls), precisions[:-1])
    
    return auprc
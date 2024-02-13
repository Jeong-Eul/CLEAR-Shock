from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression  
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

from lightgbm import plot_importance
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

import pandas as pd
import numpy as np


def classifier(algorithm, x, y, x_valid, x_test, valid_output, test_output):
    
    if algorithm == 'lgbm':
        lgbm_wrapper = LGBMClassifier(n_estimators=100, random_state=42, extra_trees = True, verbose=-1)
        lgbm_wrapper.fit(x, y)
        
        valid_preds = lgbm_wrapper.predict(x_valid)
        valid_output['prediction_label'] = valid_preds
        valid_output['prediction_prob'] = lgbm_wrapper.predict_proba(x_valid)[:, 1]

        test_preds = lgbm_wrapper.predict(x_test)
        test_output['prediction_label'] = test_preds
        test_output['prediction_prob'] = lgbm_wrapper.predict_proba(x_test)[:, 1]
        
    
        return lgbm_wrapper, valid_output, test_output
    
    elif algorithm == 'xgb':
        xgbm = XGBClassifier(n_estimators=100, random_state=42)
        xgbm.fit(x, y-1)
        
        valid_preds = xgbm.predict(x_valid)
        valid_output['prediction_label'] = valid_preds
        valid_output['prediction_prob'] = xgbm.predict_proba(x_valid)[:, 1]
        valid_output['prediction_label'] = valid_output['prediction_label']+1
        
        
        test_preds = xgbm.predict(x_test)
        test_output['prediction_label'] = test_preds
        test_output['prediction_prob'] = xgbm.predict_proba(x_test)[:, 1]
        test_output['prediction_label'] = test_output['prediction_label']+1
        
        return xgbm, valid_output, test_output
    
    elif algorithm == 'rf':
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(x, y)
        
        valid_preds = rf.predict(x_valid)
        valid_output['prediction_label'] = valid_preds
        valid_output['prediction_prob'] = rf.predict_proba(x_valid)[:, 1]

        test_preds = rf.predict(x_test)
        test_output['prediction_label'] = test_preds
        test_output['prediction_prob'] = rf.predict_proba(x_test)[:, 1]
        
        return rf, valid_output, test_output
    
    elif algorithm == 'dt':
    
        tree = DecisionTreeClassifier(random_state=42).fit(x, y)
        valid_preds = tree.predict(x_valid)
        valid_output['prediction_label'] = valid_preds

        test_preds = tree.predict(x_test)
        test_output['prediction_label'] = test_preds
        
        return tree, valid_output, test_output
    
    elif algorithm == 'lr':
        logit_regression = LogisticRegression(random_state = 42) 
        logit_regression.fit(x, y)

        valid_preds = logit_regression.predict(x_valid)
        valid_output['prediction_label'] = valid_preds
        valid_output['prediction_prob'] = logit_regression.predict_proba(x_valid)[:, 1]

        test_preds = logit_regression.predict(x_test)
        test_output['prediction_label'] = test_preds
        test_output['prediction_prob'] = logit_regression.predict_proba(x_test)[:, 1]
        
        return logit_regression, valid_output, test_output
    
def acc_pr_recall(valid_output_target, test_output_target):

    print('MIMIC')
    print('Accuracy: %.3f' % accuracy_score(valid_output_target.Case, valid_output_target.prediction_label))
    print('Precision: %.3f' % precision_score(valid_output_target.Case, valid_output_target.prediction_label))
    print('Recall: %.3f' % recall_score(valid_output_target.Case, valid_output_target.prediction_label))
    print('f1: %.3f' % f1_score(valid_output_target.Case, valid_output_target.prediction_label))
    print('--')
    print('eICU')
    print('Accuracy: %.3f' % accuracy_score(test_output_target.Case, test_output_target.prediction_label))
    print('Precision: %.3f' % precision_score(test_output_target.Case, test_output_target.prediction_label))
    print('Recall: %.3f' % recall_score(test_output_target.Case, test_output_target.prediction_label))
    print('f1: %.3f' % f1_score(test_output_target.Case, test_output_target.prediction_label))


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
    event_ioc = (1+(beta)**2)*(event_recall*event_precision)/((beta)**2 * event_precision + event_recall)
    accuracy = accuracy_score(inference_output.Case, inference_output.prediction_label)
    precision = precision_score(inference_output.Case, inference_output.prediction_label)
    
    evaluation_score = {'acc':[accuracy],'recall':[event_recall], 'precision':[event_precision], 'raw_precision':[precision], 'IoC(1.5)':[event_ioc]}
    
    
    return pd.DataFrame(evaluation_score, index=[model_name])
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import sys
module_path='/Users/DAHS/Desktop/ECP_CONT/ECP_SCL/Experiment(Supervised Learning)/evaluation/'
if module_path not in sys.path:
    sys.path.append(module_path)
    
import classifier_ML
from imp import reload
reload(classifier_ML)

import warnings

warnings.filterwarnings('ignore')


def create_analysis(eventset, eventset_eicu,X_train, y_train, X_valid, X_test, valid_output, test_output):

    model_names = ['xgb', 'lgbm', 'rf', 'dt']
    result = pd.DataFrame()
    result_e = pd.DataFrame()

    for model_name in model_names:

        model, valid_output, test_output = classifier_ML.classifier(model_name, X_train, y_train, 
                                                            X_valid, X_test, valid_output, test_output)
        
        result_sub = MULT_evaluation(valid_output, eventset, model_name, 'mimic')
        result = pd.concat([result, result_sub], axis = 0).reset_index(drop=True)
        
        result_sub_e = MULT_evaluation(test_output, eventset_eicu, model_name, 'eicu')
        result_e = pd.concat([result_e, result_sub_e], axis = 0).reset_index(drop=True)
        
    print('======================================================================')
    print('|MIMIC-Validation|')
    display(result)
    print('----------------------------------------------------------------------')
    print('|eICU-test|')
    display(result_e)
    print('======================================================================')


def roc_auc_score_multiclass(actual_class, pred_class, average = "macro"):
    
    #creating a set of all the unique classes using the actual class list
    unique_class = set(actual_class)
    roc_auc_dict = {}
    for per_class in unique_class:
        
        #creating a list of all the classes except the current class 
        other_class = [x for x in unique_class if x != per_class]

        #marking the current class as 1 and all other classes as 0
        new_actual_class = [0 if x in other_class else 1 for x in actual_class]
        new_pred_class = [0 if x in other_class else 1 for x in pred_class]

        #using the sklearn metrics method to calculate the roc_auc_score
        roc_auc = roc_auc_score(new_actual_class, new_pred_class, average = average)
        roc_auc_dict[per_class] = roc_auc

    return roc_auc_dict


def MULT_evaluation(inference_output, event, model_name, mode):

    cm = confusion_matrix(inference_output.Case, inference_output.prediction_label)
    accuracy = accuracy_score(inference_output.Case, inference_output.prediction_label)
    roc_auc_dict = roc_auc_score_multiclass(inference_output.Case, inference_output.prediction_label)
    
    event_for = inference_output[inference_output['INDEX']=='CASE1_CASE2_DF']
    event_for['prediction_label'] = event_for['prediction_label'].replace({1:0, 2:1, 3:1, 4:1})
    event_for['Case'] = event_for['Case'].replace({1:0, 2:1, 3:1, 4:1})
    
    result = classifier_ML.event_metric(event, event_for, mode, model_name)
    case2_recall_event = result['recall'].values[0]
    
    case1_tp = cm[0][0]
    case1_fp = cm[0][1] + cm[2][0] + cm[3][0]
    case1_fn = cm[0][1] + cm[0][2] + cm[0][3]
    case1_tn = cm[1][1]

    case2_tp = cm[1][1]
    case2_fp = cm[0][1] + cm[2][1] + cm[3][1]
    case2_fn = cm[1][0] + cm[1][2] + cm[1][3]
    case2_tn = cm[0][0]

    case3_tp = cm[2][2]
    case3_fp = cm[0][2] + cm[1][2] + cm[3][2]
    case3_fn = cm[2][0] + cm[2][1] + cm[2][3]
    case3_tn = cm[3][3]

    case4_tp = cm[3][3]
    case4_fp = cm[0][3] + cm[1][3] + cm[2][3]
    case4_fn = cm[3][0] + cm[3][1] + cm[3][2]
    case4_tn = cm[2][2]

    case1_recall = case1_tp / (case1_tp + case1_fn)
    case1_precision = case1_tp / (case1_tp + case1_fp)
    case1_score = np.round(2*case1_recall*case1_precision / (case1_precision + case1_recall), 4)
    case1_aucroc = np.round(roc_auc_dict[1], 4)

    case2_recall = case2_tp / (case2_tp + case2_fn)
    case2_precision = case2_tp / (case2_tp + case2_fp)
    beta = 2
    case2_score = np.round(case2_recall + (1+(beta)**2)*(case2_recall_event*case2_precision)/((beta)**2 * case2_precision + case2_recall_event), 4)
    case2_aucroc = np.round(roc_auc_dict[2], 4)

    case3_recall = case3_tp / (case3_tp + case3_fn)
    case3_precision = case3_tp / (case3_tp + case3_fp)
    case3_score = np.round(2*case3_recall*case3_precision / (case3_precision + case3_recall), 4)
    case3_aucroc = np.round(roc_auc_dict[3], 4)

    case4_recall = case4_tp / (case4_tp + case4_fn)
    case4_precision = case4_tp / (case4_tp + case4_fp)
 
    case4_score = np.round(2*case4_recall*case4_precision / (case4_precision + case4_recall), 4)
    case4_aucroc = np.round(roc_auc_dict[4], 4)



    # data = {'Model':[model_name], 'Accuracy': [accuracy],'Case 1 Score': [case1_score]
    #         ,'Case 2 Score': [case2_score], 'Case 3 Score': [case3_score],'Case 3 Score': [case3_score], 'Case 4 Score': [case4_score],
    #         'Case 1 AUROC': [case1_aucroc],'Case 2 AUROC': [case2_aucroc],'Case 3 AUROC': [case3_aucroc], 'Case 4 AUROC': [case4_aucroc],}
    
    data = {'Model':[model_name], 'Accuracy': [accuracy],'Case 1 Score': [case1_score]
            ,'Case 2 Score': [case2_score], 'Case 3 Score': [case3_score],'Case 3 Score': [case3_score], 'Case 4 Score': [case4_score]}


    return pd.DataFrame(data).fillna(0)



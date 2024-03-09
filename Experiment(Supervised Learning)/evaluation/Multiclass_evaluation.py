from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import average_precision_score

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

import pandas as pd
import numpy as np
import sys
module_path='/Users/DAHS/Desktop/ECP_CONT/ECP_SCL/Experiment(Supervised Learning)/evaluation/'
if module_path not in sys.path:
    sys.path.append(module_path)
    
import classifier_ML
from imp import reload
reload(classifier_ML)

import split
from imp import reload
reload(split)

import warnings

warnings.filterwarnings('ignore')


def create_analysis(eventset,X_train, y_train, X_valid, valid_output, mode):

    model_names = ['lgbm', 'rf', 'dt', 'lr']
    trained_models = []
    result = pd.DataFrame()
    
    for model_name in model_names:

        model, valid_output = classifier_ML.classifier(model_name, X_train, y_train, X_valid, valid_output, mode = mode)
        
        result_sub = MULT_evaluation(valid_output, eventset, model_name, 'mimic')
        
        result = pd.concat([result, result_sub], axis = 0).reset_index(drop=True)
        trained_models.append(model)
    print('|MIMIC-Validation|====================================================')
    display(result)
    print('----------------------------------------------------------------------')
    
    return trained_models, result

def create_subtask(X_train, y_train, X_valid, valid_output, event, mode, type, event_task):
    
    model_names = ['lgbm', 'rf', 'dt', 'lr']
    trained_models = []
    result = pd.DataFrame()
    
    for model_name in model_names:

        model, valid_output = classifier_ML.classifier(model_name, X_train, y_train, X_valid, valid_output, mode = mode)
        
        if type == 'binary':
            result_sub = ST_binary_evaluation(event, valid_output, model_name, mode, event_task)
            result = pd.concat([result, result_sub], axis = 0).reset_index(drop=True)
            trained_models.append(model)
            
        else:
            result_sub = ST_multi_evaluation(event, valid_output, model_name)
            result = pd.concat([result, result_sub], axis = 0).reset_index(drop=True)
            trained_models.append(model)
    print('|MIMIC-Validation|====================================================')
    display(result)
    print('----------------------------------------------------------------------')
    
    return trained_models, result

def create_threshold(eventset,X_train, y_train, X_valid, valid_output, mode):
    
    model_names = ['lgbm', 'rf', 'dt', 'lr']
    trained_models = []
    result = pd.DataFrame()
    
    for model_name in model_names:

        model, valid_output = classifier_ML.classifier(model_name, X_train, y_train, X_valid, valid_output, mode = mode)
        
        result_sub = threshold_for_capture(valid_output, eventset, model_name, 'mimic')
        result_sub['Model'] = model_name
        result = pd.concat([result, result_sub], axis = 0).reset_index(drop=True)
        trained_models.append(model)
    print('|Threshold for MIMIC-Valid|====================================================')
    print('----------------------------------------------------------------------')
    
    return result




def external_evaluation(unitadmitsource, unittype, unitstaytype, model, model_name, eicu_event):
    print('[Starting eICU-Test]')
    
    result = pd.DataFrame()
    drop_col = ['hospitalid', 'hospitaldischargeyear']
    """
    
    |Subpopulation|eICU Type|Accuracy|Case 1 AUROC|Case 1 AUPRC|Case 2 AUROC|Case 2 AUPRC|Case 3 AUROC|Case 3 AUPRC|Case 4 AUROC|Case 4 AUPRC|Case 1 Score|Case 2 Score|Case 3 Score|Case 4 Score|
    
    """
    print('eICU-UnitAdmitSource...')
    
    for type in unitadmitsource.keys():
        try:
            eval_set = unitadmitsource[type].drop(drop_col, axis = 1)
            eval_event = eicu_event[eicu_event['patientunitstayid'].isin(eval_set.patientunitstayid.unique())]
            X_test, _, test_output = split.split_X_Y(eval_set, mode = 'eicu')
            test_preds = model.predict(X_test)
            
            test_output['prediction_label'] = test_preds
            test_output['prediction_prob'] = model.predict_proba(X_test)[:, 1]
            
            result_sub = MULT_evaluation(test_output, eval_event, model_name, 'eicu')
            result_sub['Subpopulation'] = 'UnitAdmitSource'
            result_sub['eICU Type'] = type
            result = pd.concat([result, result_sub], axis = 0).reset_index(drop=True)
        except:
            pass
    
        
    print('eICU-UnitType...')
    
    for type in unittype.keys():
        try:
            eval_set = unittype[type].drop(drop_col, axis = 1)
            X_test, _, test_output = split.split_X_Y(eval_set, mode = 'eicu')
            test_preds = model.predict(X_test)
            
            test_output['prediction_label'] = test_preds
            test_output['prediction_prob'] = model.predict_proba(X_test)[:, 1]
            
            result_sub = MULT_evaluation(test_output, eicu_event, model_name, 'eicu')
            result_sub['Subpopulation'] = 'UnitType'
            result_sub['eICU Type'] = type
            result = pd.concat([result, result_sub], axis = 0).reset_index(drop=True)
        except:
            pass
        
        
    print('eICU-UnitStayType...')
    
    for type in unitstaytype.keys():
        try:
            eval_set = unitstaytype[type].drop(drop_col, axis = 1)
            X_test, _, test_output = split.split_X_Y(eval_set, mode = 'eicu')
            test_preds = model.predict(X_test)
            
            test_output['prediction_label'] = test_preds
            test_output['prediction_prob'] = model.predict_proba(X_test)[:, 1]
            
            result_sub = MULT_evaluation(test_output, eicu_event, model_name, 'eicu')
            result_sub['Subpopulation'] = 'UnitStayType'
            result_sub['eICU Type'] = type
            result = pd.concat([result, result_sub], axis = 0).reset_index(drop=True)
            
        except:
            pass
        
    
    display(result.set_index(['Subpopulation', 'eICU Type']))
    
    return result.set_index(['Subpopulation', 'eICU Type'])
  


def external_validation_hosp_RF(model, eicu, eicu_event, hosp_id):
    print('|eICU-Test|===========================================================')
    drop_col = ['hospitalid', 'hospitaldischargeyear']
    result = pd.DataFrame()
    for hospid in hosp_id:
        
        model_name = 'rf'
        testset = eicu[(eicu['hospitalid']==hospid)]
        event_set = eicu_event[(eicu_event['hospitalid']==hospid)]
        
        testset.drop(drop_col, axis = 1, inplace = True)
        
        X_test, y_test, test_output = split.split_X_Y(testset, mode = 'eicu')
        
        test_preds = model.predict(X_test)
        test_output['prediction_label'] = test_preds
        test_output['prediction_prob'] = model.predict_proba(X_test)[:, 1]
        try:
            result_sub = MULT_evaluation(test_output, event_set, model_name, 'eicu')

        except:
            print(hospid)
        result_sub['hospital_id'] = hospid
        result = pd.concat([result, result_sub], axis = 0).reset_index(drop=True)
    if len(result)>0:
        
        display(result)
    else:
        pass
    
def external_validation_hosp_RF_riskscore(model, eicu, eicu_event, hosp_id):
    print('|eICU-Test|===========================================================')
    drop_col = ['hospitalid', 'hospitaldischargeyear']
    result = pd.DataFrame()
    for hospid in hosp_id:
        
        model_name = 'rf'
        testset = eicu[(eicu['hospitalid']==hospid)]
        event_set = eicu_event[(eicu_event['hospitalid']==hospid)]
        
        testset.drop(drop_col, axis = 1, inplace = True)
        
        X_test, y_test, test_output = split.split_X_Y(testset, mode = 'eicu')
        
        test_preds = model.predict(X_test)
        test_output['prediction_label'] = test_preds
        test_output['prediction_prob'] = model.predict_proba(X_test)[:, 1]
        test_output['hospital_id'] = hospid
        result = pd.concat([result, test_output], axis = 0).reset_index(drop=True)
        
    return result


    
def external_validation_hosp_lgbm(model, eicu, eicu_event, hosp_id):
    print('|eICU-Test|===========================================================')
    drop_col = ['hospitalid', 'hospitaldischargeyear']
    result = pd.DataFrame()
    for hospid in hosp_id:
        
        model_name = 'lgbm'
        testset = eicu[(eicu['hospitalid']==hospid)]
        event_set = eicu_event[(eicu_event['hospitalid']==hospid)]
        
        testset.drop(drop_col, axis = 1, inplace = True)
        
        X_test, y_test, test_output = split.split_X_Y(testset, mode = 'eicu')
        
        test_preds = model.predict(X_test)
        test_output['prediction_label'] = test_preds
        test_output['prediction_prob'] = model.predict_proba(X_test)[:, 1]
        try:
            result_sub = MULT_evaluation(test_output, event_set, model_name, 'eicu')

        except:
            print(hospid)
        result_sub['hospital_id'] = hospid
        result = pd.concat([result, result_sub], axis = 0).reset_index(drop=True)
    if len(result)>0:
        
        display(result)
    else:
        pass
            
    
        
    print('----------------------------------------------------------------------')   
    
def external_validation_hosp(models, eicu, eicu_event, hosp_id):
    print('|eICU-Test|===========================================================')
    drop_col = ['hospitalid', 'hospitaldischargeyear']
    
    model_names = ['lgbm', 'rf', 'dt']
    for hospid in hosp_id:
        result = pd.DataFrame()
        for idx, model in enumerate(models):
            model_name = model_names[idx]
            testset = eicu[(eicu['hospitalid']==hospid)]
            event_set = eicu_event[(eicu_event['hospitalid']==hospid)]
            
            testset.drop(drop_col, axis = 1, inplace = True)
            
            X_test, y_test, test_output = split.split_X_Y(testset, mode = 'eicu')
            
            if model_name == 'xgb':
                test_preds = model.predict(X_test)
                test_output['prediction_label'] = test_preds
                test_output['prediction_prob'] = model.predict_proba(X_test)[:, 1]
                test_output['prediction_label'] = test_output['prediction_label']+1
            
            else:
                test_preds = model.predict(X_test)
                test_output['prediction_label'] = test_preds
                test_output['prediction_prob'] = model.predict_proba(X_test)[:, 1]

            result_sub = MULT_evaluation(test_output, event_set, model_name, 'eicu')
            result = pd.concat([result, result_sub], axis = 0).reset_index(drop=True)
        if len(result)>0:
            result['hospital_id'] = hospid
            display(result)
        else:
            pass
            
    
        
    print('----------------------------------------------------------------------')    

    return result

def external_validation(models, eicu, eicu_event, hosp_id):
    print('|eICU-Test|===========================================================')
    drop_col = ['hospitalid', 'hospitaldischargeyear']
    
    eicu = eicu[eicu['hospitalid'].isin(hosp_id)]
    eicu_event = eicu_event[eicu_event['hospitalid'].isin(hosp_id)]
    
    model_names = ['lgbm', 'rf', 'dt']
    interestset = eicu[(eicu['hospitaldischargeyear']==2015)|(eicu['hospitaldischargeyear']==2014)]
    
    result = pd.DataFrame()
    for idx, model in enumerate(models):
        model_name = model_names[idx]
        test = interestset.drop(drop_col, axis = 1)

        X_test, y_test, test_output = split.split_X_Y(test, mode = 'eicu')
        
        if model_name == 'xgb':
            test_preds = model.predict(X_test)
            test_output['prediction_label'] = test_preds
            test_output['prediction_prob'] = model.predict_proba(X_test)[:, 1]
            test_output['prediction_label'] = test_output['prediction_label']+1
        
        else:
            test_preds = model.predict(X_test)
            test_output['prediction_label'] = test_preds
            test_output['prediction_prob'] = model.predict_proba(X_test)[:, 1]

        result_sub = MULT_evaluation(test_output, eicu_event, model_name, 'eicu')
        result = pd.concat([result, result_sub], axis = 0).reset_index(drop=True)
 
    display(result)
 
    print('----------------------------------------------------------------------')    

    
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

def auprc_score_multiclass(actual_class, pred_class, average="macro"):
    
    unique_class = set(actual_class)
    auprc_dict = {}
    for per_class in unique_class:
        
        other_class = [x for x in unique_class if x != per_class]
        
        # Marking the current class as 1 and all other classes as 0
        new_actual_class = [1 if x == per_class else 0 for x in actual_class]
        new_pred_class = [1 if x == per_class else 0 for x in pred_class]
        
        # Calculating precision-recall curve
        precision, recall, _ = precision_recall_curve(new_actual_class, new_pred_class)
        
        # Calculating AUC for precision-recall curve
        auprc = auc(recall, precision)
        auprc_dict[per_class] = auprc
    
    return auprc_dict

def MULT_evaluation(inference_output, event, model_name, mode):
    
    accuracy = accuracy_score(inference_output.Case, inference_output.prediction_label)
    roc_auc_dict = roc_auc_score_multiclass(inference_output.Case, inference_output.prediction_label)
    prc_auc_dict = auprc_score_multiclass(inference_output.Case, inference_output.prediction_label)
    report = classification_report(inference_output.Case, inference_output.prediction_label, output_dict=True)
    report_int_keys = {int(float(k)): v for k, v in report.items() if k not in ['accuracy', 'macro avg', 'weighted avg']}
    event_for = inference_output[inference_output['INDEX']=='CASE1_CASE2_DF']
    event_for['prediction_label'] = event_for['prediction_label'].replace({1:0, 2:1, 3:1, 4:1})
    event_for['Case'] = event_for['Case'].replace({1:0, 2:1, 3:1, 4:1})
    
    result = classifier_ML.event_metric(event, event_for, mode, model_name)
    case2_recall_event = result['recall'].values[0]
    

    if 1 in inference_output.Case.values:
        try:
            case1_score = np.round(report_int_keys[1]['f1-score'], 4)
        except:
            print(report)
        case1_aucroc = np.round(roc_auc_dict[1], 4)
        case1_auprc = np.round(prc_auc_dict[1], 4)
        
    else:

        case1_score = '-'
        case1_aucroc = '-'
        case1_auprc = '-'
        
    if 2 in inference_output.Case.values:
        beta = 2
        case2_recall = np.round(report_int_keys[2]['recall'], 4)
        case2_precision = np.round(report_int_keys[2]['precision'], 4)
        case2_score = np.round((1+(beta)**2)*(case2_recall_event*case2_precision)/((beta)**2 * case2_precision + case2_recall_event), 4)
        case2_aucroc = np.round(roc_auc_dict[2], 4)
        case2_auprc = np.round(prc_auc_dict[2], 4)
        
    else:
        case2_recall = '-'
        case2_precision = '-'
        case2_score = '-'
        case2_aucroc = '-'
        case2_auprc = '-'

    if 3 in inference_output.Case.values:

        case3_score = np.round(report_int_keys[3]['f1-score'], 4)
        case3_aucroc = np.round(roc_auc_dict[3], 4)
        case3_auprc = np.round(prc_auc_dict[3], 4)
        
    else:
       
        case3_score = '-'
        case3_aucroc = '-'
        case3_auprc = '-'

    if 4 in inference_output.Case.values:

        case4_score = np.round(report_int_keys[4]['f1-score'], 4)
        case4_aucroc = np.round(roc_auc_dict[4], 4)
        case4_auprc = np.round(prc_auc_dict[4], 4)
        
    else:
 
        case4_score = '-'
        case4_aucroc = '-'
        case4_auprc = '-'
        
    
    data = {'Model':[model_name], 'Accuracy': [accuracy],
            'Case 1 AUROC': [case1_aucroc],'Case 1 AUPRC': [case1_auprc],'Case 2 AUROC': [case2_aucroc],'Case 2 AUPRC': [case2_auprc],
            'Case 3 AUROC': [case3_aucroc],'Case 3 AUPRC': [case3_auprc],'Case 4 AUROC': [case4_aucroc],'Case 4 AUPRC': [case4_auprc],
            'Case 1 Score': [case1_score],'Case 2 Score': [case2_score], 'Case 3 Score': [case3_score],'Case 3 Score': [case3_score], 
            'Case 4 Score': [case4_score]}

    return pd.DataFrame(data).fillna(0)


def threshold_for_capture(inference_output, event, model_name, mode):
    
    set = inference_output[inference_output['INDEX']=='CASE1_CASE2_DF']
    probas = set.prediction_prob
    thresholds = np.linspace(0, 1, 50)
    results = pd.DataFrame(columns=['Threshold', 'Recall', 'Precision', 'Event Recall'])
    
    for threshold in thresholds:
        predictions = np.where(probas >= threshold, 2, 1)    
            
        event_for = inference_output[inference_output['INDEX']=='CASE1_CASE2_DF']
        event_for['prediction_label'] = predictions
        event_for['prediction_label'] = event_for['prediction_label'].replace({1:0, 2:1, 3:1, 4:1})
        event_for['Case'] = event_for['Case'].replace({1:0, 2:1, 3:1, 4:1})
        
        result = classifier_ML.event_metric(event, event_for, mode, model_name)
        case2_recall_event = result['recall'].values[0]
        
        if 2 in inference_output.Case.values:
            
            case2_recall = recall_score(set.Case, predictions, pos_label = 2)
            case2_precision = precision_score(set.Case, predictions, pos_label = 2)
            
            
            results = results.append({'Threshold': threshold, 'Recall': case2_recall,
                              'Precision': case2_precision, 'Event Recall': case2_recall_event}, ignore_index=True)
            
    case2_score = np.round(result['Recall'] + (1+(2)**2)*(result['Event Recall']*result['Precision'])/((2)**2 * result['Precision'] + result['Event Recall']), 4)
    result['Case 2 Score'] = case2_score
    
    return results

def ST_binary_evaluation(event, inference_output, model_name, mode, event_task):
    
    if event_task == False:
    
        accuracy = accuracy_score(inference_output.iloc[:, -3], inference_output.prediction_label.astype(int))
        auprc = average_precision_score(inference_output.iloc[:, -3], inference_output.prediction_prob)
        f1 = f1_score(inference_output.iloc[:, -3], inference_output.prediction_label.astype(int))
        recall = recall_score(inference_output.iloc[:, -3], inference_output.prediction_label.astype(int))
        precision = precision_score(inference_output.iloc[:, -3], inference_output.prediction_label.astype(int))
        
        data = {'Model':[model_name], 'Accuracy': [accuracy],
                'AUPRC': [auprc], 
                'F1 Score': [f1],
                'Recall': [recall],
                'Precision': [precision]}

        return pd.DataFrame(data).fillna(0)
    
    elif event_task == 'ARDS 4h':
        accuracy = accuracy_score(inference_output.iloc[:, -3], inference_output.prediction_label.astype(int))
        result = classifier_ML.event_metric_ARDS4h(event, inference_output, mode, model_name)
        fb_score = result['IoC(1.5)'].values[0]
        auprc = average_precision_score(inference_output.iloc[:, -3], inference_output.prediction_prob)
        recall = result['recall'].values[0]
        precision = precision_score(inference_output.iloc[:, -3], inference_output.prediction_label.astype(int))
        # auprc = average_precision_score(inference_output.iloc[:, -3], inference_output.prediction_prob)
        auprc = classifier_ML.ARDS4h_AUPRC(event, inference_output, mode, model_name)
        
        data = {'Model':[model_name],
                'AUPRC': [auprc],
                'ARDS score':[fb_score],
                'Recall': [recall],
                'Precision': [precision]}

        return pd.DataFrame(data).fillna(0)
        
    elif event_task == 'ARDS 8h':
        accuracy = accuracy_score(inference_output.iloc[:, -3], inference_output.prediction_label.astype(int))
        result = classifier_ML.event_metric_ARDS8h(event, inference_output, mode, model_name)
        
        general_recall = recall_score(inference_output.iloc[:, -3], inference_output.prediction_label.astype(int))
        recall = result['recall'].values[0]
        precision = precision_score(inference_output.iloc[:, -3], inference_output.prediction_label.astype(int))
        # auprc = classifier_ML.ARDS8h_AUPRC(event, inference_output, mode, model_name)
        
        fb_score =  (1+(1.5)**2)*(general_recall*precision)/((1.5)**2 * precision + general_recall)
        precisions, recalls, thresholds = precision_recall_curve(inference_output.iloc[:, -3], inference_output.prediction_prob)
        auprc = auc(recalls, precisions)
        
        data = {'Model':[model_name],
                'AUPRC': [auprc],
                'ARDS score':[fb_score],
                'Recall': [recall],
                'Precision': [precision],
                'Grecall': [general_recall]}

        return pd.DataFrame(data).fillna(0)
    
    elif event_task == 'SIC 4h':
        accuracy = accuracy_score(inference_output.iloc[:, -3], inference_output.prediction_label.astype(int))
        result = classifier_ML.event_metric_SIC4h(event, inference_output, mode, model_name)
        general_recall = recall_score(inference_output.iloc[:, -3], inference_output.prediction_label.astype(int))
        fb_score = result['IoC(1.5)'].values[0]
        auprc = average_precision_score(inference_output.iloc[:, -3], inference_output.prediction_prob)
        recall = result['recall'].values[0]
        precision = precision_score(inference_output.iloc[:, -3], inference_output.prediction_label.astype(int))
        auprc = classifier_ML.SIC4h_AUPRC(event, inference_output, mode, model_name)
        
        data = {'Model':[model_name],
                'AUPRC': [auprc],
                'SIC score':[fb_score],
                'Recall': [recall],
                'Precision': [precision],
                'Grecall': [general_recall]}

        return pd.DataFrame(data).fillna(0)
        
    elif event_task == 'SIC 8h':
        accuracy = accuracy_score(inference_output.iloc[:, -3], inference_output.prediction_label.astype(int))
        general_recall = recall_score(inference_output.iloc[:, -3], inference_output.prediction_label.astype(int))
        result = classifier_ML.event_metric_SIC8h(event, inference_output, mode, model_name)
        auprc = average_precision_score(inference_output.iloc[:, -3], inference_output.prediction_prob)
        recall = result['recall'].values[0]
        precision = precision_score(inference_output.iloc[:, -3], inference_output.prediction_label.astype(int))
        # auprc = classifier_ML.SIC8h_AUPRC(event, inference_output, mode, model_name)
        precisions, recalls, thresholds = precision_recall_curve(inference_output.iloc[:, -3], inference_output.prediction_prob)
        auprc = auc(recalls, precisions)
        
        fb_score =  (1+(1.5)**2)*(general_recall*precision)/((1.5)**2 * precision + general_recall)
        
        data = {'Model':[model_name],
                'AUPRC': [auprc],
                'SIC score':[fb_score],
                'Recall': [recall],
                'Precision': [precision],
                'Grecall': [general_recall],}

        return pd.DataFrame(data).fillna(0)
        

def ST_multi_evaluation(event, inference_output, model_name):
    
    report = classification_report(inference_output.iloc[:, -3], inference_output.prediction_label.astype(int), output_dict=True)
 
    f1_score = np.round(report['macro avg']['f1-score'], 4)
    macro_precision = np.round(report['macro avg']['precision'], 4)
    macro_recall = np.round(report['macro avg']['recall'], 4)
    accuracy = np.round(report['accuracy'], 4)
    prc_auc_dict = auprc_score_multiclass(inference_output.iloc[:, -3], inference_output.prediction_prob)
    average_prc = sum(prc_auc_dict.values()) / len(prc_auc_dict)
    
    data = {'Model':[model_name], 'Accuracy': [accuracy],
            'Macro AUPRC':[average_prc],
            'F1 Score': [f1_score],
            'Macro averaged precision': [macro_precision],
            'Macro averaged Recall': [macro_recall]}

    return pd.DataFrame(data).fillna(0)
    
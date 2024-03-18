import pandas as pd
import numpy as np

def matching_patient(eicu):
    
    data = eicu.copy()
    
    # eicu_patient_dir = '/Users/DAHS/Desktop/early_prediction_of_circ_scl/eicu-crd/2.0/patient.csv.gz'
    eicu_patient_dir = '/Users/gwonjeong-eul/Desktop/ecp-scl-macbook/eicu_patients/patient.csv.gz'
    eicu_patient = pd.read_csv(eicu_patient_dir, compression = 'gzip', usecols=['patientunitstayid', 'hospitaldischargeyear', 'hospitalid'])

    result = pd.merge(data, eicu_patient, how = 'left', on ='patientunitstayid')
    
    return result

# %% 
# print('hello')
# %%

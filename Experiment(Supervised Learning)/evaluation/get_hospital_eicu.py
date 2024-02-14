import pandas as pd
import numpy as np

def hospital(eicu):
    
    sample = eicu[['patientunitstayid', 'hospitalid', 'hospitaldischargeyear']].groupby('hospitalid').agg('nunique').reset_index()
    hospital = sample[sample['patientunitstayid']>=10]
    df_target = eicu[eicu['hospitalid'].isin(hospital['hospitalid'])]
    
    return hospital['hospitalid'].values, df_target
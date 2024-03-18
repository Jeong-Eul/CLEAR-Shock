import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE

# X_train, Y_train

def Synthetic_Minority_Oversampling_Tech(X_train, y_train):
    
    print('-'*20)
    print('|Start Oversampling-SMOTE|')
    print("Before OverSampling, counts of case '1': {}".format(sum(y_train==1)))
    print("Before OverSampling, counts of case '2': {}".format(sum(y_train==2)))
    print("Before OverSampling, counts of case '3': {}".format(sum(y_train==3)))
    print("Before OverSampling, counts of case '4': {}".format(sum(y_train==4)))

    sm = SMOTE(random_state = 42)
    X_resampled, y_resampled = sm.fit_resample(X_train,list(y_train))


    print('After OverSampling, the shape of train_X: {}'.format(X_resampled.shape))

    print("After OverSampling, counts of case '1': {}".format(sum(np.array(y_resampled)==1)))
    print("After OverSampling, counts of case '2': {}".format(sum(np.array(y_resampled)==2)))
    print("After OverSampling, counts of case '3': {}".format(sum(np.array(y_resampled)==3)))
    print("After OverSampling, counts of case '4': {}".format(sum(np.array(y_resampled)==4)))
    
    return X_resampled, np.array(y_resampled)
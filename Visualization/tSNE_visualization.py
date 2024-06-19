import pandas as pd
import numpy as np
import sys
from sklearn.linear_model import LogisticRegression  
from sklearn.manifold import TSNE

module_path='/Users/DAHS/Desktop/ECP_CONT/ECP_SCL/Experiment/evaluation/'
if module_path not in sys.path:
    sys.path.append(module_path)
    
from integrating_embedding import *

import integrating_embedding
from imp import reload
reload(integrating_embedding)

module_path='/Users/DAHS/Desktop/ECP_CONT/ECP_SCL/Experiment/evaluation/'
if module_path not in sys.path:
    sys.path.append(module_path)

import matplotlib.pyplot as plt

emb_path_trn_mimic = '/Users/DAHS/Desktop/ECP_CONT/ECP_SCL/Training/Train/result/emb_train_new_version.npy'
emb_path_vld_mimic = '/Users/DAHS/Desktop/ECP_CONT/ECP_SCL/Training/Train/result/emb_valid_new_version.npy'

emb_path_trn_eicu = '/Users/DAHS/Desktop/ECP_CONT/ECP_SCL/Training/Train/result/emb_eicu_new_version.npy'

mimic_path = '/Users/DAHS/Desktop/ECP_CONT/ECP_SCL/Case Labeling/mimic_analysis.csv.gz'
eicu_path = '/Users/DAHS/Desktop/ECP_CONT/ECP_SCL/Case Labeling/eicu_analysis.csv.gz'

mimic_train_emb, mimic_valid_emb, event = integrating_embedding.integrating(mimic_path, emb_path_trn_mimic, emb_path_vld_mimic, _, 'mimic')


original = mimic_train_emb[mimic_train_emb.columns[:235]]
class_feat = original['Case'].values

n_components = 2
model = TSNE(n_components=n_components,perplexity=30.0, random_state=42)
X_embedded = model.fit_transform(original.drop(['INDEX', 'Shock_next_8h', 'Case', 'Annotation','after_shock_annotation', 'Unnamed: 0'], axis = 1))

tsne_ori_feat_df = pd.DataFrame(X_embedded, columns = ['component 0', 'component 1'])
tsne_ori_feat_df['Case'] = class_feat

#Original space

plt.rcParams["figure.figsize"] = (12,12)

ori_df_1 = tsne_ori_feat_df[tsne_ori_feat_df['Case'] == 1]
ori_df_2 = tsne_ori_feat_df[tsne_ori_feat_df['Case'] == 2]
ori_df_3 = tsne_ori_feat_df[tsne_ori_feat_df['Case'] == 3]
ori_df_4 = tsne_ori_feat_df[tsne_ori_feat_df['Case'] == 4]

plt.scatter(ori_df_1['component 0'], ori_df_1['component 1'], color = 'forestgreen', label = 'Case 1', s = 0.5, alpha=0.1)
plt.scatter(ori_df_2['component 0'], ori_df_2['component 1'], color = 'sandybrown', label = 'Case 2', s = 0.5, alpha=0.1)
plt.scatter(ori_df_3['component 0'], ori_df_3['component 1'], color = 'saddlebrown', label = 'Case 3', s = 0.5, alpha=0.1)
plt.scatter(ori_df_4['component 0'], ori_df_4['component 1'], color = 'tomato', label = 'Case 4', s = 0.5, alpha=0.1)

plt.legend(fontsize=12, handlelength=3)
plt.savefig('OriginalSpcae.png')
plt.close()


total = mimic_train_emb[mimic_train_emb.columns[235:]]
class_feat = mimic_train_emb['Case'].values

n_components = 2
model = TSNE(n_components=n_components,perplexity=30.0, random_state=42)
X_embedded = model.fit_transform(total)

tsne_emb_feat_df_t = pd.DataFrame(X_embedded, columns = ['component 0', 'component 1'])
tsne_emb_feat_df_t['Case'] = class_feat

plt.rcParams["figure.figsize"] = (12,12)

c_tsne_df_1 = tsne_emb_feat_df_t[tsne_emb_feat_df_t['Case'] == 1]
c_tsne_df_2 = tsne_emb_feat_df_t[tsne_emb_feat_df_t['Case'] == 2]
c_tsne_df_3 = tsne_emb_feat_df_t[tsne_emb_feat_df_t['Case'] == 3]
c_tsne_df_4 = tsne_emb_feat_df_t[tsne_emb_feat_df_t['Case'] == 4]

plt.scatter(c_tsne_df_1['component 0'], c_tsne_df_1['component 1'], color = 'forestgreen', label = 'Case 1',  s = 0.5, alpha=0.2)
plt.scatter(c_tsne_df_2['component 0'], c_tsne_df_2['component 1'], color = 'sandybrown', label = 'Case 2',  s = 0.5, alpha=0.2)
plt.scatter(c_tsne_df_3['component 0'], c_tsne_df_3['component 1'], color = 'saddlebrown', label = 'Case 3',  s = 0.5, alpha=0.2)
plt.scatter(c_tsne_df_4['component 0'], c_tsne_df_4['component 1'], color = 'tomato', label = 'Case 4',  s = 0.5, alpha=0.2)


plt.legend(fontsize=12, handlelength=3)
plt.savefig('ContrastiveSpace.png')
plt.close()
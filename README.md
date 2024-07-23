# CLEAR-Shock: Contrastive LEARning for Shock  
(Submitted)

<center><img src="https://github.com/Jeong-Eul/CLEAR/blob/main/Figure/fig6.png" width="800" height="500"/>

[Derived Variables]  
- Derived_Variable.py: Lactate Clearance, Cumulative sum of vasopressor use amount, Percentage change in measurement  

[Case Labeling]  
- Case.py: Annotation, Labeling, Preprocessing for Analysis.  
- main_case.py: The final analysis datasets (both mimic and eicu) are saved in csv format through this file.  

[Training]  
- main.py: Training(Optuna), Get Embedding, Get Feature Importance from Self Attention map.  
- main.sh: Script that can run the above three modes.  

[Experiment]  
This part includes case predicion, and monitoring system visualization, as shown in the Discussion section.  
- Baseline.py: Baseline experiment(Internal(or External) validation(classification performance, Naive system accuracy, Avergaed cosine index)).  
- CLEAR.py: Proposed method experiment(Internal(or External) validation(classification performance, Naive system accuracy, Avergaed cosine index)), Prognositc Monitoring system working process visualization.  

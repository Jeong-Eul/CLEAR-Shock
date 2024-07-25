import argparse
import os
import sys
import time
import numpy as np
import torch
import random
import wandb
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score ,f1_score, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from multiprocessing import freeze_support
import matplotlib.pyplot as plt


module_path='/Users/DAHS/Desktop/ECP_CONT/ECP_SCL/Training'
if module_path not in sys.path:
    sys.path.append(module_path)

from cohort_loader import *

from model import *
from pytorch_metric_learning import losses
import torch.optim as optim

import gc

warnings.filterwarnings("ignore")
import optuna
from optuna.trial import TrialState

if __name__ == '__main__':
    freeze_support()

    parser = argparse.ArgumentParser(description="Train the FT-Transformer for supervised contrastive learning", 
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Data
    parser.add_argument("--mimic_data_dir", default="/Users/DAHS/Desktop/ECP_CONT/ECP_SCL/Case Labeling/mimic_analysis.csv.gz", type=str, dest="mimic_data_dir")
    parser.add_argument("--eicu_data_dir", default="/Users/DAHS/Desktop/ECP_CONT/ECP_SCL/Case Labeling/eicu_analysis.csv.gz", type=str, dest="eicu_data_dir")
    parser.add_argument('--seed', default=9040, type=int , dest='seed')

    # Train Method
    parser.add_argument('--optimizer', default='AdamW', type=str, dest='optim')
    parser.add_argument("--lr", default=1e-5, type=float,dest="lr")
    parser.add_argument("--batch_size", default=64, type=int, dest="batch_size")
    parser.add_argument("--num_epoch", default=300, type=int, dest="num_epoch")

    # Model
    parser.add_argument("--num_cont", default=61, type=int, dest="num_cont", help = "Nums of Continuous Features")
    parser.add_argument("--num_cat", default=57, type=int, dest="num_cat", help = "Nums of Categorical Features But Not Use")
    parser.add_argument("--dim", default=32, type=int, dest="dim", help = "Embedding Dimension of Input Data ")
    parser.add_argument("--dim_head", default=16, type=int, dest="dim_head", help = "Dimension of Attention(Q,K,V)")
    parser.add_argument("--depth", default=6, type=int, dest="depth", help = "Nums of Attention Layer Depth")
    parser.add_argument("--heads", default=8, type=int, dest="heads", help='Nums of Attention head')
    parser.add_argument("--attn_dropout", default=0.1, type=float, dest="attn_dropout", help='Ratio of Attention Layer dropout')
    parser.add_argument("--ff_dropout", default=0.1, type=float, dest="ff_dropout", help='Ratio of FeedForward Layer dropout')
    parser.add_argument("--patience", default=4, type=float, dest="patience", help='Traininig step adjustment')

    # Others
    parser.add_argument("--ckpt_dir", default="./checkpoint", type=str, dest="ckpt_dir")
    parser.add_argument("--log_dir", default="./log", type=str, dest="log_dir")
    parser.add_argument("--result_dir", default="./result", type=str, dest="result_dir")
    parser.add_argument("--printiter", default=500, type=int, dest="printiter", help="Number of iters to print")
    parser.add_argument("--mode", default='train', type=str, dest="mode", help="choose train / Get_Embedding / Get_Feature_Importance")

    args = parser.parse_args()
    
    def seed_everything(random_seed):
        """
        fix seed to control any randomness from a code 
        (enable stability of the experiments' results.)
        """
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(random_seed)
        random.seed(random_seed)
        os.environ['PYTHONHASHSEED'] = str(random_seed)

    seed_everything(args.seed)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    ## Make folder 
    if not os.path.exists(args.result_dir):
        os.makedirs(os.path.join(args.result_dir))

    if not os.path.exists(args.ckpt_dir):
        os.makedirs(os.path.join(args.ckpt_dir))

    ## Build Dataset 
    print(f'Build Dataset : {args.mimic_data_dir} ....')
    dataset_train = FT_EMB_Dataset(data_path=args.mimic_data_dir, data_type='mimic',mode='train',seed=args.seed)
    # sample weight
    # y_train_indices = dataset_train.df_num.index
    # y_train = [dataset_train.y[i] for i in y_train_indices]
    # class_sample_count = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])

    # weight = 1. / class_sample_count
        
    # samples_weight = np.array([weight[int(t)-1] for t in y_train])
    # samples_weight = torch.from_numpy(samples_weight)
    # sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))
    
    loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # Tuple Containing the number of unique values within each category
    card_categories = []
    for col in dataset_train.df_cat.columns:
        card_categories.append(dataset_train.df_cat[col].nunique())
        
        
    def train(trial, search = False):
    
        
        patience = args.patience
        early_stop_counter = 0
        
        log_file = "FT-Transformer.txt" 
            
        def log_message(message):
            with open(log_file, "a") as file:
                file.write(message + "\n")
            print(message)
        def log_message_not_print(message):
            with open(log_file, "a") as file:
                file.write(message + "\n")
            
        search_iter = 0
        # search parameters
        if search == True:
            
            search_iter += 1
            
            lr = trial.suggest_uniform('lr', 0.000009, 0.0005)
            dim      = trial.suggest_int('emb dim', 60,100)
            dim_head      = trial.suggest_int('Dimension of Attention(Q,K,V)', 16,100)
            heads = trial.suggest_int('head', 2,6)
            depth      = trial.suggest_int('depth', 2,4)
            ff_dropout = trial.suggest_uniform('FeedForward Layer dropout', 0.5, 0.79)
            temp       = trial.suggest_uniform('temp', 0.1, 0.5)
            lambda_weight = trial.suggest_uniform('weight decay', 0.2, 0.9)
            total_epoch = args.num_epoch
            
        else:
            lr = args.lr
            dim      = args.dim
            dim_head      = args.dim_head
            heads = args.heads
            depth      = args.depth
            ff_dropout = args.ff_dropout
            temp       = 0.1
            total_epoch = args.num_epoch

        
        print(f'learning_rate : {lr}, \nepoch :  {total_epoch}, Embedding Dimension of Input Data : {dim}, Dimension of Attention : {dim_head}, Attention Head : {heads}, Nums of Attention Layer Depth : {depth} drop_rate : {ff_dropout:.4f} temperature : {temp:.4f}')
        wandb.init(name=f'Ftt-Baseline: {lr}',
            project="CLEAR-Shock", config={
            "learning_rate": lr,
            "dropout": ff_dropout,
            'dim': dim,
            'depth':depth,
            'heads':heads,
            'dim_head':dim_head,
            'attn_dropout':args.attn_dropout,
            'temp':temp,
            # 'weight decay':lambda_weight,
            'num_special_tokens': 2,
        })
        # model define
        emb_model = BaselineFT(categories=card_categories,
        num_continuous=args.num_cont,
        dim=dim,
        depth=depth,
        heads=heads,
        dim_head=dim_head,
        num_special_tokens = 2,
        attn_dropout=ff_dropout,
        ff_dropout=ff_dropout).to(device)

        entropy = nn.CrossEntropyLoss().to(device)
        optimizer = optim.AdamW(emb_model.parameters(), lr = lr)
        
        ## Model Train and Eval
        Best_valid_loss = 1e9
        for epoch in range(1, total_epoch+1):
            emb_model.train()
            running_loss = 0
            
            for num_iter, batch_data in enumerate(tqdm(loader_train)):
                optimizer.zero_grad()
                
                X_num, X_cat, label = batch_data
                X_num, X_cat, label = X_num.to(device), X_cat.to(device), label.to(device)

                pred = emb_model(X_cat,X_num,True)
                targets = label - 1
                label =  targets.type(torch.LongTensor).to(device)
                
                # backward pass
                loss = entropy(pred.to(device), label)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
            avg_train_loss = running_loss / len(loader_train)
            print(f'Epoch {epoch}/{total_epoch} - Train Loss: {avg_train_loss:.4f}')
            
            
            with torch.no_grad():
                emb_model.eval()
                running_loss = 0
                for num_iter, batch_data in enumerate(tqdm(loader_val)):
                    X_num, X_cat, label = batch_data
                    X_num, X_cat, label = X_num.to(device), X_cat.to(device), label.to(device)
                    
                    targets = label - 1
                    label =  targets.type(torch.LongTensor).to(device)
                    
                    pred = emb_model(X_cat,X_num,True)
                    loss = entropy(pred.to(device), label)                
                    running_loss += loss.item()
                    
            avg_valid_loss = np.round(running_loss / len(loader_val),4) 
            print(f'Epoch{epoch} / {total_epoch} Valid Loss : {avg_valid_loss}')
            wandb.log({"train loss":avg_train_loss, "valid loss":avg_valid_loss})

            if avg_valid_loss < Best_valid_loss:
                print(f'Best Loss {Best_valid_loss:.4f} -> {avg_valid_loss:.4f} Update! & Save Checkpoint')
                Best_valid_loss = avg_valid_loss
                early_stop_counter = 0
                torch.save(emb_model.state_dict(),f'{args.ckpt_dir}/FTT_EMB_pattern.pth')
                
            else:
                early_stop_counter += 1

            if early_stop_counter >= patience:
                print("Early stopping triggered due to valid loss")
                return avg_valid_loss
            
        return avg_valid_loss
    
    
    
    if args.mode == "train":
        dataset_val = FT_EMB_Dataset(data_path=args.mimic_data_dir, data_type='mimic',mode='valid',seed=args.seed)
        loader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, drop_last=True)
        
        gc.collect()
        os.environ["CUDA_VISIBLE_DEVICES"]= '0'
        os.environ['CUDA_LAUNCH_BLOCKING']= '1'
        n_gpu             = 1

        # Set parameters
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(), direction="minimize")
        study.optimize(train, n_trials = 1) 

        pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        
        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
            
        
            
    elif args.mode == 'Inference':
        eicu_train = FT_EMB_Dataset(data_path=args.eicu_data_dir, data_type='eicu',mode='all',seed=args.seed)
        loader_eicu_out = DataLoader(eicu_train, batch_size=args.batch_size, shuffle=False, drop_last=False)

        mimic_train = FT_EMB_Dataset(data_path=args.mimic_data_dir, data_type='mimic',mode='train',seed=args.seed)
        loader_trn_out = DataLoader(mimic_train, batch_size=args.batch_size, shuffle=False, drop_last=False)

        mimic_valid = FT_EMB_Dataset(data_path=args.mimic_data_dir, data_type='mimic',mode='valid',seed=args.seed)
        loader_val_out = DataLoader(mimic_valid, batch_size=args.batch_size, shuffle=False, drop_last=False)
        
        model = BaselineFT(categories=card_categories,
        num_continuous=args.num_cont,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        dim_head=args.dim_head,
        num_special_tokens = 2,
        attn_dropout=args.attn_dropout,
        ff_dropout=args.ff_dropout).to(device)
        
        checkpoint = torch.load(f'{args.ckpt_dir}/FTT_EMB_pattern.pth')
        model.load_state_dict(checkpoint)
    

        print('Start Getting the Valid Prediction value')
        model.eval()
        with torch.no_grad():
            for idx, batch_data in enumerate(tqdm(loader_val_out)):
                X_num, X_cat, label = batch_data
                X_num, X_cat, label = X_num.to(device), X_cat.to(device), label.to(device)
            
                pred = model(X_cat,X_num,True)
                
                probabilities = F.softmax(pred, dim=1)
                predicted_classes = torch.argmax(probabilities, dim=1)
                predicted_classes = predicted_classes.unsqueeze(1)
                
                targets = predicted_classes + 1
             
                if not idx:
                    pred_arrays = targets.detach().cpu().numpy()
            
                else:
                    pred_arrays = np.vstack((pred_arrays,targets.detach().cpu().numpy()))
            
            np.save('/Users/DAHS/Desktop/ECP_CONT/ECP_SCL/Training/Train/result/FTT_inference_valid_emb.npy',pred_arrays)       
        
    
        print('Start Getting the Test Prediction value')
        model.eval()
        with torch.no_grad():
            for idx, batch_data in enumerate(tqdm(loader_eicu_out)):
                X_num, X_cat, label = batch_data
                X_num, X_cat, label = X_num.to(device), X_cat.to(device), label.to(device)
            
                pred = model(X_cat,X_num,True)
                
                probabilities = F.softmax(pred, dim=1)
                predicted_classes = torch.argmax(probabilities, dim=1)
                predicted_classes = predicted_classes.unsqueeze(1)
                
                targets = predicted_classes + 1

                if not idx:
                    pred_arrays = targets.detach().cpu().numpy()
            
                else:
                    pred_arrays = np.vstack((pred_arrays,targets.detach().cpu().numpy()))

            np.save('/Users/DAHS/Desktop/ECP_CONT/ECP_SCL/Training/Train/result/FTT_inference_test_emb.npy',pred_arrays)
                
    # elif args.mode == 'Get_Feature_Importance':

    #     model = FTTransformer(categories=card_categories,
    #     num_continuous=args.num_cont,
    #     dim=args.dim,
    #     depth=args.depth,
    #     heads=args.heads,
    #     dim_head=args.dim_head,
    #     num_special_tokens = 2,
    #     attn_dropout=args.attn_dropout,
    #     ff_dropout=args.ff_dropout).to(device)
        
    #     checkpoint = torch.load("Contrastive_Embedding_Net_ftt(0313_best_batch=64).pt")
    #     model.load_state_dict(checkpoint["model_state_dict"])
                
    #     case2 = Positive_Case2(data_path=args.mimic_data_dir, data_type='mimic',mode='valid',seed=args.sees)
    #     case2_loader = DataLoader(case2, batch_size=32, shuffle=True, drop_last=False)

    #     case4 = Positive_Case4(data_path=args.mimic_data_dir, data_type='mimic',mode='valid',seed=args.sees)
    #     case4_loader = DataLoader(case4, batch_size=32, shuffle=True, drop_last=False)
        
        
            
    #     def attention_map_case2(model_name, case2_loader, device):
    #         print('Start Getting the Attention positive(Case2)')
    #         model_name.eval()
    #         att_maps = [] 

    #         with torch.no_grad():
    #             for idx, batch_data in enumerate(tqdm(case2_loader)):
    #                 X_num, X_cat, label = batch_data
    #                 X_num, X_cat = X_num.to(device), X_cat.to(device)
    #                 _, att_valid = model_name(X_cat, X_num, True)

    #                 att_maps.append(att_valid.cpu().numpy()) 
    #             final = np.concatenate(att_maps, axis = 1)
                
    #         return final
        
    #     def attention_map_case4(model_name, case4_loader, device):
    #         print('Start Getting the Attention positive(Case4)')
    #         model_name.eval()
    #         att_maps = [] 

    #         with torch.no_grad():
    #             for idx, batch_data in enumerate(tqdm(case4_loader)):
    #                 X_num, X_cat, label = batch_data
    #                 X_num, X_cat = X_num.to(device), X_cat.to(device)
    #                 _, att_valid = model_name(X_cat, X_num, True)

    #                 att_maps.append(att_valid.cpu().numpy())
            
    #             final = np.concatenate(att_maps, axis = 1)
    #         return final
        
        
    #     case2_attn = attention_map_case2(model, case2_loader, device)
    #     case4_attn = attention_map_case4(model, case4_loader, device)
        
    #     att_dict = {1:[], 2:[], 3:[], 4:[]}
        
    #     att_dict[2] = case2_attn

    #     att_dict[4] = case4_attn
        
    #     def softmax(x):
    #         e_x = np.exp(x - np.max(x))
    #         return e_x / e_x.sum(axis=1, keepdims=True)

    #     heads = 5
    #     layers = 4

    #     columns = ['CLS_Token'] + dataset_train.df_cat.columns.tolist() + dataset_train.df_num.columns.tolist()
    #     cls = [2.0, 4.0]

    #     valid_indices = [i for i, col in enumerate(columns) if  col != "CLS_Token"]

    #     for classes in cls:
    #         ps = []
    #         class_attention_map = att_dict[classes]  # (l, b, h, f, f)
            
    #         if classes == 2:
    #             sample_len = 196
    #         else:
    #             sample_len = 763 

    #         for sample in range(sample_len):
    #             get_cls_attentionmap = class_attention_map[:, sample, :, 0:1, :]  # (l, h, 1, f), Sample requires 1 sample for each message (layer, head, cls 1, width)
    #             sigma = get_cls_attentionmap.sum(axis=0).sum(axis=0) #(1, 227)
    #             p = sigma / (heads * layers)
    #             ps.append(p)

    #         p = np.array(ps).sum(axis=0) / sample_len * 100 #scaling
            
    #         if p.ndim == 2 and p.shape[0] == 1:
    #             p = p.ravel()

    #         p_filtered = p[valid_indices]
    #         top_feature_indices = np.argsort(p_filtered)[-20:][::-1]
    #         top_p_values = p_filtered[top_feature_indices]
    #         top_feature_names = [columns[valid_indices[i]] for i in top_feature_indices]

            
    #         sorted_indices = np.argsort(top_p_values)[::-1]
    #         sorted_p_values = np.array(top_p_values)[sorted_indices]
    #         sorted_feature_names = np.array(top_feature_names)[sorted_indices]
            
      
    #         bars = plt.barh(range(len(sorted_p_values)), sorted_p_values, color='gray', edgecolor='black')

    #         plt.yticks(ticks=range(len(sorted_feature_names)), labels=sorted_feature_names, fontsize=12)
    #         plt.xlabel('Importance', fontsize=12) 
         
    #         plt.tight_layout() 
    #         plt.gca().invert_yaxis() 

    #         plt.savefig('feature_importance.png')
    #         plt.close()
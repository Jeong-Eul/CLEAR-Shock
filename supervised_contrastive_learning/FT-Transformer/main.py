import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from model import *
from data_provider import *
from utils import *

import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser(description="Train the UNet",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Data
parser.add_argument("--eicu_data_dir", default="./dataset/eicu_ARDS.csv", type=str, dest="eicu_data_dir")
parser.add_argument("--mimic_data_dir", default="./dataset/mimic_ARDS.csv", type=str, dest="mimic_data_dir")
parser.add_argument('--seed', default=42, type=int , dest='seed')

# Train Method
parser.add_argument('--optimizer', default='AdamW', type=str, dest='optim')
parser.add_argument("--lr", default=1e-3, type=float,dest="lr")
parser.add_argument("--batch_size", default=128, type=int, dest="batch_size")
parser.add_argument("--num_epoch", default=100, type=int, dest="num_epoch")
parser.add_argument("--weight_decay", default=1e-5, type=float, dest="weight_decay")
parser.add_argument("--scheduler", action='store_true', dest="scheduler", help="True or False to enable CosineAnnealing scheduler")
parser.add_argument("--T_max", default=25, type=int, dest="T_max")
parser.add_argument("--early_stop", action='store_true', dest="early_stop", help="True or False to enable early stop")

# Model
parser.add_argument("--num_cont", default=61, type=int, dest="num_cont", help = "Nums of Continuous Features")
parser.add_argument("--num_cat", default=57, type=int, dest="num_cat", help = "Nums of Categorical Features But Not Use")
parser.add_argument("--dim", default=32, type=int, dest="dim", help = "Embedding Dimension of Input Data ")
parser.add_argument("--dim_head", default=16, type=int, dest="dim_head", help = "Dimension of Attention(Q,K,V)")
parser.add_argument("--dim_out", default=1, type=int, dest="dim_out", help="Task output dimension")
parser.add_argument("--depth", default=6, type=int, dest="depth", help = "Nums of Attention Layer Depth")
parser.add_argument("--heads", default=8, type=int, dest="heads", help='Nums of Attention head')
parser.add_argument("--attn_dropout", default=0.1, type=float, dest="attn_dropout", help='Ratio of Attention Layer dropout')
parser.add_argument("--ff_dropout", default=0.1, type=float, dest="ff_dropout", help='Ratio of FeedForward Layer dropout')

# Others
parser.add_argument("--ckpt_dir", default="./checkpoint", type=str, dest="ckpt_dir")
parser.add_argument("--log_dir", default="./log", type=str, dest="log_dir")
parser.add_argument("--result_dir", default="./result", type=str, dest="result_dir")
parser.add_argument("--threshold", default=0.5, type=float, dest="threshold", help="Set Threshold for the Binary Classification")
parser.add_argument("--mode", default='train', type=str, dest="mode", help="choose train / test")

args = parser.parse_args()

def fix_seed(random_seed):
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

fix_seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## Make folder 
if not os.path.exists(args.result_dir):
    os.makedirs(os.path.join(args.result_dir))

if not os.path.exists(args.ckpt_dir):
    os.makedirs(os.path.join(args.ckpt_dir))

## Build Dataset 
print(f'Build Dataset ....')
if args.mode == 'train':
    dataset_train = TableDataset(data_path=args.eicu_data_dir, data_type='eicu',mode='train',seed=args.seed)
    loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=4)

    dataset_val = TableDataset(data_path=args.eicu_data_dir, data_type='eicu',mode='valid',seed=args.seed)
    loader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Tuple Containing the number of unique values within each category
    card_categories = []
    for col in dataset_train.df_cat.columns:
        card_categories.append(dataset_train.df_cat[col].nunique())

else:
    dataset_test = TableDataset(data_path=args.mimic_data_dir, data_type='mimic',mode='test',seed=args.seed)
    loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=4)

    # Tuple Containing the number of unique values within each category
    card_categories = []
    for col in dataset_test.df_cat.columns:
        card_categories.append(dataset_test.df_cat[col].nunique())

## Prepare Model
model = FTTransformer(
    categories = card_categories,      
    num_continuous = args.num_cont,                
    dim = args.dim,                       
    dim_out = args.dim_out,                        
    depth = args.depth,                         
    heads = args.heads, 
    dim_head = args.dim_head,                      
    attn_dropout = args.attn_dropout,              
    ff_dropout = args.ff_dropout                   
).to(device)

criterion = nn.BCELoss().to(device)

if args.optim == 'AdamW':
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
else:
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

if args.scheduler:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.T_max)

if args.early_stop:
    patience = 3
    early_stop_counter = 0
 
## Tensorboard Setting
writer_train = SummaryWriter(log_dir=os.path.join(args.log_dir, 'train'))
writer_val = SummaryWriter(log_dir=os.path.join(args.log_dir, 'val'))

## Model Train and Eval
start_epoch = 0
Best_valid_loss = 1e9

## Train mode
if args.mode == 'train':
    print(f'Train Start....')
    for epoch in range(start_epoch + 1, args.num_epoch + 1):
        model.train()
        running_loss = 0
        corr = 0

        for num_iter, batch_data in enumerate(tqdm(loader_train)):
            optimizer.zero_grad()
            X_num, X_cat, label = batch_data
            X_num, X_cat, label = X_num.to(device), X_cat.to(device), label.to(device)

            output,attn_map = model(X_cat,X_num,True)

            # backward pass
            output = torch.sigmoid(output)
            loss = criterion(output, label.unsqueeze(dim = 1))
            loss.backward()
            optimizer.step()
            
            # Check Accuracy
            pred = output >= torch.FloatTensor([args.threshold]).to(device).int() 
            corr += pred.eq(label.unsqueeze(dim=1)).sum().item()
            running_loss += loss.item()
            
            if args.scheduler:
                scheduler.step()

            if num_iter % 3000 == 0:
                print("TRAIN: EPOCH %04d / %04d | ITER %04d / %04d | LOSS %.4f" %
                    (epoch, args.num_epoch, num_iter+1, len(loader_train), running_loss / (num_iter+1)))
        print(f'Epoch{epoch} / {args.num_epoch} Train Loss : {running_loss / len(loader_train)} | Train Accuracy : {corr / (len(loader_train) * args.batch_size)}')

        writer_train.add_scalar('Train_Epoch_loss', running_loss / len(loader_train), epoch)
        writer_train.add_scalar('Train_Epoch_Accuracy', corr / (len(loader_train) * args.batch_size), epoch)
        print(f'---------Epoch{epoch} Training Finish---------')

        with torch.no_grad():
            model.eval()
            running_loss = 0
            corr = 0

            for num_iter, batch_data in enumerate(tqdm(loader_val)):
                X_num, X_cat, label = batch_data
                X_num, X_cat, label = X_num.to(device), X_cat.to(device), label.to(device)
                
                output,attn_map = model(X_cat,X_num,True)

                output = torch.sigmoid(output)
                loss = criterion(output, label.unsqueeze(dim = 1))

                pred = output >= torch.FloatTensor([args.threshold]).to(device).int() 
                corr += pred.eq(label.unsqueeze(dim=1)).sum().item()
                running_loss += loss.item()

            if num_iter % 3000 == 0:
                print("VALID: EPOCH %04d / %04d | ITER %04d / %04d | LOSS %.4f" %
                    (epoch, args.num_epoch, num_iter+1, len(loader_val), running_loss / (num_iter+1)))
                
        print(f'Epoch{epoch} / {args.num_epoch} Valid Loss : {running_loss / len(loader_val)} | Valid Accuracy : {corr / (len(loader_val) * args.batch_size)}')

        writer_train.add_scalar('Valid_Epoch_loss', running_loss / len(loader_val), epoch)
        writer_train.add_scalar('Valid_Epoch_Accuracy', corr / (len(loader_val) * args.batch_size), epoch)

        if running_loss / len(loader_val) < Best_valid_loss:
            print(f'Best Loss {Best_valid_loss:.4f} -> {running_loss / len(loader_val):.4f} Update! & Save Checkpoint')
            Best_valid_loss = running_loss / len(loader_val)
            early_stop_counter = 0
            torch.save(model.state_dict(),f'{args.ckpt_dir}/FTTransformer.pth')
        else:
            early_stop_counter += 1
        
        print(f'---------Epoch{epoch} Valid Finish---------')

    writer_train.close()
    writer_val.close()

## Inference(Test) mode
else:
    print(f'Inference Start....')
    print(f'Checkpoint Load....')
    model.load_state_dict(torch.load(f'{args.ckpt_dir}/FTTransformer.pth'))

    with torch.no_grad():
        model.eval()
        running_loss = 0
        corr = 0
        output_list = []
        pred_list = []

        for num_iter, batch_data in enumerate(tqdm(loader_test)):
            X_num, X_cat, label = batch_data
            X_num, X_cat, label = X_num.to(device), X_cat.to(device), label.to(device)
            
            output,attn_map = model(X_cat,X_num,True)

            output = torch.sigmoid(output)
            loss = criterion(output, label)

            pred = output >= torch.FloatTensor([args.threshold]).to(device).int() 
            corr += pred.eq(label.unsqueeze(dim=1)).sum().item()
            running_loss += loss.item()

            output_list.append(output.detach().cpu().numpy())
            pred_list.append(pred.detach().cpu().numpy())

        print("TEST RESULT: LOSS %.4f | ACCURACY %.4f"  %
            ( running_loss / len(loader_test), corr / len(loader_test)))
        
        result_csv = pd.DataFrame({'Output' : output_list, 'Pred' : pred_list})
        result_csv.to_csv(f'{args.result_dir}/Test_result.csv',index = False)
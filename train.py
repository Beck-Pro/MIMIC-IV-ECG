import json
import os
import random
import numpy as np
import torch
import wandb
import pandas as pd
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

from model.model import Net1D, ECGNet
from dataset.dataset import ECGDataset
from utils.utils import *


def main():
    # parser
    args = parse_args('ECG Training')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.atrial == 'afib':
        config_path = './config/afib_config.json'
    if args.atrial == 'aflu':
        config_path = './config/aflu_config.json'
    config = load_config(config_path)
    print(config)
    set_seed(config['seed'])

    wandb.login()
    wandb.init(
        project=config['wandb_project'], 
        name = config['name'],
        config = {k: getattr(config, k) for k in dir(config) if not k.startswith("__") and not callable(getattr(config, k))}
    )

    # train csv
    train_df = pd.read_csv(config['train_path'])
    train_df = train_df.dropna(subset=['report_0', 'study_id'])
    train_df = train_df.reset_index(drop=True)

    # validation csv
    val_df = pd.read_csv(config['val_path'])
    val_df = val_df.reset_index(drop=True)
    
    # compute label quality - afib & aflu 공통
    # label quality 의 의미
    # label이 aflu or afib(양성)에 가까운 정도를 score로 나타냄(0~1)
    label_quality = compute_all_label_qualities(train_df, config['wave_dir'])

    # 1) high qual label training ------------------------------------------------------

    report = train_df['report_0'].values

    # 양성 라벨 = label quality >=0.5 
    # 음성 라벨 = label quality <=0.3
    high_quality_mask = (
        ((report == 1) & (label_quality >= 0.5)) |  # 양성 
        ((report == 0) & (label_quality <= 0.3))    # 음성 
    )

    # high & low qual dataframe
    df_high = train_df[high_quality_mask].copy()
    df_low = train_df[~high_quality_mask].copy()

    loader_init = DataLoader(
        ECGDataset(df_high, ecg_dir = config['wave_dir'], cfg = config, test=False),
        batch_size=8, shuffle=True)
    
    #focal = FocalLoss(alpha=0.75, gamma=2.0)
    model = Net1D(in_channels=1, num_classes=1).to(device)
    #optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    optim = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    for epoch in range(config['epochs']):
        print(f"\nepoch {epoch+1}/{config['epochs']}")
        model.train()
        total_loss = 0
        for x, y in tqdm(loader_init, desc='high qual train'):
            x, y = x.to(device), y.to(device).unsqueeze(1)
            pred = model(x)
            loss = F.binary_cross_entropy_with_logits(pred, y) # 기본 BCE
            #loss = focal(pred, yb)
            optim.zero_grad(); loss.backward(); optim.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader_init)
        print(f"Train Loss: {avg_loss:.4f}")

    # 2) pseudo label 생성------------------------------------------------------------
    loader_low = DataLoader(
        ECGDataset(df_low, ecg_dir = config['wave_dir'], cfg=config, test=True),
        batch_size=config['batch_size'], shuffle=False)
    
    loader_high_all = DataLoader(
        ECGDataset(df_high, ecg_dir=config['wave_dir'], cfg = config, test=True),
        batch_size = config['batch_size'], shuffle=False)

    model.eval()
    pseudo_low = []
    pseudo_high = []
    with torch.no_grad():
        for x in loader_low: # low qual
            x = x.to(device)
            pred = model(x).sigmoid().cpu().numpy().flatten()
            pseudo_low.append(pred)

        for x in loader_high_all: # high qual
            x = x.to(device)
            pred = model(x).sigmoid().cpu().numpy().flatten()
            pseudo_high.append(pred)

    pseudo_low = np.concatenate(pseudo_low)
    pseudo_high = np.concatenate(pseudo_high)

    # 2-1) pseudo label 필터링------------------------------------------------------------
    # pseudo pred_proba의 값들 중 특정 값 이상 이하만 남김
    confidence_threshold_low = 0.6  # low qual pred_proba는 0.6 이상, 0.4 이하만 남김
    confidence_threshold_high = 0.5 # high qual pred_proba는 전부

    low_mask = (pseudo_low >= confidence_threshold_low) | (pseudo_low <= (1-confidence_threshold_low)) # 
    high_mask = (pseudo_high >= confidence_threshold_high) | (pseudo_high <= (1-confidence_threshold_high))

    df_low = df_low[low_mask].reset_index(drop=True)
    pseudo_low = pseudo_low[low_mask]
    label_quality_low = label_quality[~high_quality_mask][low_mask]

    df_high = df_high[high_mask].reset_index(drop=True)
    pseudo_high = pseudo_high[high_mask]
    label_quality_high = label_quality[high_quality_mask][high_mask]

    # 3) refined label 생성------------------------------------------------------------
    # Soft label refinement
    df_low['refined'] = (1-label_quality_low) * pseudo_low
    df_high['refined'] = label_quality_high * df_high['report_0'].values + (1-label_quality_high) * pseudo_high

    # 4) student 학습 (refined low + high의 50%)------------------------------------------------------------
    num_half_high = len(df_high) // 2   # high의 50%
    df_high_half = df_high.sample(n = num_half_high, random_state = 42)
    df_train_student = pd.concat([df_low, df_high_half]).reset_index(drop=True)

    loader_student = DataLoader(
        ECGDataset(df_train_student, ecg_dir=config['wave_dir'], cfg = config, test=False, refined=True),
        batch_size=config['batch_size'], shuffle=True)
    
    student = ECGNet(kernels=[3, 5, 7, 9]).to(device)   # 새로운 모델(student) 
    #optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
    optimizer = AdamW(student.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=10)
    #focal = FocalLoss(alpha=0.75, gamma=2.0)

    for epoch in range(12):
        student.train()
        total_loss = 0
        for x, y in loader_student:
            x, y = x.to(device), y.to(device).unsqueeze(1)
            pred = student(x)
            loss = F.binary_cross_entropy_with_logits(pred, y)
            #loss = focal(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()    # cosinea scheduler
        print(f"[student train] epoch {epoch+1}: loss = {total_loss/len(loader_student):.4f}")

    # 5) refined high qual data fine tuning
    loader_finetune = DataLoader(
        ECGDataset(df_high, ecg_dir = config['wave_dir'], cfg = config, test=False, refined=True),
        batch_size = config['batch_size'], shuffle=True)
    
    val_loader = DataLoader(
        ECGDataset(val_df, ecg_dir=config['wave_dir'], cfg = config, test=False),
        batch_size = config['batch_size'], shuffle=False)

    best_sens = 0.0
    early_stopping_counter = 0
    patience = 5  

    # fine tuning
    for epoch in range(20):  
        print(f"\n[finetune] epoch {epoch+1}")
        student.train()
        total_loss = 0
        for x, y in loader_finetune:
            x, y = x.to(device), y.to(device).unsqueeze(1)
            pred = student(x)
            loss = F.binary_cross_entropy_with_logits(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader_finetune)
        print(f"[finetune] train loss: {avg_loss:.4f}")

        # val 
        student.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = student(x).sigmoid().cpu().numpy().flatten()
                val_preds.append(pred)
                val_targets.append(y.cpu().numpy())

        pred_proba = np.concatenate(val_preds)
        true = np.concatenate(val_targets)
        preds = (pred_proba > 0.5).astype(int) # 0.5로 항상 고정

        # metric 
        val_f1, val_auc, val_auprc, val_acc, val_sens, val_spec, bal_acc = compute_metrics(true, preds, pred_proba)
        print(f"[val] f1: {val_f1:.4f} | sensitivity: {val_sens:.4f} | auroc: {val_auc:.4f} | auprc: {val_auprc:.4f}| specificity: {val_spec:.4f} | bal_Acc: {bal_acc:.4f}")

        wandb.log({
            'fine_tune_loss': avg_loss,
            'val_acc': val_acc,
            'val_f1': val_f1,
            'val_auc': val_auc,
            'val_auprc': val_auprc,
            'val_sensitivity': val_sens,
            'val_specificity': val_spec,
            'val_bal_acc': bal_acc
        }, step=epoch)

        # early stopping 
        if val_sens > best_sens:
            best_sens = val_sens
            early_stopping_counter = 0
            torch.save(student.state_dict(), os.path.join(config['output_dir'], config['model_name']))
            print("best sens model saved")
        else:
            early_stopping_counter += 1
            print(f"early stopping counter: {early_stopping_counter}/{patience}")

        if early_stopping_counter >= patience:
            print("early stopping!!")
            break
    wandb.finish()

    # val pred 저장
    val_df['pred'] = preds
    val_df.to_csv(os.path.join(config['output_dir'], f"val_{config['wandb_project']}_pred.csv"), index=False)

if __name__ == "__main__":
    main()

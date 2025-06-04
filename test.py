import argparse
import json
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from utils.utils import *
from dataset.dataset import ECGDataset
from model.model import ECGNet

def predict(model, df, wave_dir, device, cfg, threshold=0.5, batch_size=32):
    '''
    Args:
        df : 데이터프레임
        wave_dir : wave .npy path
        device : device
        cfg : config
        threshold : 0.5 
        batch_size : 32
    Returns:
        dataframe 반환
    '''
    test_dataset = ECGDataset(df, ecg_dir=wave_dir, cfg=cfg, test=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    all_preds = []
    with torch.no_grad():
        for x in test_loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            all_preds.extend(probs)
    df['pred_proba'] = all_preds
    df['pred_label'] = (df['pred_proba'] >= threshold).astype(int)  # 0.5로 고정
    return df


def main():
    args = parse_args('ECG Test Inference')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.atrial == 'afib':
        config_path = './config/afib_config.json'
    if args.atrial == 'aflu':
        config_path = './config/aflu_config.json'
    config = load_config(config_path)
    set_seed(config['seed'])

    model = ECGNet(kernels=config['kernels']).to(device)
    model_path = os.path.join(config['output_dir'], config['model_name'])
    model.load_state_dict(torch.load(model_path, map_location=device))
 
    test_df = pd.read_csv(config['test_path'])
    pred_df = predict(model, test_df, config['wave_dir'], device, cfg=config, threshold=config['threshold'], batch_size=config['batch_size'])
    pred_df.to_csv(os.path.join(config['output_dir'], f"test_{config['wandb_project']}_pred.csv"), index=False)

    y_true = pred_df['report_0'].values
    y_pred = pred_df['pred_label'].values
    y_probs = pred_df['pred_proba'].values

    # test metric
    auc, auprc, acc, f1, sensitivity, specificity, bal_acc = compute_metrics(y_true, y_pred, y_probs)
    print(f"AUC: {auc:.4f}")
    print(f'AUPRC: {auprc:.4f}')
    print(f"F1 Score: {f1:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f'Bal_acc : {bal_acc:.4f}')


if __name__ == '__main__':
    main()
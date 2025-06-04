# dataset.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.utils import butter_highpass_filter, butter_lowpass_filter, notch_filter, normalize_signal

class ECGDataset(Dataset):
    '''
    Args:
        df : dataframe
        ecg_dir : wave directory
        cfg : config
        test : test 일 시 x만 반환
        refined : refined한 dataset일 시 'refined'칼럼에서 데이터 가져오기
    Returns:
        x,y or x만
    '''
    def __init__(self, df, ecg_dir, cfg, test=False, refined=False):
        self.df = df.reset_index(drop=True)
        self.ecg_dir = ecg_dir
        self.test = test
        self.refined = refined
        self.cfg = cfg
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        
        row = self.df.iloc[index]
        path = os.path.join(self.ecg_dir, f"{row.study_id}.npy")

        # signal 불러오기
        signal = np.load(path)
        signal = signal[:,1] # lead ii
        signal = np.nan_to_num(signal)
        signal = np.clip(signal, -1024, 1024) / 32.0

        # 필터링
        signal = butter_highpass_filter(signal, cutoff = 0.5, fs=self.cfg['fs']) 
        signal = butter_lowpass_filter(signal, cutoff = 45, fs=self.cfg['fs'])
        signal = notch_filter(signal, freq = 60.0, fs=self.cfg['fs'])
        signal = signal.copy()

        # z-score normalization
        mean = np.mean(signal)
        std = np.std(signal) + 1e-6 
        signal = (signal - mean) / std

        x = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)  # shape: (12, 5000)
        
        if self.test:
            return x

        if self.refined:
            # refined label 가져오기
            y = torch.tensor(row.refined, dtype=torch.float32)
        else:
            # 원래 report_0 label
            y = torch.tensor(row.report_0, dtype=torch.float32)

        return x, y
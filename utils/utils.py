import wandb
import os
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    f1_score, recall_score, confusion_matrix, matthews_corrcoef
)
import argparse
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
import torch
import random
import json
from scipy.signal import welch
from tqdm import tqdm

def parse_args(desc):
    '''
    Args:
        desc : description 문자열 (ex: 'train description')
    Returns:
        atrial (afib or aflu)를 포함한 인자들
    '''
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-a', '--atrial', default='afib', type=str, help='afib or aflu')
    return parser.parse_args()

def set_seed(seed=42):
    '''
    시드 고정
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi-GPU일 경우
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(config_path):
    '''
    config 파일 불러오기
    '''
    with open(config_path) as f:
        config = json.load(f)
    return config

def compute_metrics(y_true, y_pred, y_pred_probs):
    '''
    Args:
        y_true : 실제 라벨
        y_pred : 예측 라벨 
        y_pred_probs : 예측 확률 
    Returns:
        auc, auprc, acc, f1, sensitivity, specificity, bal_acc
    '''
    auc = roc_auc_score(y_true, y_pred_probs)
    auprc = average_precision_score(y_true, y_pred_probs)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    bal_acc = 0.5 * (sensitivity + specificity)
 
    return auc, auprc, acc, f1, sensitivity, specificity, bal_acc

def butter_highpass_filter(data, cutoff=0.5, fs=500, order=4):
    '''
    Args:
        data : 입력 신호
        cutoff : high pass cutoff frequency 
        fs : sampling rate
        order : 필터 차수
    Returns:
        high pass filtering된 신호
    '''
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data, axis=0)

def butter_lowpass_filter(data, cutoff=150, fs=500, order=4):
    '''
    Args:
        data : 입력 신호
        cutoff : low pass cutoff frequency 
        fs : sampling rate
        order : 필터 차수
    Returns:
        low pass filtering된 신호
    '''
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data, axis=0)

def notch_filter(data, freq=60.0, fs=500.0, Q=30.0):
    '''
    노치 필터 (60hz 제거)
    Args:
        freq : 제거할 frequency
        fs : sampling rate
        Q : Quality factor 제거 하고 싶은 주파수 대역의 폭을 결정. 높을수록 60Hz 부근 제거 폭이 좁아짐.
    Returns:
        60Hz 대역 제거된 신호
    '''
    b, a = iirnotch(freq / (0.5 * fs), Q)
    return filtfilt(b, a, data, axis=0)

def normalize_signal(signal):
    '''
    Args:
        signal : 입력 신호
    Returns:
        평균 0, 표준편차 1로 정규화된 신호
    '''
    mean = np.mean(signal, axis=0, keepdims=True)
    std = np.std(signal, axis=0, keepdims=True) + 1e-6
    return (signal - mean) / std

def estimate_f_wave_periodicity(signal: np.ndarray, fs: int = 500) -> float:
    '''
        Args:
            signal : 입력 신호 lead ii
            fs : sampling rate
        Returns:
            f wave 주기성 점수 (0~1 사이)
    '''
    f, pxx = welch(signal, fs=fs, nperseg=1024)
    f_band = (4.0, 7.0)  # AFLU typical band
    band_power = np.sum(pxx[(f > f_band[0]) & (f < f_band[1])])
    total_power = np.sum(pxx)
    return band_power / total_power if total_power > 0 else 0

def compute_label_quality_atrial(row, wave, fs=500):
    '''
    Args:
        row : 메타데이터 한 줄 (rr_interval, qrs_onset, qrs_end 포함)
        wave : ECG 신호 lead ii
        fs : sampling rate
    Returns:
        0~1 사이의 label quality score
    '''
    score = 0.0

    # 1. RR interval standard deviation (irregular 허용)
    rr = row.get("rr_interval", None)
    if rr is not None and 200 < rr < 2000:
        rr_std = np.std([rr])  # 단일 RR은 std가 의미 없음 → 복수 RR에서 쓰면 더 좋음
        if rr_std < 80:
            score += 0.3

    # 2. f- wave periodicity
    f_score = estimate_f_wave_periodicity(wave, fs)
    if f_score > 0.6:
        score += 0.5
    elif f_score > 0.3:
        score += 0.3

    # 3. QRS width (정상범위)
    qrs_onset = row.get("qrs_onset", None)
    qrs_end = row.get("qrs_end", None)
    if qrs_onset is not None and qrs_end is not None:
        qrs_dur = qrs_end - qrs_onset
        if 60 <= qrs_dur <= 120:
            score += 0.2

    return np.clip(score, 0.0, 1.0)

def compute_all_label_qualities(meta_df, ecg_dir):
    '''
    Args:
        meta_df : train_flu.csv 또는 train_fib.csv 데이터프레임
        ecg_dir : .npy ecg 파일들이 저장된 폴더 경로
    Returns:
        모든 샘플에 대한 label quality 점수 배열
    '''
    scores = []
    for _, row in tqdm(meta_df.iterrows(), total=len(meta_df)):
        try:
            signal = np.load(os.path.join(ecg_dir, f"{row['study_id']}.npy"))[:, 1]  # lead ii
            score = compute_label_quality_atrial(row, signal)
        except Exception as e:
            print(f"Error on {row['study_id']}: {e}")
            score = 0.0
        scores.append(score)
    return np.array(scores)
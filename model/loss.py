import torch
from torch import nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    '''
    Focal loss
    Args:
        alpha : 양성 클랙스에 얼마나 가중치 줄지 
        gamma : 어려운 샘플에 집중하기 위한 계수
        reduction : 'mean' 이면 평균', 'sum'이면 총합, 'none'이면 그대로 loss 반환
    Returns:
        focal_weight가 계산된 bce_loss의 mean 또는 sum
    '''
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        probs = torch.sigmoid(inputs)   # 예측된 확률값
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = self.alpha * (1 - pt) ** self.gamma  # 어려운 샘플일수록 더 큰 가중치 부여
        loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
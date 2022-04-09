import torch
from torch import nn, Tensor
import numpy as np
import random
class OhemCrossEntropy(nn.Module):
    def __init__(self, ignore_label: int = 255, weight: Tensor = None, thresh: float = 0.7, aux_weights: list = [1, 1]) -> None:
        super().__init__()
        self.ignore_label = ignore_label
        self.aux_weights = aux_weights
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float))
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label, reduction='none')

    def _forward(self, preds: Tensor, labels: Tensor) -> Tensor:
        # preds in shape [B, C, H, W] and labels in shape [B, H, W]
        n_min = labels[labels != self.ignore_label].numel() // 16
        loss = self.criterion(preds, labels).view(-1)
        loss_hard = loss[loss > self.thresh]

        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)

        return torch.mean(loss_hard)

    def forward(self, preds, labels: Tensor) -> Tensor:
        if isinstance(preds, tuple):
            return sum([w * self._forward(pred, labels) for (pred, w) in zip(preds, self.aux_weights)])
        return self._forward(preds, labels)




import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import einops
class FocalLoss(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.size_average = size_average
        self.class_num=class_num

    def forward(self, inputs, targets):
        device=targets.device
        l = len(inputs.shape)
        if l == 3:
            inputs = einops.rearrange(inputs, 'b c l ->(b l) c')
            targets = einops.rearrange(targets, 'b l -> (b l)')
        if l == 4:
            inputs = einops.rearrange(inputs, 'b c h w ->(b h w) c')
            targets = einops.rearrange(targets, 'b h w -> (b h w)')
        P = F.softmax(inputs, dim=-1)

        bs = inputs.shape[0]
        if inputs.shape[-1]!=self.class_num:
            raise IndexError('the final dim of output should be the same as class_num')
        P = P[(torch.arange(bs), targets)].contiguous()   #True class对应的分数
        P += 1e-16
        log_p = P.log()
        weight=self.alpha[targets]

        weight = weight.to(device)

        batch_loss = -weight * (torch.pow((1 - log_p), self.gamma)) * log_p
        #batch_loss = -weight * (torch.pow((1 - P), self.gamma)) * log_p
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    setup_seed(20)
    input = torch.randn(32,2,224,224)
    input = torch.zeros(32,2,224,224)
    label = torch.ones(32,224,224,dtype=torch.long)

    loss = FocalLoss(2,alpha=torch.tensor([0.6,2]))
    loss_ = loss(input,label)
    print(loss_.item())


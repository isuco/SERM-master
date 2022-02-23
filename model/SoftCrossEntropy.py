import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,input,target):
        input=F.log_softmax(input,dim=-1)
        input=-torch.mul(input,target).sum(dim=-1)
        return torch.mean(input)
#
# input=torch.rand((3,3))
# targets=torch.eye(3)
# target=torch.Tensor([0,1,2]).long()
# criter=torch.nn.CrossEntropyLoss()
# criters=SoftCrossEntropyLoss()
# print(criter(input,target))
# print(criters(input,targets))


import torch
import torch.nn as nn


class Contastive_Loss(nn.Module):
    def __init__(self, m):
        super(Contastive_Loss, self).__init__()
        self.m = m

    def forward(self, inputs, outputs):
        N = inputs.shape[0]
        loss1 = torch.sum((inputs - outputs) ** 2) / N
        loss2 = self.m - torch.sum((inputs[:, :3] - outputs[:, :3]) ** 2) / N
        loss = loss1 + loss2
        return loss
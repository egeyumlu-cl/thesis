import torch.nn as nn
import torch


class CS_Loss(nn.Module):
    def __init__(self):
        super(CS_Loss, self).__init__()

    def forward(self, outputs, targets):
        numerator = torch.sum(outputs * targets)
        denominator = torch.sqrt(torch.sum(outputs ** 2)) * torch.sqrt(torch.sum(targets ** 2))
        cs = numerator / denominator
        return 1 - cs.mean()


class FDD_Loss(nn.Module):
    def __init__(self, lambda_value):
        super(FDD_Loss, self).__init__()
        self.lambda_value = lambda_value
        self.mse_loss = nn.MSELoss()
        self.cs_loss = CS_Loss()

    def forward(self, outputs, targets):
        mse = self.mse_loss(outputs, targets)
        cs = self.cs_loss(outputs, targets)
        fdd = mse + self.lambda_value * (1 - cs) / 2
        return fdd

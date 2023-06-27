import torch
import torch.nn as nn


class CustomLoss(nn.Module):
    def __init__(self, lambda_factor):
        super(CustomLoss, self).__init__()
        self.lambda_factor = lambda_factor
        self.mean_feature_vectors = torch.zeros((1,))

    def forward(self, predicted_output, target):
        # Calculate the reconstruction loss
        reconstruction_loss = torch.mean((predicted_output - target) ** 2)

        # Calculate the distance penalty term
        feature_vectors = predicted_output.view(predicted_output.shape[0], -1)
        distances = torch.norm(feature_vectors - self.mean_feature_vectors, dim=1)
        distance_penalty = torch.mean(torch.clamp(distances - torch.mean(distances), min=0) ** 2)

        # Calculate the total loss
        total_loss = (1 - self.lambda_factor) * reconstruction_loss + self.lambda_factor * distance_penalty

        return total_loss

    def update_mean_feature_vectors(self, predicted_output):
        feature_vectors = predicted_output.view(predicted_output.shape[0], -1)
        self.mean_feature_vectors = torch.mean(feature_vectors, dim=0)
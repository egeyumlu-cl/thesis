import torch
import torch.nn as nn
import torch.optim as optim


class MLPClassifier(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(100, 30),
            nn.ReLU(),
        )

        self.classification_layer = nn.Linear(30, 2)

    def forward(self, x):
        out = self.encoder(x)
        out = self.fc2(out)
        return out

    def train_model(self, epochs, inputs, labels, optimizer, loss_fn):
        for epoch in range(epochs):
            # Forward pass
            outputs = self(inputs)
            loss = loss_fn(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print loss for monitoring
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}")

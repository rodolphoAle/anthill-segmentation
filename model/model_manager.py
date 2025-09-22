import torch
import torch.nn as nn
import torch.optim as optim

class ModelManager:
    def __init__(self, model, device=None):
        self.model = model
        self.device = device or (torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model.to(self.device)

    def train(self, dataloader, criterion, optimizer, num_epochs=20):
        self.model.train()
        for epoch in range(num_epochs):
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        return self.model

    def evaluate(self, dataloader, criterion):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
        return total_loss / len(dataloader)

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)

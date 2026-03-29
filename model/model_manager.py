import torch
import torch.nn as nn


def dice_score(pred, target):
    pred = torch.argmax(pred, dim=1)
    intersection = (pred & target).float().sum()
    union = pred.float().sum() + target.float().sum()
    return (2. * intersection) / (union + 1e-8)


class ModelManager:
    def __init__(self, model, device=None):
        self.model = model
        self.device = device or (torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model.to(self.device)

    def train(self, train_loader, val_loader, criterion, optimizer, epochs=20):
        best_loss = float('inf')

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0

            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)

                optimizer.zero_grad()
                out = self.model(x)

                loss = criterion(out, y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            val_loss = self.evaluate(val_loader, criterion)

            print(f"Epoch {epoch} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

            if val_loss < best_loss:
                best_loss = val_loss
                self.save("best_model.pth")

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

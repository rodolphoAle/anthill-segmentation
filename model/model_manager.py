import torch
import torch.nn as nn
import torch.nn.functional as F

def iou_score(pred, target, ignore_index=255):
    pred = torch.argmax(pred, dim=1)

    valid = target != ignore_index
    pred = pred[valid]
    target = target[valid]

    intersection = ((pred == 1) & (target == 1)).float().sum()
    union = ((pred == 1) | (target == 1)).float().sum()
    if union == 0:
        return torch.tensor(1.0, device=pred.device)

    return intersection / (union + 1e-8)


def dice_score(pred, target, ignore_index=255):
    pred = torch.argmax(pred, dim=1)

    valid = target != ignore_index
    pred = pred[valid]
    target = target[valid]

    intersection = ((pred == 1) & (target == 1)).float().sum()
    union = (pred == 1).float().sum() + (target == 1).float().sum()
    if union == 0:
        return torch.tensor(1.0, device=pred.device)

    return (2. * intersection) / union


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, ignore_index=255):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        ce = F.cross_entropy(
            inputs,
            targets,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction="none",
        )

        valid = targets != self.ignore_index
        ce = ce[valid]

        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.4, beta=0.6, gamma=1.33, smooth=1.0, ignore_index=255):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        probs = F.softmax(inputs, dim=1)
        pred = probs[:, 1, :, :]

        valid = targets != self.ignore_index
        target = (targets == 1).float()

        pred = pred[valid]
        target = target[valid]

        tp = (pred * target).sum()
        fp = (pred * (1 - target)).sum()
        fn = ((1 - pred) * target).sum()

        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )

        return (1 - tversky) ** self.gamma


class LovaszHingeLoss(nn.Module):
    def __init__(self, ignore_index=255):
        super().__init__()
        self.ignore_index = ignore_index

    def lovasz_grad(self, gt_sorted):
        p = gt_sorted.numel()
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.cumsum(0)
        union = gts + (1 - gt_sorted).cumsum(0)
        jaccard = 1 - intersection / union

        if p > 1:
            jaccard[1:p] = jaccard[1:p] - jaccard[:-1]

        return jaccard

    def forward(self, inputs, targets):
        margin = inputs[:, 1] - inputs[:, 0]

        valid = targets != self.ignore_index
        margin = margin[valid]
        target = (targets[valid] == 1).float()

        if margin.numel() == 0:
            return inputs.sum() * 0.0

        signs = 2 * target - 1
        errors = 1 - margin * signs
        errors_sorted, perm = torch.sort(errors, descending=True)
        gt_sorted = target[perm]

        grad = self.lovasz_grad(gt_sorted)

        return torch.dot(F.relu(errors_sorted), grad)

    
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

            val_loss, val_iou, val_dice = self.evaluate(val_loader, criterion)

            lr = optimizer.param_groups[0]['lr']
            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Train: {train_loss:.4f} | "
                f"Val: {val_loss:.4f} | "
                f"IoU: {val_iou:.4f} | "
                f"Dice: {val_dice:.4f} | "
                f"LR: {lr:.2e}"
            )

            if val_loss < best_loss:
                best_loss = val_loss
                best_model_path = f"best_model_epoch{epoch+1}_iou{val_iou:.3f}.pth"
                self.save(best_model_path)

    def evaluate(self, dataloader, criterion):
        self.model.eval()
        total_loss = 0
        total_iou = 0
        total_dice = 0
        batches = 0

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)

                loss = criterion(outputs, labels)
                total_loss += loss.item()

                total_iou += iou_score(outputs, labels).item()
                total_dice += dice_score(outputs, labels).item()
                batches += 1

        if batches == 0:
            return 0.0, 0.0, 0.0

        return (
            total_loss / batches,
            total_iou / batches,
            total_dice / batches,
        )

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)

from __future__ import annotations
import torch
from torch.utils.data import DataLoader


def evaluate(model, dataset, device="cpu", batch_size=256):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    correct = total = 0
    loss_sum = 0.0
    ce = torch.nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_sum += float(ce(logits, y).item())
            correct += int((logits.argmax(1) == y).sum().item())
            total += int(y.numel())
    return {"accuracy": correct / max(total, 1), "loss": loss_sum / max(total, 1)}

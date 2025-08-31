\
import torch

def accuracy(logits, labels):
    if labels is None or logits.size(0) != labels.size(0):
        return 0.0
    preds = logits.argmax(dim=1)
    return float((preds == labels).sum().item() / len(labels))

def clone_state_dict(sd):
    return {k: v.detach().clone() for k,v in sd.items()}

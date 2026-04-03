"""
Training / eval steps for the DJ model.

Adapted from upstream `utils.py`:
https://github.com/AhmedBegggaUA/Diffusion-Jump-GNNs
"""
from __future__ import annotations

import torch


def train(adj, data, model, train_mask, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out, loss_norm = model(data.x, adj)
    pred = out.argmax(dim=1).squeeze(0)
    train_correct = pred[train_mask] == data.y[train_mask]
    train_acc = int(train_correct.sum()) / int(train_mask.sum())
    loss = loss_norm + criterion(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    return loss, train_acc


def val(adj, data, model, val_mask):
    model.eval()
    out, _ = model(data.x, adj)
    pred = out.argmax(dim=1).squeeze(0)
    test_correct = pred[val_mask] == data.y[val_mask]
    test_acc = int(test_correct.sum()) / int(val_mask.sum())
    return test_acc


def test(adj, data, model, test_mask):
    model.eval()
    out, _ = model(data.x, adj)
    pred = out.argmax(dim=1).squeeze(0)
    test_correct = pred[test_mask] == data.y[test_mask]
    test_acc = int(test_correct.sum()) / int(test_mask.sum())
    return test_acc

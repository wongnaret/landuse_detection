#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
 File name: lstm_utils.py
 Date Create: 14/9/2021 AD 08:45
 Author: Wongnaret Khantuwan 
 Email: wongnaet.khantuwan@nectec.or.th, wongnaret@gmail.com
 Python Version: 3.9
"""

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.optim.lr_scheduler import _LRScheduler

from conf import ID_COLS


def numpy_fill(arr):
    '''Solution provided by Divakar.'''
    mask = np.isnan(arr)
    idx = np.where(~mask,np.arange(mask.shape[1]),0)
    np.maximum.accumulate(idx,axis=1, out=idx)
    out = arr[np.arange(idx.shape[0])[:,None], idx]
    return out

def loops_fill(arr):
    out = arr.copy()
    for row_idx in range(out.shape[0]):
        for col_idx in range(out.shape[1]):
            for t_idx in range(1, out.shape[2]):
                if np.isnan(out[row_idx, col_idx, t_idx]):
                    out[row_idx, col_idx, t_idx] = out[row_idx, col_idx, t_idx - 1]
    return out

def convert_to_df(x_trn, product):
    x_df = pd.DataFrame(x_trn)
    x_df = x_df.stack(level=0)
    x_df = x_df.to_frame().reset_index()
    x_df = x_df.rename(columns={"level_0": "series_id", "level_1": "measurement_number", 0: product}, errors="raise")
    return x_df

def convert_to_df(x_trn, product):
    x_df = pd.DataFrame(x_trn)
    x_df = x_df.stack(level=0)
    x_df = x_df.to_frame().reset_index()
    x_df = x_df.rename(columns={"level_0": "series_id", "level_1": "measurement_number", 0: product}, errors="raise")
    return x_df

# PyTorch Wrappers
def create_datasets(X, y, test_size=0.2, dropcols=ID_COLS, time_dim_first=False):
    enc = LabelEncoder()
    y_enc = enc.fit_transform(y)
    X_grouped = create_grouped_array(X)
    if time_dim_first:
        X_grouped = X_grouped.transpose(0, 2, 1)

    X_train, X_valid, y_train, y_valid = train_test_split(X_grouped, y_enc, test_size=0.1)

    X_train, X_valid = [torch.tensor(arr, dtype=torch.float32) for arr in (X_train, X_valid)]
    y_train, y_valid = [torch.tensor(arr, dtype=torch.long) for arr in (y_train, y_valid)]
    train_ds = TensorDataset(X_train, y_train)
    valid_ds = TensorDataset(X_valid, y_valid)
    return train_ds, valid_ds, enc


def create_grouped_array(data, group_col='series_id', drop_cols=ID_COLS):
    X_grouped = np.row_stack([
        group.drop(columns=drop_cols).values[None]
        for _, group in data.groupby(group_col)])
    return X_grouped


def create_test_dataset(X, drop_cols=ID_COLS):
    X_grouped = np.row_stack([
        group.drop(columns=drop_cols).values[None]
        for _, group in X.groupby('series_id')])
    X_grouped = torch.tensor(X_grouped.transpose(0, 2, 1)).float()
    y_fake = torch.tensor([0] * len(X_grouped)).long()
    return TensorDataset(X_grouped, y_fake)


def create_loaders(train_ds, valid_ds, bs=512, jobs=0):
    train_dl = DataLoader(train_ds, bs, shuffle=True, num_workers=jobs)
    valid_dl = DataLoader(valid_ds, bs, shuffle=False, num_workers=jobs)
    return train_dl, valid_dl


def accuracy(output, target):
    return (output.argmax(dim=1) == target).float().mean().item()

def feed(x_trn, model, IS_CUDA):
    print('Preparing datasets')
    test_dl = DataLoader(create_test_dataset(x_trn), batch_size=64, shuffle=False)

    y_pred = []
    conf_lv = []
    print('Predicting on test dataset')
    for batch, _ in test_dl:
        batch = batch.permute(0, 2, 1)
        if IS_CUDA:
            batch = batch.cuda()
        else:
            batch = batch.cpu()

        out = model(batch)
        y_hat = F.log_softmax(out, dim=1).argmax(dim=1)
        y_pred += y_hat.tolist()
        
    return y_pred


def feed_raw(x_trn, model, IS_CUDA):
    print('Preparing datasets')
    test_dl = DataLoader(create_test_dataset(x_trn), batch_size=64, shuffle=False)

    y_pred = []
    conf_lv = []
    print('Predicting on test dataset')
    for batch, _ in test_dl:
        batch = batch.permute(0, 2, 1)
        if IS_CUDA:
            batch = batch.cuda()
        else:
            batch = batch.cpu()
        out = model(batch)
        y_hat = out.argmax(dim=1)
        y_pred += y_hat.tolist()
        conf_lv += out.tolist()

    return (y_pred, conf_lv)

def feed_confident(x_trn, model, IS_CUDA):
    print('Preparing datasets')
    test_dl = DataLoader(create_test_dataset(x_trn), batch_size=64, shuffle=False)

    y_pred = []
    conf_lv = []
    print('Predicting on test dataset')
    for batch, _ in test_dl:
        batch = batch.permute(0, 2, 1)
        if IS_CUDA:
            batch = batch.cuda()
        else:
            batch = batch.cpu()
        out = model(batch)
        y_hat = F.log_softmax(out, dim=1).argmax(dim=1)
        conf = F.softmax(out, dim=1)

        y_pred += y_hat.tolist()
        conf_lv += conf.tolist()

    return (y_pred, conf_lv)

# Cyclic Learning Rate
class CyclicLR(_LRScheduler):

    def __init__(self, optimizer, schedule, last_epoch=-1):
        assert callable(schedule)
        self.schedule = schedule
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.schedule(self.last_epoch, lr) for lr in self.base_lrs]


def cosine(t_max, eta_min=0):
    def scheduler(epoch, base_lr):
        t = epoch % t_max
        return eta_min + (base_lr - eta_min) * (1 + np.cos(np.pi * t / t_max)) / 2

    return scheduler
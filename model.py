# Copyright 2023 Carnegie Mellon University.
# MIT (SEI)
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
# This material is based upon work funded and supported by the Department of
# Defense under Contract No. FA8702-15-D-0002 with Carnegie Mellon University
# for the operation of the Software Engineering Institute, a federally funded
# research and development center.
# The view, opinions, and/or findings contained in this material are those of
# the author(s) and should not be construed as an official Government position,
# policy, or decision, unless designated by other documentation.
# NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING
# INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS" BASIS. CARNEGIE MELLON
# UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED,
# AS TO ANY MATTER INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR
# PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED FROM USE OF THE
# MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND
# WITH RESPECT TO FREEDOM FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.
# [DISTRIBUTION STATEMENT A] This material has been approved for public release
# and unlimited distribution.  Please see Copyright notice for non-US
# Government use and distribution.
# DM23-0186

import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.datasets import load_iris
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.ao.quantization import get_default_qat_qconfig_mapping
import torch.ao.quantization.quantize_fx as quantize_fx
from torch.autograd import Variable

import tqdm

class Model(nn.Module):
    def __init__(self, input_dim):
        super(Model, self).__init__()
        self.relu = nn.ReLU()
        self.layer1 = nn.Linear(input_dim, 64)
        self.layer2 = nn.Linear(64,3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.softmax(x,dim=1)
        return x

def check_logits(m):
    # read data
    X = np.load("X.npy")
    y_gold = np.load("y.npy")
    # get logits model
    logits = get_logits_model(m)
    # evaluate
    X = Variable(torch.from_numpy(X)).float()
    y = logits(X)
    y = y.detach().numpy()
    print("\n\nlogits are", end = " ")
    if not np.allclose(y, y_gold, rtol=1.0e-5, atol=1.0e-8):
        print("not", end = " ")
    print("equal")

def get_logits_model(m):
    logits = m
    logits.softmax = nn.Identity()
    return logits

def load_data():
    iris = load_iris()
    X = iris['data']
    y = iris['target']
    names = iris['target_names']
    feature_names = iris['feature_names']
    return X,y,names,feature_names

def load_model(file,quantized=False,X=None):
    if not os.path.isfile(file):
        print("To create the model files run:")
        print("python train.py")
        quit()

    if not quantized:
        model = torch.load(file)
    else:
        model = Model(X.shape[1])
        model.eval()
        qconfig_mapping = get_default_qat_qconfig_mapping("x86")
        model = quantize_fx.prepare_qat_fx(model, qconfig_mapping,X)
        model = quantize_fx.convert_fx(model)
        model.load_state_dict(torch.load(file))

    return model

def plot_roc_curve(model,X_test,y_test,label=None):
    plt.figure(figsize=(10, 10))
    plt.plot([0, 1], [0, 1], 'k--')

    X_test  = Variable(torch.from_numpy(X_test)).float()
    # One hot encoding
    enc = OneHotEncoder()
    Y_onehot = enc.fit_transform(y_test[:, np.newaxis]).toarray()

    with torch.no_grad():
        y_pred = model(X_test).numpy()
        fpr, tpr, threshold = roc_curve(Y_onehot.ravel(),y_pred.ravel())

    plt.plot(fpr, tpr, label='AUC = {:.3f}'.format(auc(fpr, tpr)))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    title = "ROC curve"
    if label is not None:
        title += ": " + label
    plt.title(title)
    plt.legend()
    plt.show()

def print_model(model, note=None):
    if note is not None:
        print("\n\nPrinting model: " + note)
    print(model)
    # NB: this is not the size of the weights only, but includes other metadata
    torch.save(model.state_dict(), "temp.p")
    print("Model size (bytes):", os.path.getsize("temp.p"))
    os.remove("temp.p")

def save_model(model,file,quantized=False):
    if not quantized:
        torch.save(model,file)
    else:
        torch.save(model.state_dict(),file)

def split_data(X,y):
    # Scale data to have mean 0 and variance 1
    # which is important for convergence of the neural network
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data set into training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=2)

    return X_train, X_test, y_train, y_test

def train_model(X_train,X_test,y_train,y_test,quantize=False,verbose=False):
    X_train = Variable(torch.from_numpy(X_train)).float()
    y_train = Variable(torch.from_numpy(y_train)).long()
    X_test  = Variable(torch.from_numpy(X_test)).float()
    y_test  = Variable(torch.from_numpy(y_test)).long()

    model = Model(X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn   = nn.CrossEntropyLoss()
    if quantize:
        qconfig_mapping = get_default_qat_qconfig_mapping("x86")
        model = quantize_fx.prepare_qat_fx(model,qconfig_mapping,X_train)
    model.train()
    EPOCHS  = 300
    loss_list     = np.zeros((EPOCHS,))
    accuracy_list = np.zeros((EPOCHS,))

    for epoch in tqdm.trange(EPOCHS):
        y_pred = model(X_train)
        loss = loss_fn(y_pred, y_train)
        loss_list[epoch] = loss.item()

        # Zero gradients
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            y_pred = model(X_test)
            correct = \
                (torch.argmax(y_pred, dim=1) == y_test).type(torch.FloatTensor)
            accuracy_list[epoch] = correct.mean()

    # plot training results
    if verbose:
        fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 6), sharex=True)
        ax1.plot(accuracy_list)
        ax1.set_ylabel("validation accuracy")
        ax2.plot(loss_list)
        ax2.set_ylabel("validation loss")
        ax2.set_xlabel("epochs");
        plt.show()

    print(f'\nmodel accuracy: {accuracy_list[-1]:2f}')
    if verbose:
        print_model(model, "model")

    # quantize model
    if quantize:
        model = quantize_fx.convert_fx(model)
        print_model(model, "quantized model")

    return model

def vis_data(X,y,names,feature_names):
    plt.style.use('ggplot')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    for target, target_name in enumerate(names):
        X_plot = X[y == target]
        ax1.plot(X_plot[:, 0], X_plot[:, 1],
                 linestyle='none',
                 marker='o',
                 label=target_name)
    ax1.set_xlabel(feature_names[0])
    ax1.set_ylabel(feature_names[1])
    ax1.axis('equal')
    ax1.legend();

    for target, target_name in enumerate(names):
        X_plot = X[y == target]
        ax2.plot(X_plot[:, 2], X_plot[:, 3],
                 linestyle='none',
                 marker='o',
                 label=target_name)
    ax2.set_xlabel(feature_names[2])
    ax2.set_ylabel(feature_names[3])
    ax2.axis('equal')
    ax2.legend();

    plt.show()

def write_logits(m,X):
    # write X
    np.save("X.npy", X)
    # get logits model
    logits = get_logits_model(m)
    # convert to tensor, run, and convert to numpy
    X = Variable(torch.from_numpy(X)).float()
    y = logits(X)
    y = y.detach().numpy()
    np.save("y.npy", y)

# -*- coding: utf-8 -*-

import torch
import numpy as numpy
from glob2 import glob

x_train = torch.tensor([[1],[2],[3],[4],[5]],dtype=torch.float)
y_train = torch.tensor([[10],[20],[30],[40], [50]], dtype=torch.float)


dtype = torch.float
device = torch.device("cpu")

model =  torch.nn.Linear(1, 1)
loss_fn = torch.nn.MSELoss(size_average=False)
learning_rate = .01
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for t in range(10000):
    y_pred = model(x_train)
    loss = loss_fn(y_pred, y_train)
    print("i = ", t)
    print("*************** loss = ", loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

torch.save(model, 'model/model.pt')

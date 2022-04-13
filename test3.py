import torch
import numpy as np

import torch.nn as nn
from torch.autograd import Variable
from gan_model import HSGAN_Generator, HSGAN_Discriminator

FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor


x = Variable(FloatTensor(np.random.randint(0, 2, (100, 1, 100))))
x2 = Variable(FloatTensor(np.random.randint(0, 2, (100, 1, 200))))

gen = HSGAN_Generator()
dis = HSGAN_Discriminator()
x = gen(x)
validity, label = dis(x2)
temp = torch.relu(torch.randn(2))


m = nn.Sigmoid()
input = torch.randn(2)
output = m(input)

# m1 = nn.Linear(100, 1024)
# m2 = nn.Linear(1024, 6400)
#
# # need add view
# m3 = nn.Upsample(size=100)
# m4 = nn.Conv1d(128, 64, 1)
# m5 = nn.Upsample(size=200)
# m6 = nn.Conv1d(64, 1, 1)
#
# x = m1(x)  # output shape (100, 1, 1024)
# x = m2(x)  # output shape (100, 1, 6400)
#
# x = x.view(100, 128, 50)  # output shape (100, 128, 50)
# x = m3(x)  # output shape (100, 128, 100)
# x = m4(x)  # output shape (100, 64, 100)
# x = m5(x)  # output shape (100, 64, 200)
# x = m6(x)

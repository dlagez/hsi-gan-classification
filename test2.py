import torch
import numpy as np

import torch.nn as nn
from torch.autograd import Variable

cuda = True if torch.cuda.is_available() else False


FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


x = Variable(FloatTensor(np.random.randint(0, 2, (100, 1, 200))))

m = nn.Conv1d(1, 32, 1)
m1 = nn.Conv1d(32, 32, 3)
m2 = nn.MaxPool1d(2, stride=2)
m3 = nn.Conv1d(32, 32, 3)
# m3
# m4 = nn.MaxPool1d(2, 2)
# view
m5 = nn.Linear(1504, 1024)

#
seq1 = nn.Sequential(
    nn.Linear(1024, 1),
    nn.Sigmoid()
)

#

seq2 = nn.Sequential(
    nn.Linear(1024, 16),
    nn.Softmax(dim=1)
)
m7 = nn.Linear(1024, 16)
softmax = nn.Softmax()

x = m(x)  # output shape (100, 32, 200)
x = m1(x)  # output shape (100, 32, 98)
x = m2(x)  # output shape (100, 32, 99)
x = m3(x)  # output shape (100, 32, 97)
x = m3(x)  # output shape (100, 32, 95)
x = m2(x)  # output shape (100, 32, 47)
x = x.view(100, 1, 1504)  # output shape (100, 1, 1504)
x = m5(x)  # output shape (100, 1, 1024)

valiate = seq1(x)
classify = seq2(x)















# class HSGAN_Discriminator(nn.Module):
#     """
#     Semisupervised Hyperspectral Image Classification Based on Generative Adversarial Networks
#     Ying Zhan , Dan Hu, Yuntao Wang
#     http://www.ieee.org/publications_standards/publications/rights/index.html
#     """
#
#     # input_channels=200 n_classes=17
#     def __init__(self, input_channels, n_classes):
#         # The proposed network model uses a single recurrent layer that adopts our modified GRUs of size 64 with sigmoid gate activation and PRetanh activation functions for hidden representations
#         super(HSGAN_Discriminator, self).__init__()
#         self.input_channels = input_channels
#         self.conv_blocks1 = nn.Sequential(
#             nn.Conv1d(1, 32, 1),
#             nn.Conv1d(32, 32, 3),
#         )
#         self.max_pooling = nn.MaxPool1d(2, stride=2)
#         self.conv_blocks2 = nn.Conv1d(32, 32, 3)
#         self.linear_block1 = nn.Linear(1504, 1024)
#         # real or fake
#         self.real_or_fake = nn.Sequential(
#             nn.Linear(1024, 1),
#             nn.Sigmoid(),
#         )
#         # class
#         self.classify = nn.Sequential(
#             nn.Linear(1024, 16),
#             nn.Softmax(),
#         )
#
#     def forward(self, x):   #(100, 1, 200)
#         x = self.conv_blocks1(x)
#         x = self.conv_blocks2(x)
#         x = self.max_pooling(x)
#         x = self.conv_blocks2(x)
#         x = self.max_pooling(x)
#         x = x.view(100, 1, 1504)
#         x = self.linear_block1(x)
#         # real or fake
#         validity = self.real_or_fake(x)
#         # classify
#         label = self.classify(x)
#         return validity, label
# dis = HSGAN_Discriminator(1, 0)
# gen = torch.randn(100, 1, 200)
# output = dis(gen)
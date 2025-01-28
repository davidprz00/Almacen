import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def Reg_L2(net, device):
    L2_pen = t.zeros(1, dtype = t.float32, device = device)

    for param in net.parameters():
        L2_pen += t.sum(param**2)

    return L2_pen

def Reg_L1(net):
    device = net.device

    L1_pen = t.zeros(1, dtype = t.float32, device = device)

    for param in net.parameters():
        L1_pen += t.sum(t.abs(param))

    return L1_pen

def Reg_Wavelet_mean0(net, device):
    Wavelet_mean0 = t.zeros(1, dtype = t.float32, device = device)

    parameters = dict(net.named_parameters())

    Wavelet_mean0 += t.sum(parameters['lift_cnn.weight'])**2
    Wavelet_mean0 += t.sum(parameters['group_cnn_1.weight'])**2
    Wavelet_mean0 += t.sum(parameters['group_cnn_2.weight'])**2

    return Wavelet_mean0

def Reg_Wavelet_2norm1(net, device):
    Wavelet_2norm1 = t.zeros(1, dtype = t.float32, device = device)

    parameters = dict(net.named_parameters())

    Wavelet_2norm1 += (t.sum(parameters['lift_cnn.weight']**2) - 1)**2
    Wavelet_2norm1 += (t.sum(parameters['group_cnn_1.weight']**2) - 1)**2
    Wavelet_2norm1 += (t.sum(parameters['group_cnn_2.weight']**2) - 1)**2

    return Wavelet_2norm1


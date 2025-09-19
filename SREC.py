# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as  F
#import MyLibForSteerCNN as ML
# import scipy.io as sio    
import math
# from PIL import Image


class Tconv3d(nn.Module):

    def __init__(self, sizeP, Tsize, inNum, outNum, tranNum, padding=None, bias=True,
                 iniScale=1):

        super(Tconv3d, self).__init__()

        self.tranNum = tranNum
        self.outNum = outNum
        self.inNum = inNum
        self.sizeP = sizeP  # spatial dimension =1
        self.Tsize = Tsize  # time conv size =3
        self.expand = tranNum  # default: ifIni=0, expand=4
        self.ifbias = bias

        # iniw = Getini_reg(sizeP, inNum, outNum, self.expand) * iniScale
        # (outNum=46,1,inNum=46,expand=4,sizeP=1,sizeP=1) 1 is for tranNum rotate
        # iniw = weights = (outNum,1,inNum,expand,sizeP,sizeP)
        iniw = Getini_reg_T3d(sizeP, Tsize, inNum, outNum, self.expand) * iniScale

        self.weights = nn.Parameter(iniw, requires_grad=True)
        if padding == None:
            self.padding = 0
        else:
            self.padding = padding

        if bias:
            self.c = nn.Parameter(torch.Tensor(1, outNum, 1, 1))
        else:
            self.register_parameter('c', None)
        self.reset_parameters()
        # self.c = nn.Parameter(torch.zeros(1, outNum, 1, 1), requires_grad=bias)
        # self.register_buffer("filter", torch.zeros(outNum*tranNum, inNum*self.expand, sizeP, sizeP))

    def forward(self, input):

        if self.training:
            tranNum = self.tranNum
            outNum = self.outNum
            inNum = self.inNum
            expand = self.expand

            # tempW = torch.einsum('ijok,mnak->monaij', self.Basis, self.weights)
            # tempW = [torch.rot90(self.weights, i, [4, 5]) for i in range(tranNum)]
            # tempW = torch.cat(tempW, dim=1)
            tempW_3d = self.weights.repeat([1, tranNum, 1, 1, 1, 1, 1])

            Num = tranNum // expand
            tempWList_3d = [torch.cat(
                [tempW_3d[:, i * Num:(i + 1) * Num, :, -i:, :, :, :],
                 tempW_3d[:, i * Num:(i + 1) * Num, :, :-i, :, :, :]], dim=3)
                for i in range(expand)]
            tempW_3d = torch.cat(tempWList_3d, dim=1)

            _filter = tempW_3d.reshape([outNum * tranNum, inNum * self.expand, self.Tsize, self.sizeP, self.sizeP])
            if self.ifbias:
                _bias = self.c.repeat([1, 1, tranNum, 1]).reshape([1, outNum * tranNum, 1, 1])
            # _bias = self.c.repeat([1, 1, tranNum, 1]).reshape([1, outNum * tranNum, 1, 1])
        else:
            _filter = self.filter
            if self.ifbias:
                _bias = self.bias
            # _bias = self.bias

        output = F.conv3d(input, _filter,
                          padding=self.padding,
                          dilation=1,
                          groups=1)
        if self.ifbias:
            output = output + _bias
        return output

    def train(self, mode=True):
        if mode:
            # TODO thoroughly check this is not causing problems
            if hasattr(self, "filter"):
                del self.filter
                if self.ifbias:
                    del self.bias
        elif self.training:
            # avoid re-computation of the filter and the bias on multiple consecutive calls of `.eval()`
            # print('Using Fixed Filter')
            tranNum = self.tranNum
            outNum = self.outNum
            inNum = self.inNum
            expand = self.expand

            # tempW = [torch.rot90(self.weights, i, [4, 5]) for i in range(tranNum)]
            # tempW = torch.cat(tempW, dim=1)
            tempW_3d = self.weights.repeat([1, tranNum, 1, 1, 1, 1, 1])

            Num = tranNum // expand
            tempWList_3d = [torch.cat(
                [tempW_3d[:, i * Num:(i + 1) * Num, :, -i:, :, :, :],
                 tempW_3d[:, i * Num:(i + 1) * Num, :, :-i, :, :, :]], dim=3)
                for i in range(expand)]
            tempW_3d = torch.cat(tempWList_3d, dim=1)

            _filter = tempW_3d.reshape([outNum * tranNum, inNum * self.expand, self.Tsize, self.sizeP, self.sizeP])
            self.register_buffer("filter", _filter)
            if self.ifbias:
                _bias = self.c.repeat([1, 1, tranNum, 1]).reshape([1, outNum * tranNum, 1, 1])
                self.register_buffer("bias", _bias)

        return super(Tconv3d, self).train(mode)
    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.c is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.c, -bound, bound)

class Fconv_PCA_3d(nn.Module):

    def __init__(self, sizeP, inNum, outNum, tranNum=8, inP=None, padding=None, ifIni=0, bias=True, Smooth=True,
                 iniScale=1.0):

        super(Fconv_PCA_3d, self).__init__()
        if inP == None:
            inP = sizeP
        self.tranNum = tranNum
        self.outNum = outNum
        self.inNum = inNum
        self.sizeP = sizeP
        Basis, Rank, weight = GetBasis_PCA(sizeP, tranNum, inP, Smooth=Smooth)
        self.register_buffer("Basis", Basis)  # .cuda())
        self.ifbias = bias
        if ifIni:
            expand = 1
        else:
            expand = tranNum
        # iniw = Getini_reg(Basis.size(3), inNum, outNum, self.expand, weight)*iniScale
        self.expand = expand
        self.weights = nn.Parameter(torch.Tensor(outNum, inNum, expand, Basis.size(3)), requires_grad=True)
        # nn.init.kaiming_uniform_(self.weights, a=0,mode='fan_in', nonlinearity='leaky_relu')
        if padding == None:
            self.padding = 0
        else:
            self.padding = padding

        if bias:
            self.c = nn.Parameter(torch.Tensor(1, outNum, 1, 1))
        else:
            self.register_parameter('c', None)
        self.reset_parameters()
        # self.c = nn.Parameter(torch.Tensor(1,outNum,1,1), requires_grad=True)

    def forward(self, input):

        if self.training:
            tranNum = self.tranNum
            outNum = self.outNum
            inNum = self.inNum
            expand = self.expand
            tempW = torch.einsum('ijok,mnak->monaij', self.Basis, self.weights)

            Num = tranNum // expand
            tempWList = [torch.cat(
                [tempW[:, i * Num:(i + 1) * Num, :, -i:, :, :], tempW[:, i * Num:(i + 1) * Num, :, :-i, :, :]], dim=3)
                         for i in range(expand)]
            tempW = torch.cat(tempWList, dim=1)

            # transform to 3d
            ini_filter = tempW.reshape([outNum * tranNum, inNum * self.expand, self.sizeP, self.sizeP])
            _filter = ini_filter.reshape([outNum * tranNum, inNum * self.expand, 1, self.sizeP, self.sizeP])

            if self.ifbias:
                _bias = self.c.repeat([1, 1, tranNum, 1]).reshape([1, outNum * tranNum, 1, 1])
        else:
            _filter = self.filter
            if self.ifbias:
                _bias = self.bias
        # output = F.conv2d(input, _filter,
        #                   padding=self.padding,
        #                   dilation=1,
        #                   groups=1)
        output = F.conv3d(input, _filter,
                          padding=self.padding,
                          dilation=1,
                          groups=1)
        if self.ifbias:
            output = output + _bias
        return output

    def train(self, mode=True):
        if mode:
            # TODO thoroughly check this is not causing problems
            if hasattr(self, "filter"):
                del self.filter
                if self.ifbias:
                    del self.bias
        elif self.training:
            # avoid re-computation of the filter and the bias on multiple consecutive calls of `.eval()`
            tranNum = self.tranNum
            outNum = self.outNum
            inNum = self.inNum
            expand = self.expand
            tempW = torch.einsum('ijok,mnak->monaij', self.Basis, self.weights)
            Num = tranNum // expand
            tempWList = [torch.cat(
                [tempW[:, i * Num:(i + 1) * Num, :, -i:, :, :], tempW[:, i * Num:(i + 1) * Num, :, :-i, :, :]], dim=3)
                         for i in range(expand)]
            tempW = torch.cat(tempWList, dim=1)
            # transform to 3d
            ini_filter = tempW.reshape([outNum * tranNum, inNum * self.expand, self.sizeP, self.sizeP])
            _filter = ini_filter.reshape([outNum * tranNum, inNum * self.expand, 1, self.sizeP, self.sizeP])


            self.register_buffer("filter", _filter)
            if self.ifbias:
                _bias = self.c.repeat([1, 1, tranNum, 1]).reshape([1, outNum * tranNum, 1, 1])
                self.register_buffer("bias", _bias)

        return super(Fconv_PCA_3d, self).train(mode)

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.c is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.c, -bound, bound)

class Fconv_PCA_out_3d(nn.Module):

    def __init__(self, sizeP, inNum, outNum, tranNum=8, inP=None, padding=None, ifIni=0, bias=True, Smooth=True,
                 iniScale=1.0):

        super(Fconv_PCA_out_3d, self).__init__()
        if inP == None:
            inP = sizeP
        self.tranNum = tranNum
        self.outNum = outNum
        self.inNum = inNum
        self.sizeP = sizeP
        Basis, Rank, weight = GetBasis_PCA(sizeP, tranNum, inP, Smooth=Smooth)
        self.register_buffer("Basis", Basis)  # .cuda())

        self.weights = nn.Parameter(torch.Tensor(outNum, inNum, 1, Basis.size(3)), requires_grad=True)
        # nn.init.kaiming_uniform_(self.weights, a=0,mode='fan_in', nonlinearity='leaky_relu')
        if padding == None:
            self.padding = 0
        else:
            self.padding = padding

        self.ifbias = bias
        if bias:
            self.c = nn.Parameter(torch.Tensor(1, outNum, 1, 1))
        else:
            self.register_parameter('c', None)
        self.reset_parameters()
        # self.c = nn.Parameter(torch.zeros(1,outNum,1,1), requires_grad=bias)

    def forward(self, input):

        if self.training:
            tranNum = self.tranNum
            outNum = self.outNum
            inNum = self.inNum
            tempW = torch.einsum('ijok,mnak->manoij', self.Basis, self.weights)

            # transform to 3d
            ini_filter = tempW.reshape([outNum, inNum * tranNum, self.sizeP, self.sizeP])
            _filter = ini_filter.reshape([outNum, inNum * tranNum, 1, self.sizeP, self.sizeP])

        else:
            _filter = self.filter
        if self.ifbias:
            _bias = self.c

        output = F.conv3d(input, _filter,
                          padding=self.padding,
                          dilation=1,
                          groups=1)
        if self.ifbias:
            output = output + _bias
        return output

    def train(self, mode=True):
        if mode:
            # TODO thoroughly check this is not causing problems
            if hasattr(self, "filter"):
                del self.filter
        elif self.training:
            # avoid re-computation of the filter and the bias on multiple consecutive calls of `.eval()`
            tranNum = self.tranNum
            tranNum = self.tranNum
            outNum = self.outNum
            inNum = self.inNum
            tempW = torch.einsum('ijok,mnak->manoij', self.Basis, self.weights)

            # transform to 3d
            ini_filter = tempW.reshape([outNum, inNum * tranNum, self.sizeP, self.sizeP])
            _filter = ini_filter.reshape([outNum, inNum * tranNum, 1, self.sizeP, self.sizeP])
            self.register_buffer("filter", _filter)
        return super(Fconv_PCA_out_3d, self).train(mode)

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.c is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.c, -bound, bound)

def Getini_reg(nNum, inNum, outNum,expand, weight = 1): 
    A = (np.random.rand(outNum,inNum,expand,nNum)-0.5)*2*2.4495/np.sqrt((inNum)*nNum)*np.expand_dims(np.expand_dims(np.expand_dims(weight, axis = 0),axis = 0),axis = 0)

    return torch.FloatTensor(A)

def Getini_reg_T3d(sizeP, Tsize, inNum, outNum, expand, weight=1):  # 等于ensium之后的
    A = (np.random.rand(outNum, 1, inNum, expand, Tsize, sizeP, sizeP) - 0.5) * 2 * 2.4495 / np.sqrt(
        (inNum) * sizeP * sizeP) / 10
    return torch.FloatTensor(A)

def GetBasis_PCA(sizeP, tranNum=8, inP=None, Smooth = True):
    if inP==None:
        inP = sizeP
    inp = inP//2
    inX, inY, Mask = MaskC(sizeP, tranNum)
    X0 = np.expand_dims(inX,2)
    Y0 = np.expand_dims(inY,2)
    Mask = np.expand_dims(np.expand_dims(Mask,2),3)
    theta = np.arange(tranNum)/tranNum*2*np.pi
    theta = np.expand_dims(np.expand_dims(theta,axis=0),axis=0)

    X = np.cos(theta)*X0-np.sin(theta)*Y0
    Y = np.cos(theta)*Y0+np.sin(theta)*X0
    
    X = X*inp
    Y = Y*inp

    X = np.expand_dims(np.expand_dims(X,3),4)
    Y = np.expand_dims(np.expand_dims(Y,3),4)
    
    k = np.reshape(np.arange(-inp, inp+1),[1,1,1,inP,1])
    l = np.reshape(np.arange(-inp, inp+1),[1,1,1,1,inP])
    
    # print(X[:,:,0,0,0])
    Basis = BicubicIni(X-k)*BicubicIni(Y-l)
    # print(Basis[:,:,1,2,2])
    
    Rank = inP*inP
    Weight = 1
    Basis = Basis.reshape([sizeP, sizeP, tranNum, Rank])*Mask
    
    return torch.FloatTensor(Basis), Rank, Weight

def BicubicIni(x):
    absx = np.abs(x)
    absx2 = absx**2
    absx3 = absx**3
    Ind1 = (absx<=1)
    Ind2 = (absx>1)*(absx<=2)
    temp = Ind1*(1.5*absx3-2.5*absx2+1)+Ind2*(-0.5*absx3+2.5*absx2-4*absx+2)
    return temp

def MaskC(SizeP, tranNum):
        p = (SizeP-1)/2
        x = np.arange(-p,p+1)/p
        X,Y  = np.meshgrid(x,x)
        C    =X**2+Y**2
        if tranNum ==4:
            Mask = np.ones([SizeP, SizeP])
        else:
            if SizeP>4:
                Mask = np.exp(-np.maximum(C-1,0)/0.2)
            else:
                Mask = np.exp(-np.maximum(C-1,0)/2)
        return X, Y, Mask

def build_mask(s, margin=2, dtype=torch.float32):
    mask = torch.zeros(1, 1, s, s, dtype=dtype)
    c = (s-1) / 2
    t = (c - margin/100.*c)**2
    sig = 2.
    for x in range(s):
        for y in range(s):
            r = (x - c) ** 2 + (y - c) ** 2
            if r > t:
                mask[..., x, y] = math.exp((t - r)/sig**2)
            else:
                mask[..., x, y] = 1.
    return mask

class GroupPooling(nn.Module):
    def __init__(self, tranNum):
        super(GroupPooling, self).__init__()
        self.tranNum = tranNum
        
    def forward(self, input):
        
        output = input.reshape([input.size(0), -1, self.tranNum, input.size(2), input.size(3)]) 
        output = torch.max(output,2).values
        return output
    

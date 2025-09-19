# import tensorflow as tf
# from tensorflow.keras import layers
# from tools.tools import Emat_xyt
from utils import ifft2c, fft2c, r2c,c2r, Emat_xyt
import torch
import torch.nn as nn
from torch.nn import init
import SREC as fn

class CNNLayer(nn.Module):
    def __init__(self, n_f=32, Trannum=4, sizep=3, tsize=3, inp=3 ):
        super(CNNLayer, self).__init__()

        self.seq_CNN = nn.Sequential(
            fn.Fconv_PCA_3d(sizeP=sizep, inNum=2, outNum= n_f, tranNum=Trannum, inP=inp, padding='same', ifIni=1, bias=False, Smooth=False),
            nn.LeakyReLU(0.2, True),

            fn.Tconv3d(sizeP=1, Tsize=tsize, inNum=n_f, outNum=n_f, tranNum=Trannum, padding='same', bias=False),
            nn.LeakyReLU(0.2, True),

            fn.Fconv_PCA_3d(sizeP=sizep, inNum=n_f, outNum= n_f, tranNum=Trannum, inP=inp, padding='same', ifIni=0, bias=False, Smooth=False),
            nn.LeakyReLU(0.2, True),

            fn.Tconv3d(sizeP=1, Tsize=tsize, inNum=n_f, outNum=n_f, tranNum=Trannum, padding='same', bias=False),
            nn.LeakyReLU(0.2, True),

            fn.Fconv_PCA_out_3d(sizeP=sizep, inNum=n_f, outNum= 2, tranNum=Trannum, inP=inp, padding='same', ifIni=0, bias=False, Smooth=False),
            nn.LeakyReLU(0.2, True),

            nn.Conv3d(in_channels=2, out_channels=2, kernel_size=(sizep, 1, 1), stride=1, padding='same', bias=False),
        )

    def forward(self, x):
        x2c = torch.stack([x.real, x.imag], axis=1)  # x2c.shape=[1,2,18,192,192]
        res = self.seq_CNN(x2c)
        res_c = r2c(res)                                     # [1,18,192,192]

        res_with_minus = x - res_c
        return res_with_minus


class DCLayer(nn.Module):
    def __init__(self, n_f_DC=12, sizep_DC=3 , Trannum=4,  tsize=3, inp=3):
        super(DCLayer, self).__init__()

        self.seq_DC = nn.Sequential(
            fn.Fconv_PCA_3d(sizeP=sizep_DC, inNum=4, outNum= n_f_DC, tranNum=Trannum, inP=inp, padding='same', ifIni=1, bias=False, Smooth=False),
            nn.ReLU(),
            fn.Tconv3d(sizeP=1, Tsize=tsize, inNum=n_f_DC, outNum=n_f_DC, tranNum=Trannum, padding='same', bias=False),
            nn.ReLU(),
            fn.Fconv_PCA_out_3d(sizeP=sizep_DC, inNum=n_f_DC, outNum= 2, tranNum=Trannum, inP=inp, padding='same', ifIni=0, bias=False, Smooth=False),
            nn.ReLU(),
            nn.Conv3d(in_channels=2, out_channels=2, kernel_size=(sizep_DC, 1, 1), stride=1, padding='same', bias=False),
        )

    def forward(self, x_DC, is_minus=False):
        nb, ncoil, nc, nt, nx, ny = x_DC.shape                               # x_DC(torch)=[1,20,2,18,192,192]
        x2c = torch.reshape(x_DC, (nb * ncoil, nc, nt, nx, ny))        # [20,2,18,192,192], let coil become batch
        x2c = torch.cat( (x2c.real, x2c.imag), dim=1 )                # [20,4,18,192,192]
        res = self.seq_DC(x2c)                                               # [20,2,18,192,192]
        re, im = torch.chunk(res, 2, 1)                          # 2*[20,1,18,192,192]
        res_c = torch.complex(re, im)                                         # [20,1,18,192,192]
        res_c = res_c.squeeze(1)                                              # [20,18,192,192]
        res_c = torch.reshape(res_c, (nb, ncoil, nt, nx, ny))         # [1,20,18,192,192]

        if is_minus:
            return x_DC - res_c
        return res_c



class RelaxConvNet(nn.Module):
    def __init__(self, niter):
        super(RelaxConvNet, self).__init__() #name='RelaxConvNet'
        self.niter = niter

        blocks = []

        for iter in range(niter-1):
            blocks.append(RelaxCell())
        blocks.append(RelaxCell_last())

        self.net = nn.ModuleList(blocks)

    def forward(self, d, csm, mask):

        Mpre = Emat_xyt(d, inv=True, csm=csm, mask=mask)   # (1,18,192,192)  # im_zf under-sampled
        dc = torch.zeros_like(Mpre)

        data = [Mpre, dc, d, csm]

        for iter in range(self.niter):
            data = self.net[iter](data=data, mask=mask)

        M, dc, _, _ = data
        # M = M - dc

        return M


class RelaxCell(nn.Module):
    def __init__(self):
        super(RelaxCell, self).__init__()

        self.sparseconv_xt = CNNLayer()
        self.dcconv = DCLayer()

    def forward(self, data, mask):

        Mpre, dc, d, csm = data

        # CNN_Layer
        M = self.sparseconv_xt(Mpre-dc)  # [1,18,192,192]

        # DC update : dc = self.dataconsis(M, d, csm)
        DC_input = torch.stack((Emat_xyt(M, inv=False, csm=csm, mask=mask), d), dim=2)
        # Emat_xyt(M)=[1,20,18,192,192]  stack with d on dim=2
        dc_resk = self.dcconv(DC_input)  # DC_input = [1,20,2,18,192,192]
        dc = Emat_xyt(dc_resk, inv=True, csm=csm, mask=mask)

        data[0] = M
        data[1] = dc

        return data


class RelaxCell_last(nn.Module):
    def __init__(self):
        super(RelaxCell_last, self).__init__()

        self.sparseconv_xt = CNNLayer()

    def forward(self, data, mask):

        Mpre, dc, d, csm = data
        M = self.sparseconv_xt(Mpre-dc)  # [1,18,192,192]

        data[0] = M

        return data



import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.utils.data as Data
import numpy as np
import scipy.io as sio
import mat73
from model_DUN_SRE import RelaxConvNet
from utils import ifft2c, mse
from argparse import ArgumentParser
import hdf5storage


class get_data(Data.Dataset):
    def __init__(self, mat_path):
        self.data_dir = mat_path
        self.sample_list = sorted(os.listdir(self.data_dir))
        self.data_len = len(self.sample_list)
        # self.data_list = hdf5storage.loadmat(self.data_dir)

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        sample_name = self.sample_list[idx]

        sample_path = os.path.join(self.data_dir, sample_name)
        data = sio.loadmat(sample_path)
        k0 = data['k0']
        csm = data['csm']
        sample = (torch.tensor(k0), torch.tensor(csm))

        return sample

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--num_epoch', metavar='int', nargs=1, default=['50'], help='number of epochs')
    parser.add_argument('--batch_size', metavar='int', nargs=1, default=['1'], help='batch size')
    parser.add_argument('--learning_rate', metavar='float', nargs=1, default=['0.001'], help='initial learning rate')
    parser.add_argument('--niter', metavar='int', nargs=1, default=['10'], help='number of network iterations')
    parser.add_argument('--acc', metavar='int', nargs=1, default=['16'], help='accelerate rate')
    parser.add_argument('--net', metavar='str', nargs=1, default=['RelaxConvNet'], help='network')
    parser.add_argument('--data', metavar='str', nargs=1, default=['cine'], help='dataset name')
    parser.add_argument('--model_ver_name', metavar='str', nargs=1, default=['model_name'], help='model version name')
    args = parser.parse_args()

    # os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Parameter Definition
    dataset_name = args.data[0].upper()
    batch_size = int(args.batch_size[0])
    num_epoch = int(args.num_epoch[0])
    learning_rate = float(args.learning_rate[0])
    multi_coil = 1
    acc = int(args.acc[0])
    net_name = args.net[0].upper()
    niter = int(args.niter[0])
    model_version_name = args.model_ver_name[0]

    # Load dataset
    mat_dir = './training_dataset/'
    trainset = get_data(mat_path = mat_dir)
    train_loader = Data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)
    print('dataset loaded.')
    print(net_name,  dataset_name, 'acc_', acc, 'iter', niter, 'batch_size', batch_size)

    # Definite dir
    model_id = net_name + '_' + dataset_name + 'iter'+ args.niter[0].upper() + 'nf32' + '_vista'
    modeldir = os.path.join('models/',model_version_name, model_id)
    if not os.path.exists(modeldir):
        os.makedirs(modeldir)

    loss_dir = os.path.join('loss/', model_version_name)
    if not os.path.isdir(loss_dir):
        os.makedirs(loss_dir)

    # prepare undersampling mask
    mask_dir = './Mask/'
    ## ========= vista ========= #
    mask_path = os.path.join(mask_dir, 'vista_18_192_192_acc_20.mat')
    mask_matlab_data = mat73.loadmat(mask_path)
    mask_t = mask_matlab_data['mask'].astype(np.complex64)
    # ========== mask post processing ========== #
    mask_t = np.expand_dims(mask_t, axis=0)
    mask = torch.from_numpy(mask_t)
    mask = mask.unsqueeze(0)

    # initialize network
    netR = RelaxConvNet(niter).to(device)
    print('network initialized. Model_version_name is :',model_version_name, '\nModel id is', model_id)
    # netR = init_net(netR, init_type='kaiming')

    # Training parameter definition
    learning_rate_org = learning_rate
    paramsR = netR.parameters()
    optimR = torch.optim.Adam(paramsR, lr=learning_rate_org, betas=(0.9, 0.999))
    ## ======= change lr at milestones ======== ##
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimR, milestones=[25, 35], gamma=0.1)
    ## ===== change lr every epcoh ===== ##
    gamma = 0.95   # lr_change rate
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimR, gamma=gamma) # every epoch lr=lr*0.95
    ## ================================= ##

    # Iterate over epochs.   # 定义训练参数，初始化
    total_step = 0
    param_num = 0
    loss = 0
    lossR = []
    Loss_list = []

    print('# ========= start training ========== #')
    for epoch in range(num_epoch):
        netR.train()

        for step, sample in enumerate(train_loader):
            k0, csm = sample
            k0, csm = k0.to(device), csm.to(device)
            mask_device = mask.to(device)

            image = ifft2c(k0)
            label = torch.sum(image * torch.conj(csm), dim=1)

            k0 = k0 * mask_device

            im_dc = ifft2c(k0)
            im_zf = torch.sum(im_dc * torch.conj(csm), dim=1)

            # # ===== Show the parameters in model ===== #
            # for param_tensor in netR.state_dict():
            #     print(param_tensor, '\t', netR.state_dict()[param_tensor].size())
            # # print('/n',netR.net[0].sparseconv_xt.seq_a[0].get_parameter(target="weights"))
            # print('###################### \n ',netR)

            # im_zf_abs = torch.abs(im_zf)
            # label_abs = torch.abs(label)

            # recon = netR(im_zf,csm, k0,mask_device)  # !!! 将dataset输入网络
            recon = netR(k0, csm, mask_device)

            optimR.zero_grad()

            recon_abs = torch.abs(recon)
            loss_mse = mse(recon, label)
            loss_R = loss_mse

            loss_R.backward()
            optimR.step()
            lossR += [loss_R.item()]

            # loss_R is current single sample loss
            # lossR is loss of all samples(loss_R*step or loss_R*sample_number) in current epoch
            if total_step == 0:
                param_num = np.sum([np.prod(v.nelement()) for v in netR.parameters()])

            if (step + 1) % 100 == 0:
                # sio.savemat('PGD_relax.mat', {'csm':csm_fake.cpu().data.numpy(), 'init':x_init.cpu().data.numpy(),
                # 'ref':im_real.cpu().data.numpy(), 'rec':im_fake[-1].cpu().data.numpy()})
                print('Epoch', epoch + 1, '/', num_epoch, 'Step', step + 1, 'LOSS', loss_R.detach().cpu().numpy(),
                      'param_num:',param_num
                      )

        avgLoss = np.mean(lossR)
        Loss_list.append(avgLoss) # to record loss in each epoch
        print('epoch', epoch + 1, 'trnLoss:', avgLoss)
        lossR = []

        scheduler.step()
        for param_group in optimR.param_groups:
            print("\n*learning rate {:.2e}*\n" .format(param_group['lr']))

        # stepG.step()
        if (epoch + 1) % 10 == 0:
            torch.save(netR.state_dict(),
                       "%s/net_params_%d.pkl" % (modeldir, epoch + 1))  # save only the parameters

    ## ========== save for loss curve ========== ##
    loss_curve_filename = 'loss_curve.npy'
    loss_final_dir = os.path.join(loss_dir,loss_curve_filename)
    np.save(loss_final_dir, Loss_list)
    print('loss saved')
    ## ========= Training END ========== ##
    print('Training END')










import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.utils.data as Data
import numpy as np
import scipy.io as sio
import mat73
from model_DUN_SRE import RelaxConvNet
from utils import ifft2c, mse, psnr, ssim_DLESPIRiT
from argparse import ArgumentParser
import hdf5storage
import time

class get_data(Data.Dataset):
    def __init__(self, mat_path):
        self.data_dir = mat_path
        self.sample_list = sorted(os.listdir(self.data_dir))
        self.data_len = len(self.sample_list)

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

    # os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Parameter Definition
    dataset_name = args.data[0].upper()
    batch_size = int(args.batch_size[0])
    num_epoch = int(args.num_epoch[0])
    learning_rate = float(args.learning_rate[0])
    multi_coil = 1
    acc = int(args.acc[0])
    acc_str = str(acc)
    net_name = args.net[0].upper()
    niter = int(args.niter[0])
    model_version_name = args.model_ver_name[0]

    # Load dataset
    mat_dir = './test_dataset/'
    testset = get_data(mat_path = mat_dir)
    test_loader = Data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)
    print('test_dataset loaded.')
    print(net_name,  dataset_name, 'acc_', acc, 'iter', niter)

    # Definite dir
    model_id = net_name + '_' + dataset_name + 'iter'+ args.niter[0].upper() + 'nf32' + '_vista'
    modeldir = os.path.join('models/',model_version_name, model_id)

    # result dir definition
    result_dir = os.path.join('results/stable', model_version_name)
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    # prepare undersampling mask
    mask_dir = './Mask/'
    ## ========= vista ========= #
    mask_path = os.path.join(mask_dir, 'vista_18_192_192_acc_16.mat')
    mask_matlab_data = mat73.loadmat(mask_path)
    mask_t = mask_matlab_data['mask'].astype(np.complex64)
    # ========== mask post processing ========== #
    mask_t = np.expand_dims(mask_t, axis=0)
    mask = torch.from_numpy(mask_t)
    mask = mask.unsqueeze(0)

    # define network
    netR = RelaxConvNet(niter).to(device)

    # load weight from trained model
    pkl_path = modeldir
    weight_file = pkl_path + '/net_params_50.pkl'
    netR.load_state_dict(torch.load(weight_file))
    print('load weight:', modeldir)
    print('network initialized. Model_version_name is :',model_version_name, '\nModel id is', model_id)

    # paramsR = netR.parameters()

    # Iterate over epochs.   # 定义训练参数，初始化
    rec = []
    ref = []
    undersampled = []
    total_loss_mse = []
    total_loss_psnr = []
    total_loss_ssim = []
    mse_list = []
    psnr_list = []
    ssim_list = []

    print('# ========= start testing ========== #')
    netR.eval()
    with torch.no_grad():
        t_start = time.time()
        for step, sample in enumerate(test_loader):
            k0, csm = sample
            k0, csm = k0.to(device), csm.to(device)
            mask_device = mask.to(device)

            image = ifft2c(k0)
            label = torch.sum(image * torch.conj(csm), dim=1)

            k0 = k0 * mask_device  # mask_device

            im_dc = ifft2c(k0)
            im_zf = torch.sum(im_dc * torch.conj(csm), dim=1)
            # im_zf_abs = torch.abs(im_zf)
            # label_abs = torch.abs(label)

            recon = netR(k0, csm, mask_device)
            # recon = netR(im_zf, k0, csm, mask)

            # optimR.zero_grad()

            # recon_abs = torch.abs(recon)
            loss_mse = mse(recon, label)
            loss_psnr = psnr(recon, label)
            loss_ssim = ssim_DLESPIRiT(recon, label)

            mse_list.append(loss_mse.detach().cpu().numpy())
            psnr_list.append(loss_psnr.detach().cpu().numpy())
            ssim_list.append(loss_ssim)

            total_loss_mse += [loss_mse.item()]
            total_loss_psnr += [loss_psnr.item()]
            total_loss_ssim += [loss_ssim.item()]

            # loss_mse = torch.mean(torch.pow(torch.abs(recon - label), 2))
            # print(step, 'mse=',loss_mse.detach().cpu().numpy())

            print('\nsample:', step, '\nmse =', loss_mse.detach().cpu().numpy())
            print('psnr =', loss_psnr.detach().cpu().numpy())
            print('ssim =', loss_ssim)

            # rec_ini = recon.detach().cpu().numpy()
            rec_ini = recon.detach().cpu().numpy()
            rec_ini = np.transpose(rec_ini, (0, 2, 3, 1))
            rec_ini = np.squeeze(rec_ini)

            ref_ini = label.detach().cpu().numpy()
            # ref_ini = label.detach().numpy()
            ref_ini = np.transpose(ref_ini, (0, 2, 3, 1))
            ref_ini = np.squeeze(ref_ini)

            # unders_ini = im_zf.detach().cpu().numpy()
            # unders_ini = np.transpose(unders_ini, (0, 2, 3, 1))
            # unders_ini = np.squeeze(unders_ini)

            rec.append(rec_ini)
            ref.append(ref_ini)
            # undersampled.append(unders_ini)

        t_end = time.time()
        mean_loss_mse = np.mean(total_loss_mse)
        mean_loss_psnr = np.mean(total_loss_psnr)
        mean_loss_ssim = np.mean(total_loss_ssim)
        std_mse = np.std(mse_list)
        std_psnr = np.std(psnr_list)
        std_ssim = np.std(ssim_list)
        print('\nmean_value_of_test_dataset:', '\nmse =', mean_loss_mse, 'mse_std =', std_mse)
        print('psnr =', mean_loss_psnr, 'psnr_std = ', std_psnr)
        print('ssim =', mean_loss_ssim, 'ssim_std = ', std_ssim)
        print('average time', (t_end - t_start) / (step + 1))

        result_file_name = 'x' + acc_str + '_' + model_version_name + '.mat'
        result_file = os.path.join(result_dir, result_file_name)
        datadict = {'recon': np.array(rec)}
        # datadict = {'label': np.array(ref)}
        # datadict = {'undersampled': np.array(undersampled)}
        # sio.savemat(result_file, datadict)

## re-print training and model information
print(net_name,  dataset_name, 'acc_', acc, 'iter', niter)
print('Model_version_name is :',model_version_name, '\nModel id is', model_id)
print('end test')

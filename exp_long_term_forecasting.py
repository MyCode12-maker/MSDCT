from thop import profile

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single
import torch_dct as dct
import torch.nn.functional as F
from layers.DWT_Decomposition import Decomposition
warnings.filterwarnings('ignore')

import torch
import torch.nn.functional as F
import torch_dct as dct
import inspect


def multi_scale_dct_loss(pred, target, scales=(1, 2, 4), weights=None, loss_type="mae"):
    """
    pred/target: [B, pred_len, C]
    Compute multi-scale frequency-domain loss with DCT.
    """
    if weights is None:
        weights = [1.0 / len(scales)] * len(scales)

    total_loss = pred.new_tensor(0.0)

    for scale, weight in zip(scales, weights):
        if pred.size(1) < scale:
            continue

        pred_s = pred
        target_s = target

        if scale > 1:
            pred_s = F.avg_pool1d(
                pred.permute(0, 2, 1),
                kernel_size=scale,
                stride=scale
            ).permute(0, 2, 1)

            target_s = F.avg_pool1d(
                target.permute(0, 2, 1),
                kernel_size=scale,
                stride=scale
            ).permute(0, 2, 1)

        pred_f = dct.dct(pred_s.permute(0, 2, 1), norm="ortho")
        target_f = dct.dct(target_s.permute(0, 2, 1), norm="ortho")

        if loss_type == "mae":
            loss = torch.mean(torch.abs(pred_f - target_f))
        else:
            loss = torch.mean((pred_f - target_f) ** 2)

        total_loss = total_loss + weight * loss

    return total_loss
import torch
import torch.nn.functional as F
import torch_dct as dct


def multi_scale_time_freq_loss(
    pred,
    target,
    scales=(1, 2, 4),
    time_weight=1.0,
    freq_weight=0.1,
    scale_weights=None,
    time_loss_type="mse",
    freq_loss_type="mse",
    downsample_method="avg",
):
    """
    pred/target: [B, pred_len, C]

    Returns:
        total_loss, time_loss, freq_loss
    """
    if scale_weights is None:
        scale_weights = [1.0 / len(scales)] * len(scales)

    time_total = pred.new_tensor(0.0)
    freq_total = pred.new_tensor(0.0)
    criterion1 = nn.L1Loss()
    for scale, scale_weight in zip(scales, scale_weights):
        if pred.size(1) < scale:
            continue

        pred_s = pred
        target_s = target

        if scale > 1:
            pred_ch = pred.permute(0, 2, 1)
            target_ch = target.permute(0, 2, 1)

            if downsample_method == "max":
                pred_s = F.max_pool1d(pred_ch, kernel_size=scale, stride=scale)
                target_s = F.max_pool1d(target_ch, kernel_size=scale, stride=scale)
            else:
                pred_s = F.avg_pool1d(pred_ch, kernel_size=scale, stride=scale)
                target_s = F.avg_pool1d(target_ch, kernel_size=scale, stride=scale)

            pred_s = pred_s.permute(0, 2, 1)
            target_s = target_s.permute(0, 2, 1)

        if time_loss_type == "mae":
            time_loss = criterion1(pred_s ,target_s)
        else:
            time_loss = torch.mean((pred_s - target_s) ** 2)

        pred_f = dct.dct(pred_s.permute(0, 2, 1), norm="ortho")
        target_f = dct.dct(target_s.permute(0, 2, 1), norm="ortho")

        if freq_loss_type == "mae":
            freq_loss = criterion1(pred_f ,target_f)
        else:
            freq_loss = torch.mean((pred_f - target_f) ** 2)

        time_total = time_total + scale_weight * time_loss
        freq_total = freq_total + scale_weight * freq_loss

    total = time_weight * time_total + freq_weight * freq_total
    return total, time_total, freq_total

class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        self.wavelet_name = 'db2'
        self.h_model = int(args.seq_len * 0.2)
        self.Decomposition_model = Decomposition(input_length=args.seq_len,
                                                 pred_length=args.pred_len,
                                                 wavelet_name= self.wavelet_name ,
                                                 level=1,
                                                 batch_size=args.batch_size,
                                                 channel=args.enc_in,
                                                 d_model=args.d_model,
                                                 tfactor=2,
                                                 dfactor=4,
                                                 device= args.device,
                                                 no_decomposition=False,
                                                 use_amp=False)
        self.patch_stride = 24
        self.patch_len = 24

        self.patch_num = int((args.seq_len - self.patch_len) / self.patch_stride + 2)

    def do_patching(self, x):
        x_end = x[:, :, -1:]
        x_padding = x_end.repeat(1, 1, self.patch_stride)
        x_new = torch.cat((x, x_padding), dim=-1)
        x_patch = x_new.unfold(dimension=-1, size=self.patch_len, step=self.patch_stride)
        return x_patch

    def _build_model(self):
        model = self.model_dict[self.args.model](self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _model_regularization_loss(self, pred=None, target=None):
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        if hasattr(model, 'regularization_loss'):
            if len(inspect.signature(model.regularization_loss).parameters) >= 2:
                return model.regularization_loss(pred, target)
            return model.regularization_loss()
        return None

    def _get_profile(self, model):

        import torch
        from thop import profile

        model = model.float().to(self.device)
        model.eval()

        _input = torch.randn(
            self.args.batch_size,
            self.args.seq_len,
            self.args.enc_in
        ).float().to(self.device)

        _input1 = torch.randn(
            self.args.batch_size,
            self.args.seq_len,
            4
        ).float().to(self.device)

        # ---------------------------------------------------
        # FLOPs & Params
        # ---------------------------------------------------
        macs, params = profile(
            model,
            inputs=(_input, _input1, _input1, _input1),
            verbose=False
        )

        # ---------------------------------------------------
        # GPU Memory Profile
        # ---------------------------------------------------
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)

        with torch.no_grad():
            _ = model(_input, _input1, _input1, _input1)

        # 当前显存
        current_mem = torch.cuda.memory_allocated(self.device) / 1024 ** 2

        # 峰值显存
        peak_mem = torch.cuda.max_memory_allocated(self.device) / 1024 ** 2

        print(f'FLOPs: {macs / 1e9:.3f} GFLOPs')
        print(f'Params: {params / 1e6:.3f} M')
        print(f'Current GPU Memory: {current_mem:.2f} MB')
        print(f'Peak GPU Memory: {peak_mem:.2f} MB')

        return macs, params, peak_mem
    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach()
                true = batch_y.detach()

                loss = criterion(pred, true)

                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        self._get_profile(self.model)
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        criterion1 = nn.L1Loss()
        criterion2 = nn.SmoothL1Loss()
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        reg_loss = self._model_regularization_loss(outputs, batch_y)
                        if reg_loss is not None:
                            loss = loss + reg_loss
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    outputs_d = dct.dct(outputs, norm='ortho')
                    batch_y_d = dct.dct(batch_y, norm='ortho')

                    loss = 0.9 * criterion1(outputs, batch_y) + 0.1 * criterion1(outputs_d, batch_y_d)
                    # reg_loss = self._model_regularization_loss(outputs, batch_y)
                    # if reg_loss is not None:
                    #     loss = loss + reg_loss
                    #loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'Not calculated'

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)

        return

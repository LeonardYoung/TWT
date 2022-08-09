import pickle
import numpy as np
import os
import scipy.sparse as sp
import torch
from scipy.sparse import linalg
from sklearn import metrics
from scipy.stats import pearsonr
import random
import generate.water_generate as water_generate
import generate.traffic_generate as traffic_generate

class DataLoaderXY(object):
    def __init__(self, xy, batch_size, pad_with_last_sample=True):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        # self.origin_xs,self.origin_ys = xs,ys
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xy) % batch_size)) % batch_size
            x_padding = np.repeat(xy[-1:], num_padding, axis=0)
            xy = np.concatenate([xy, x_padding], axis=0)
        self.size = len(xy)
        self.num_batch = int(self.size // self.batch_size)
        self.xy = xy

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xy = self.xy[permutation]
        self.xy = xy

    def get_iterator(self):
        self.current_ind = 0
        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                xy_i = self.xy[start_ind: end_ind, ...]
                yield (xy_i)
                self.current_ind += 1
        return _wrapper()


class DataLoader(object):
    def __init__(self, xs, ys,targets, batch_size, pad_with_last_sample=True):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        # self.origin_xs,self.origin_ys = xs,ys
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            tar_padding = np.repeat(targets[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
            targets = np.concatenate([targets, tar_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys
        self.targets = targets

    # def set_xy(self,x,y):
    #     self.xs = x
    #     self.ys = y
    #     self.origin_xs, self.origin_ys = x, y
    #
    # def get_origin(self):
    #     return self.origin_xs,self.origin_ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys, targets = self.xs[permutation], self.ys[permutation], self.targets[permutation]
        self.xs = xs
        self.ys = ys
        self.targets = targets

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                tar_i = self.targets[start_ind: end_ind, ...]
                yield (x_i, y_i, tar_i)
                self.current_ind += 1

        return _wrapper()

class StandardScaler():
    """
    Standard the input
    dimen_size:维度大小，一般为1
    """

    def __init__(self,dimen_size, mean, std):
        self.mean = mean
        self.std = std
        self.dimen_size = dimen_size

    def transform(self, data):
        # 单维
        if self.dimen_size == 1:
            return (data - self.mean[0]) / self.std[0]
        # 多维
        resule_list = []
        for i in range(self.dimen_size):
            mean_reslut = (data[..., i] - self.mean[i]) / self.std[i]
            resule_list.append(np.expand_dims(mean_reslut, axis=-1))
        result = np.concatenate(resule_list, axis=-1)
        return result

    # 只反标准化第一个维度
    def inverse_transform_first_dimen(self, data):
        return (data * self.std[0]) + self.mean[0]

    def inverse_transform(self, data):
        # 单维
        if self.dimen_size == 1:
            return (data * self.std[0]) + self.mean[0]
        # 多维
        std_tensor = torch.ones_like(data, requires_grad=False)
        mean_tensor = torch.zeros_like(data, requires_grad=False)
        for i in range(self.dimen_size):
            std_tensor[:, i, :, :] = self.std[i]
            mean_tensor[:, i, :, :] = self.mean[i]
        return (data * std_tensor) + mean_tensor





#   输出数据四个维度 [batch,seqence,node,feature]
def load_dataset_from_generate(datafile,fac_idx,batch_size = 64,
                               transf=True,seq_length_xy=32,dataset='water'):
    if dataset == 'water':
        xy_train,  xy_val,  xy_test = water_generate.generate_dataset(
            datafile,fac_idx, out_dir='', seq_length_xy=seq_length_xy)
    elif dataset == 'traffic':
        xy_train, xy_val, xy_test = traffic_generate.generate_train_val_test(
            seq_length_xy=seq_length_xy,traffic_df_filename=datafile)

    data = {}

    for category in ['train', 'val', 'test']:
        # cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['xy_' + category] = locals()['xy_'+category]
        # data['y_' + category] = locals()['y_'+category]
        # data['target_' + category] = locals()['target_'+category]

    xy_train = data['xy_train']
    mean_list = []
    std_list = []
    dimen_size = xy_train.shape[3]
    for i in range(dimen_size):
        mean_list.append(xy_train[..., i].mean())
        std_list.append(xy_train[..., i].std())

    scaler = StandardScaler(dimen_size=dimen_size, mean=mean_list, std=std_list)

    # 标准化
    if transf:
        for category in ['train', 'val', 'test']:
            # if dimen_size == 2:
            data['xy_' + category] = scaler.transform(data['xy_' + category])
            # data['target_' + category] = scaler.transform(data['target_' + category])
            # else:
            #     data['x_' + category][..., 0:-1] = scaler.transform(data['x_' + category][..., 0:-1])
            #     data['target_' + category][..., 0:-1] = scaler.transform(data['target_' + category][..., 0:-1])

    data['train_loader'] = DataLoaderXY(data['xy_train'], batch_size)
    data['val_loader'] = DataLoaderXY(data['xy_val'], batch_size)
    data['test_loader'] = DataLoaderXY(data['xy_test'], batch_size)
    data['scaler'] = scaler
    return data

def load_dataset(dataset_dir, batch_size = 64, valid_batch_size= 64, test_batch_size=64, transf=True):

    data = {}

    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']

    x_train = data['x_train']
    mean_list = []
    std_list = []
    dimen_size = x_train.shape[3] - 1 if len(x_train.shape) > 3 else 1
    for i in range(dimen_size):
        mean_list.append(x_train[..., i].mean())
        std_list.append(x_train[..., i].std())

    scaler = StandardScaler(dimen_size=dimen_size, mean=mean_list, std=std_list)

    # 标准化
    if transf:
        for category in ['train', 'val', 'test']:
            if dimen_size == 2:
                data['x_' + category] = scaler.transform(data['x_' + category])
            else:
                data['x_' + category][..., 0:-1] = scaler.transform(data['x_' + category][..., 0:-1])

    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], valid_batch_size)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size)
    data['scaler'] = scaler
    return data

def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def MAE_loss(y_pred, target):
    y_pred =  y_pred.squeeze()
    target = target.squeeze()
    loss = torch.abs(y_pred-target)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def metric(pred, real):
    mae = masked_mae(pred,real,0.0).item()
    mape = masked_mape(pred,real,0.0).item()
    rmse = masked_rmse(pred,real,0.0).item()
    return mae,mape,rmse


def np_rmspe(y_true, y_pred):
    loss = np.sqrt(np.mean(np.square(((y_true - y_pred) / (y_true + np.mean(y_true)))), axis=0))
    return loss


# 计算mape，公式经过更改。每次除以标签值加上所有标签的均值，最后结果乘以2
def np_mape(y_true, y_pred):
    loss = np.abs(y_true - y_pred) / (y_true + np.mean(y_true))
    return np.mean(loss) * 2


def metrix_six(y_pred,y_test):
    mae_list = []
    mape_list = []
    rmse_list = []
    rmspe_list = []
    r2_list = []
    r_list = []

    for step in range(3):
        y_test_t = y_test[:,step]
        y_pred_t = y_pred[:,step]

        mae = metrics.mean_absolute_error(y_test_t, y_pred_t)
        # mape = metrics.mean_absolute_percentage_error(y_test_t, y_pred_t)
        mape = np_mape(y_test_t, y_pred_t)
        rmse = metrics.mean_squared_error(y_test_t, y_pred_t) ** 0.5
        rmspe = np_rmspe(y_test_t, y_pred_t)
        # rmspe2 = masked_rmspe(y_test_t,y_pred_t)
        r2 = metrics.r2_score(y_test_t, y_pred_t)
        r = pearsonr(y_test_t, y_pred_t)[0]


        # break
        # break

        mae_list.append(mae)
        mape_list.append(mape)
        rmse_list.append(rmse)
        rmspe_list.append(rmspe)
        r2_list.append(r2)
        r_list.append(r)

        # 计算平均值
    print(f'平均,{np.mean(mae_list):.3f},{np.mean(mape_list):.3f},'
          f'{np.mean(rmse_list):.3f},{np.mean(rmspe_list):.3f},'
          f'{np.mean(r2_list):.3f},{np.mean(r_list):.3f}')
    # mae = metrics.mean_absolute_error(y_test_t, y_pred_t)
    # mape = np_mape(y_test_t, y_pred_t)
    # rmse = metrics.mean_squared_error(y_test_t, y_pred_t) ** 0.5
    # rmspe = np_rmspe(y_test_t, y_pred_t)
    # r2 = metrics.r2_score(y_test_t, y_pred_t)
    # r = pearsonr(y_test_t, y_pred_t)[0]
    return np.mean(mae_list),np.mean(mape_list),np.mean(rmse_list),\
           np.mean(rmspe_list),np.mean(r2_list),np.mean(r_list),


def set_seed(seed):
    os.environ['PYTHONHASHSEED']=str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


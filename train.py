import torch
import os
import utils
import config as Config
import torch.nn as nn
from model import TimeSeriesTransformer
import numpy as np
import earlystopping
import datetime

loss_fn = utils.MAE_loss
# optimizer = torch.optim.SGD(ttModel.parameters(), lr=1e-3)


# node_idx = -1
# 切分时间序列，
def split_seqence(xy,step):
    xy = torch.Tensor(xy).to(Config.device)
    # 错位1
    # x = xy[:,:Config.seq_len_x,...]
    # target_in = xy[:,Config.seq_len_x - 1:-1,...]
    # y = xy[:,Config.seq_len_x:,...]

    # 只预测第step 个步长；方式一
    # x = xy[:, :Config.seq_len_x, ...]
    # a = Config.seq_len_x - 1
    # b = -Config.seq_len_forcast
    # target_in = xy[:, a:b, ...]
    # if b+step+1 == 0:
    #     y = xy[:, a + step + 1:, ...]
    # else:
    #     y = xy[:, a+step+1:b+step+1, ...]

    # # 只预测第step 个步长；方式二
    x = xy[:, :Config.seq_len_x, ...]
    a = 1
    b = -Config.seq_len_forcast
    target_in = xy[:, a:b, ...]
    if b + step + 1 == 0:
        y = xy[:, a + step + 1:, ...]
    else:
        y = xy[:, a + step + 1:b + step + 1, ...]

    # 直接训练多步序列
    # a = Config.seq_len_a
    # b = Config.seq_len_a + Config.seq_len_b
    # x = xy[:,:a,...]
    # target_in = xy[:,a:b,...]
    # y = xy[:,b:,...]

    # 图数据直接当做嵌入
    x = x[..., 0].squeeze()
    y = y[..., 0].squeeze()
    target_in = target_in[..., 0].squeeze()

    return x,target_in,y

def train(dataloader, scale, model,  optimizer,step):
    # size = len(dataloader.dataset)
    model.train()
    loss_epoch = []
    dataloader.shuffle()
    for batch, xy in enumerate(dataloader.get_iterator()):
        x, y, target_in = split_seqence(xy,step)

        # Compute prediction error
        pred = model((x,target_in))
        pred = scale.inverse_transform_first_dimen(pred)
        y = scale.inverse_transform_first_dimen(y)

        if Config.loss_only_last_step:
            loss = loss_fn(pred.squeeze()[:,-1,:], y[:,-1,:])
        else:
            loss = loss_fn(pred.squeeze(), y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_epoch.append(loss.item())
    return np.mean(loss_epoch)


def validate(dataloader,scale, model,step):
    # size = len(dataloader.dataset)
    model.eval()
    loss_epoch = []
    with torch.no_grad():
        for batch, xy in enumerate(dataloader.get_iterator()):
            x, y, target_in = split_seqence(xy,step)

            # Compute prediction error
            # pred = model.eval_forward(x,Config.seq_len_y)
            pred = model((x, target_in))
            pred = scale.inverse_transform_first_dimen(pred)
            y = scale.inverse_transform_first_dimen(y)
            # 只有最后一个预测有效，因为前面的预测在target_in中已输入
            loss = loss_fn(pred[:,-1,:], y[:,-1,:])

            loss_epoch.append(loss.item())
    return np.mean(loss_epoch)


def test(dataloader,scale, model, model_save_path,step):
    # size = len(dataloader.dataset)
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    loss_epoch = []
    yhats = []
    reallys =[]
    with torch.no_grad():
        for batch, xy in enumerate(dataloader.get_iterator()):
            x, y, target_in = split_seqence(xy,step)

            # Compute prediction error
            # pred = model.eval_forward(x,target_in,Config.seq_len_y)
            pred = model((x, target_in))
            pred = scale.inverse_transform_first_dimen(pred)
            y = scale.inverse_transform_first_dimen(y)
            # 只有最后一个预测有效，因为前面的预测在target_in中已输入?
            loss = loss_fn(pred[:,-1,:], y[:,-1,:])
            yhats.append(pred.squeeze())
            reallys.append(y.squeeze())

            loss_epoch.append(loss.item())

    yhats = torch.cat(yhats, dim=0)
    reallys = torch.cat(reallys, dim=0)

    # for idx in range(yhats.shape[1]):
    #     loss0 = loss_fn(yhats[:, idx, :], reallys[:, idx, :])
    #     print(f"test fac:{Config.fac_idx} step:{idx}   MAE:{loss0} ")

    yhats = yhats.to('cpu').numpy()
    reallys = reallys.to('cpu').numpy()

    # 保存预测结果到本地
    np.savez_compressed(
        os.path.join(Config.save_root, f"out{step}.npz"),
        y_pred=yhats,
        y_test=reallys
    )
    print(os.path.join(Config.save_root, f"out{step}.npz"))

    return np.mean(loss_epoch)


def one_step(step,dataloader):
    utils.set_seed(42)
    model_save_path = f'{Config.save_root}/model{step}.pth'
    early_stopping = earlystopping.EarlyStopping(patience=Config.patience, path=model_save_path,
                                                 verbose=True)

    ttModel = TimeSeriesTransformer(n_encoder_inputs=Config.num_node,
                                    n_decoder_inputs=Config.num_node,
                                    out_dim=Config.num_node,
                                    channels=Config.d_model).to(Config.device)
    # 使用上一个步长的数据 
    if step != 0 and Config.use_last_train:
        last_model_save_path = f'{Config.save_root}/model{step-1}.pth'
        ttModel.load_state_dict(torch.load(last_model_save_path))
    # print(ttModel)
    optimizer = torch.optim.Adam(ttModel.parameters(), lr=1e-5)

    start_time = datetime.datetime.now()
    for t in range(Config.epoch):
        train_loss = train(dataloader['train_loader'], dataloader['scaler'], ttModel, optimizer,step)
        val_loss = validate(dataloader['val_loader'], dataloader['scaler'], ttModel,step)
        # print(f"train_loss:{train_loss:.4f}  ")
        print(f"fac: {Config.fac_idx}  step:{step}   Epoch {t + 1}  train_loss:{train_loss:.4f}    val_loss:{val_loss:.4f}")
        early_stopping(val_loss, ttModel)
        if early_stopping.early_stop:
            print(f"fac: {Config.fac_idx}  step:{step}  Early stopping. actual epoch={t+1}")
            break
        # print("\n")
    test_loss = test(dataloader['test_loader'], dataloader['scaler'], ttModel, model_save_path,step)

    end_time = datetime.datetime.now()
    print(f"fac: {Config.fac_idx}  step:{step}  test_loss:{test_loss:.4f}   usage time:{end_time - start_time}")
    return test_loss,end_time - start_time


def run_once():
    utils.set_seed(42)
    torch.cuda.set_device(Config.gpu_num)
    dataloader = utils.load_dataset_from_generate(Config.data_file, Config.fac_idx,dataset=Config.dataset,
                                                  seq_length_xy=Config.seq_len_xy, batch_size=64, transf=True)

    print(f"load dataset done \n shape of xy_train:{dataloader['xy_train'].shape}:")

    Config.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {Config.device} device")


    if not os.path.exists(Config.save_root):
        os.makedirs(Config.save_root)

    # for step in [2]:
    tests = []
    total_time = None
    for step in range(Config.seq_len_forcast):
        # 跳过最后一次预测
        if step != Config.seq_len_forcast - 1:
            l,t = one_step(step,dataloader)

            # 汇总
            tests.append(l)
            total_time = t if total_time is None else total_time+t

    tests = np.mean(tests)
    print(f"mean test loss={tests:.3f}, total time cost:{total_time}")




if __name__ == '__main__':

    run_once()
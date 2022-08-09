
# basic config

patience = 50
gpu_num = 1
device = f"cuda:{gpu_num}"
epoch = 2000

tanser_after_model = True # y是否反标准化

encoder_only = False  # 是否只包含编码器（去掉解码器）(效果很差）
swap_dimen = False      # 交换维度。默认False，此时一个时间点上的所有节点数据相当于一个词嵌入。
loss_only_last_step = True # 训练的损失计算时，只计算最后一个步长
mask_on_decoder = True # 解码器是否有掩码
transformer_of_torch = True # 使用torch中的transformer

use_last_train = True # transfer learning

num_layers = 2 # transformer layers
d_model = 512 # transformer param
n_head = 8  # transformer param
# eval_type = 'a' # 自回归，target初始值是src的最后一个数据
# eval_type = 'half_in' # 自回归，target初始值有一半是已知的
# eval_type = 'back_encoder'
# eval_type = 'truth_regression' #使用真实值进行自回归
# eval_type = 'test'

# ######## water config
dataset = 'water'
# data_file = "data/changtai_4H.csv"
data_file = "data/water_4H.csv"
# num_node = 7
num_node = 10
fac_idx = 8
seq_len_xy = 33
seq_len_x = 23
seq_len_forcast = 9
# pre_fix='re_train_24_8_8'
pre_fix=f'basic_{num_layers}_{d_model}_{n_head}'
save_root = f"./result/water/{pre_fix}_{fac_idx}"

# ######## traffic metr  config
# dataset = 'traffic'
# data_file= "data/traffic/metr-la.h5"
# num_node = 207
# seq_len_xy = 27
# seq_len_x = 12
# seq_len_forcast = 13
# fac_idx = 0
# # pre_fix='pure' # 模型代码没有改动
# # pre_fix='split_after' # 在数据进入模型前才拆分x,y,target_in
# pre_fix='basic' # 临时测试
# # pre_fix='transfer_4layer'
# save_root = f"./result/{dataset}/{pre_fix}_metr_12_13_13"

# ######## traffic pems  config
# dataset = 'traffic'
# data_file= "data/traffic/pems-bay.h5"
# num_node = 325
# seq_len_xy = 27
# seq_len_x = 12
# seq_len_forcast = 13
# fac_idx = 0
# # pre_fix='temp' # 临时测试
# pre_fix='basic'
# save_root = f"./result/{dataset}/{pre_fix}_pems_12_13_13"

print(f"dataset={dataset} ;data_file={data_file}; fac_idx={fac_idx};\n"
      f" num_layers={num_layers};d_model={d_model};n_head={n_head}\n"
      f" use_last_train={use_last_train};swap_dimen={swap_dimen};loss_only_last_step={loss_only_last_step};mask_on_decoder={mask_on_decoder}"
      f" seq_len_xy={seq_len_xy};"
      f"seq_len_x={seq_len_x}; seq_len_forcast={seq_len_forcast}"
      # f"seq_len_a={seq_len_a}; seq_len_b={seq_len_b};"
      f" save_root={save_root}\n\n")
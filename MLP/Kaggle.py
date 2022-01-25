import hashlib
import os
import tarfile
import zipfile

import fnet_modules
import pandas
import requests
import pandas as pd
import torch
import math
from torch import nn
from torch.utils.data import DataLoader
from d2l import torch as d2l

# @save
DATA_HUB = dict()  # 数据集-下载地址映射二元组
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'


def download(name, cache_dir=os.path.join('..', 'data')):  # @save
    """下载一个DATA_HUB中的文件，返回本地文件名"""
    # 判断是否是DATA_HUB中有的数据集
    assert name in DATA_HUB, f"{name}不存在于{DATA_HUB}"
    # 提取数据集的URL以及对应的sha-1密钥
    url, sha1_hash = DATA_HUB[name]
    # 如果cache_dir不存在就创建该文件夹
    os.makedirs(cache_dir, exist_ok=True)
    # URL按'/'分割得到的结尾作为cache_dir下数据集子文件夹的名称
    fname = os.path.join(cache_dir, url.split('/')[-1])
    # 如果文件已经存在就核验sha-1密钥
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        # 打开文件进行读取
        with open(fname, 'rb') as f:
            # 一直读到数据结束
            while True:
                data = f.read(1048576)
                # 没有数据了就退出while循环
                if not data:
                    break
                # 向sha1传入数据内容
                sha1.update(data)
            # 如果sha1生成的摘要与目标摘要一致则核验成功，退出程序
            # 否则进入后续的下载流程
            if sha1.hexdigest() == sha1_hash:
                return fname

    # 如果文件不存在就从URL下载数据集
    print(f"正在从{url}下载数据集{name}...")
    # 数据集下载
    r = requests.get(url, stream=True, verify=True)
    # 往fname记录的文件路径载入数据
    with open(fname, 'wb') as f:
        f.write(r.content)
    # 返回本地文件名
    return fname


def download_extract(name, folder=None):
    """下载并解压zip/tar文件"""
    # 数据集下载，返回下载的压缩文件的本地文件名
    fname = download(name)
    # 获取压缩文件所在目录（即去掉fname中的文件名）
    base_dir = os.path.dirname(fname)
    # 获取分离压缩文件后缀名
    # data_dir即提取结果所在的文件夹目录，ext为文件名
    data_dir, ext = os.path.splitext(base_dir)
    # 解压缩
    if ext == '.zip':
        fp = zipfile.ZipFile(data_dir, 'r')
    elif ext == '.tar':
        fp = tarfile.open(data_dir, 'r')
    else:
        assert False, "只有zip/tar文件可以被解压"
    # 将解压结果写入到base_dir指定的路径(即与压缩包同路径)
    fp.extractall(base_dir)
    # 返回解压结果所在位置
    return os.path.join(base_dir, folder) if folder else data_dir


def download_all():
    """下载DATA_HUB中的所有文件"""
    for name in DATA_HUB:
        download(name)


def get_net():
    # num_inputs = training_features.shape[1]
    # net = nn.Sequential(nn.Linear(num_inputs, 1))

    # 改进版网络
    num_inputs = training_features.shape[1]
    num_hidden = 1024
    dropout_rate = 0.5
    net = fnet_modules.FLock(num_inputs, 1)
    # net = nn.Sequential(
    #     nn.Linear(num_inputs, num_hidden),
    #     nn.ReLU(),
    #     nn.Dropout(dropout_rate),
    #     nn.Linear(num_hidden, 1)
    # )

    return net


def log_rmse(net, features, labels):
    """计算相对误差损失"""
    # 考虑到房价的绝对值非常大,因此在对数放缩后再计算MSE
    # 为了在取对数时进一步稳定结果,将预测值中小于1的值转换为1
    net.eval()
    with torch.no_grad():
        preds = net(features)
    clamped_preds = torch.clamp(preds, min=1).float()
    rmse = torch.norm(torch.log(clamped_preds) - torch.log(labels)) / math.sqrt(labels.shape[0])

    # 直接预测价格的对数
    # preds = net(features)
    # rmse = torch.norm(preds - torch.log(labels)) / math.sqrt(labels.shape[0])

    return rmse


def init_weights(m):
    if type(m) == nn.Linear:
        # 选择了较大的标准差，避免因初始权重太小导致预测值结果集中在0附近而影响收敛
        nn.init.normal_(m.weight, std=10)


def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    # 使用Adam算法进行优化
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss = nn.MSELoss()
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            # l = loss(net(X), torch.log(y))  # 直接预测价格的对数
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels).detach().numpy())
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels).detach().numpy())

    return train_ls, test_ls


def get_k_fold_data(k, i, X, y):
    assert k > 1 and 0 <= i < k
    fold_size = int(X.shape[0] / k)
    begin_index = i * fold_size
    end_index = min((i + 1) * fold_size, X.shape[0])
    X_valid = X[begin_index:end_index]
    y_valid = y[begin_index:end_index]
    X_train = torch.cat((X[:begin_index], X[end_index:]))
    y_train = torch.cat((y[:begin_index], y[end_index:]))

    return X_train, y_train, X_valid, y_valid


def k_fold(k, train_features, train_labels, num_epochs,
           learning_rate, weight_decay, batch_size):
    train_ls_sum, valid_ls_sum = 0, 0
    for i in range(k):
        net = get_net()
        # net.apply(init_weights)
        X_train, y_train, X_valid, y_valid = get_k_fold_data(k, i, train_features, train_labels)
        train_ls, valid_ls = train(net, X_train, y_train, X_valid, y_valid,
                                   num_epochs, learning_rate, weight_decay, batch_size)
        train_ls_sum += train_ls[-1]
        valid_ls_sum += valid_ls[-1]
        if i == 0:
            d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
                     xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
                     legend=['train', 'valid'], yscale='log')
            d2l.plt.show()
        print(f"折{i + 1},训练log rmse: {float(train_ls[-1]):f}, 验证log rmse: {float(valid_ls[-1]):f}")

    return train_ls_sum / k, valid_ls_sum / k


def train_and_predict(train_features, train_labels, test_features, test_data,
                      num_epochs, learning_rate, weight_decay, batch_size):
    net = get_net()
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, learning_rate, weight_decay, batch_size)
    d2l.plot(list(range(1, num_epochs + 1)), [train_ls],
             xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
             legend=['train', 'valid'], yscale='log')
    d2l.plt.show()
    print(f'训练log rmse：{float(train_ls[-1]):f}')
    # 用训练好的网络进行预测
    net.eval()
    with torch.no_grad():
        preds = net(test_features).detach().numpy()
    submission = pd.DataFrame(preds)
    submission.columns = ['SalePrice']
    submission = pd.concat([test_data['Id'], submission], axis=1)
    submission.to_csv("submission.csv", index=False)


"""数据集下载与查看"""
# 建立数据集名称-下载地址的映射
DATA_HUB['kaggle_house_train'] = (  # @save
    DATA_URL + 'kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce')
DATA_HUB['kaggle_house_test'] = (  # @save
    DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')

# Kaggle房价预测数据集为csv文件，即以纯文本形式存储的表格数据，
# 可以使用pandas进行读取和大规模批量处理
training_data = pd.read_csv(download('kaggle_house_train'))
test_data = pd.read_csv(download('kaggle_house_test'))

# training_data在最后一列附有房屋价格的标签，因此比test_data多一列
# print(training_data.shape)
# print(test_data.shape)
#
# print(training_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])
# print(test_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])

"""数据预处理"""
# 舍弃没有意义的"Id"信息, 串联训练与测试数据的特征进行统一预处理
# training_data = training_data.drop(['Id'], axis=1)
# test_data = test_data.drop(['Id'], axis=1)
all_features = pd.concat((training_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
print(all_features.shape)
print(all_features.dtypes)

# 对数值数据进行标准化
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
for i in numeric_features:
    # 注意all_features[i]提取出来的列不再是DataFrame而是Series对象
    # 对应求均值和标准差的操作略有不同
    s = all_features[i]
    miu = all_features[i].mean()
    sigma = all_features[i].std()
    all_features[i] = (all_features[i] - miu) / sigma
# 将所有na赋值为标准化后的均值0
all_features[numeric_features] = all_features[numeric_features].fillna(0)

# 对非数值数据进行one-hot编码
all_features = pd.get_dummies(all_features, dummy_na=True)
print(all_features.shape)

train_num = training_data.shape[0]
training_features = torch.tensor(all_features[:train_num].values, dtype=torch.float32)
test_features = torch.tensor(all_features[train_num:].values, dtype=torch.float32)
train_labels = torch.tensor(training_data.iloc[:, -1], dtype=torch.float32).reshape(-1, 1)

"""超参数设置"""
batch_size = 256
base_lr = 0.005
weight_decay = 0.0001
num_epochs = 50
k = 5

"""K折验证"""
train_l, valid_l = k_fold(k, training_features, train_labels,
                          num_epochs, base_lr, weight_decay, batch_size)
print(f"{k}折验证,平均训练log rmse:{float(train_l):f}, 验证log rmse:{float(valid_l):f}")

"""训练与预测结果保存"""
# train_and_predict(training_features, train_labels, test_features, test_data,
#                   num_epochs, base_lr, weight_decay, batch_size)

"""作业题解答"""
# 1. 提交后得到的分数为rmse = 0.18566
# 2. 当选择直接预测价格的对数而不是价格本身时，预测结果的期望会有显著的下降，原来的学习率就会显得太大了，导致loss出现剧烈抖动，
# 甚至无法收敛。因此将学习率下调为0.01，此时再进行K-折交叉验证，得到验证集上平均的log rmse=0.1536
# 3. 并不总是好的，有时丢失的恰恰是与均值相差很远的一些极端值，比如LotFrontage，即房屋到街道的直接距离，缺失的往往是那些距离
# 街道较远、中间间隔太多建筑物的房屋，而没有缺失的则往往本身离街道近，因此用非缺失值的均值来代替可能会严重低估原始结果，从而影响结果
# 4. 调整batch size为256，训练400个epoch，得到验证集的log rmse=0.169，在Kaggl官网上得到的分数
# 为0.16446，相比于优化前下降了约0.02
# 5. 增加一个非线性隐藏层，并启用dropout(rate=0.5)及weight decay(rate=0.0001)，训练30个epoch，得到rmse=0.1384，
# 提交后得到分数rmse = 0.12894，又有了进一步的提升
# 6. 那么不同特征的数据尺度会有巨大差异，举个简单的例子，z=ax+by，如果x比y的大几个数量级，那么显然预测结果会被x主导，而可能
# 同样十分重要的特征y几乎无法影响预测值；此外dz/da=x, dz/db=y，也可以看出a可以有效更新，但b的更新幅度则小的多，反过来又进一步
# 扩大了x的影响，因此不做标准化会直接导致结果被几个数量级大的特征主导，但这些特征的相关性可能并没有数量级小的特征好，从而劣化预测性能

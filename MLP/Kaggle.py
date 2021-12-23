import hashlib
import os
import tarfile
import zipfile
import requests
import pandas as pd
import torch
from torch.utils.data import DataLoader

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
print(training_data.shape)
print(test_data.shape)

print(training_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])
print(test_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])

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
    all_features[i] = (all_features[i]-miu)/sigma
# 将所有na赋值为标准化后的均值0
all_features[numeric_features] = all_features[numeric_features] .fillna(0)

# 对非数值数据进行one-hot编码
all_features = pd.get_dummies(all_features, dummy_na=True)
print(all_features.shape)

train_num = training_data.shape[0]
training_features = torch.tensor(all_features[:train_num].values, dtype=torch.float32)
test_features = torch.tensor(all_features[train_num:].values, dtype=torch.float32)
train_labels = torch.tensor(training_data.iloc[:, -1], dtype=torch.float32).reshape(-1, 1)
# batch_size = 256
# num_workers = 4
#
# train_iter = DataLoader(training_data, batch_size, shuffle=True, num_workers=num_workers)
# test_iter = DataLoader(test_data, batch_size, shuffle=True, num_workers=num_workers)

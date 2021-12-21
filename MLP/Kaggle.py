import hashlib
import os
import tarfile
import zipfile
import requests

# @save
DATA_HUB = dict()  # 数据集映射二元组
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
        with open(fname, 'r') as f:
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
    """下载一个DATA_HUB中的所有文件"""
    for name in DATA_HUB:
        download(name)

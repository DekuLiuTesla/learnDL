import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l


# 定义模型
def lenet(X, params):
    h1_conv = F.conv2d(input=X, weight=params[0], bias=params[1])
    h1_activation = F.relu(h1_conv)
    h1 = F.avg_pool2d(input=h1_activation, kernel_size=(2, 2), stride=(2, 2))
    h2_conv = F.conv2d(input=h1, weight=params[2], bias=params[3])
    h2_activation = F.relu(h2_conv)
    h2 = F.avg_pool2d(input=h2_activation, kernel_size=(2, 2), stride=(2, 2))
    h2 = h2.reshape((h2.shape[0], -1))
    h3_linear = torch.mm(h2, params[4]) + params[5]
    h3 = F.relu(h3_linear)
    y_hat = torch.mm(h3, params[6]) + params[7]
    return y_hat


def get_params(params, device):
    new_params = [p.to(device) for p in params]
    for p in new_params:
        p.requires_grad_()
    return new_params


def allreduce(data):
    """收集数据到data[0]所在设备上并求和，随后分发到其他设备上"""
    for i in range(1, len(data)):
        data[0][:] += data[i].to(data[0].device)
    for i in range(1, len(data)):
        data[i][:] = data[0].to(data[i].device)


def split_batch(X, y, devices):
    """同时对数据和标签进行拆分"""
    assert X.shape[0] == y.shape[0]
    return (nn.parallel.scatter(X, devices),
            nn.parallel.scatter(y, devices))


def train_batch(X, y, net, device_params, loss, devices, lr):
    X_split, y_split = split_batch(X, y, devices)
    # 在每个GPU上分别计算损失
    ls = [loss(net(X_shards, device_W), y_shards).sum()
          for X_shards, y_shards, device_W in zip(X_split, y_split, device_params)]
    for l in ls:
        l.backward()
    # 对每个参数，收集每个GPU上的梯度，求和后分发
    with torch.no_grad():
        for i in range(len(device_params[0])):
            # 汇总每个GPU上的参数，送入allreduce进行合并分发
            allreduce([device_params[c][i].grad for c in range(len(device_params))])
    # 在每个GPU上分别更新模型参数
    for param in device_params:
        d2l.sgd(param, lr, X.shape[0])


def train(net, loss, params, num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    devices = [d2l.try_gpu(i) for i in range(num_gpus)]
    # 将模型参数复制到num_gpus个GPU
    device_params = [get_params(params, d) for d in devices]
    num_epochs = 10
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    timer = d2l.Timer()
    for epoch in range(num_epochs):
        timer.start()
        for X, y in train_iter:
            # 为单个小批量执行多GPU训练
            train_batch(X, y, net, device_params, loss, devices, lr)
            # 等待所有GPU上的训练及梯度反传完成
            torch.cuda.synchronize()
        timer.stop()
        # 在GPU0上评估模型
        # animator.add(epoch + 1, (d2l.evaluate_accuracy_gpu(
        #     lambda x: lenet(x, device_params[0]), test_iter, devices[0]),))
        # 多GPU测试
        animator.add(epoch + 1, evaluate_accuracy_multi_gpu(net, test_iter, device_params, devices), )

    print(f'测试精度：{animator.Y[0][-1]:.2f}，{timer.avg():.1f}秒/轮，'
          f'在{str(devices)}')


def evaluate_accuracy_multi_gpu(net, data_iter, device_params, devices=None):
    """Compute the accuracy for a model on a dataset using a GPU.

    Defined in :numref:`sec_lenet`"""
    if isinstance(net, nn.Module):
        net.eval()  # Set the model to evaluation mode
        if not devices:
            devices = [next(iter(net.parameters())).device]

    # No. of correct predictions, no. of predictions
    num_devices = len(devices)
    metrics = [d2l.Accumulator(2), ] * num_devices

    with torch.no_grad():
        for X, y in data_iter:
            X_split, y_split = split_batch(X, y, devices)
            # 在每个GPU上分别计算进行统计
            for i, (X_shards, y_shards, device_W) in enumerate(zip(X_split, y_split, device_params)):
                metrics[i].add(d2l.accuracy(net(X_shards, device_W), y_shards), d2l.size(y_shards))

        # 在每个GPU上分别计算指标
        result = metrics[0][0] / metrics[0][1]
        for i in range(1, num_devices):
            result += metrics[i][0] / metrics[i][1]

    return result / num_devices


# 使用交叉熵损失
loss = nn.CrossEntropyLoss(reduction='none')
# 初始化模型参数
scale = 0.01
W1 = torch.randn(size=(20, 1, 3, 3)) * scale
b1 = torch.zeros(20)
W2 = torch.randn(size=(50, 20, 5, 5)) * scale
b2 = torch.zeros(50)
W3 = torch.randn(size=(800, 128)) * scale
b3 = torch.zeros(128)
W4 = torch.randn(size=(128, 10)) * scale
b4 = torch.zeros(10)
params = [W1, b1, W2, b2, W3, b3, W4, b4]

new_params = get_params(params, d2l.try_gpu(0))
print('b1 权重:', new_params[1])
print('b1 梯度:', new_params[1].grad)

data = [torch.ones((1, 2), device=d2l.try_gpu(i)) * (i + 1) for i in range(2)]
print('allreduce之前：\n', data[0], '\n', data[1])
allreduce(data)
print('allreduce之后：\n', data[0], '\n', data[1])

data = torch.arange(20).reshape(4, 5)
devices = [torch.device('cuda:0'), torch.device('cuda:1')]
# 将数据均等地分配在各个GPU设备上
split = nn.parallel.scatter(data, devices)
print("input: ", data)
print("load into ", devices)
print("output: ", split)

train(lenet, loss, params, num_gpus=2, batch_size=256, lr=0.3)
d2l.plt.show()

"""作业题解答"""
# 1. 将batch size调整为k*b=512，则
# 调整前：测试精度：0.80，3.0秒/轮，在[device(type='cuda', index=0), device(type='cuda', index=1)]
# 调整后：测试精度：0.72，2.0秒/轮，在[device(type='cuda', index=0), device(type='cuda', index=1)]
# 可见测试速度显著提高，但由于batch size增大，相应参数更新次数会减小，因而精度有所下降

# 2. 对lr进行相应的调整，则
# lr=0.2：测试精度：0.84，2.9秒/轮，在[device(type='cuda', index=0), device(type='cuda', index=1)]
# lr=0.3：测试精度：0.84，2.8秒/轮，在[device(type='cuda', index=0), device(type='cuda', index=1)]
# lr=0.4：测试精度：0.83，2.9秒/轮，在[device(type='cuda', index=0), device(type='cuda', index=1)]
# lr=0.5：测试精度：0.86，2.8秒/轮，在[device(type='cuda', index=0), device(type='cuda', index=1)]
# 可见GPU数量增倍，学习率也增加到相应的倍数

# 3. 略（并无更好的见解）

# 4. 参见函数evaluate_accuracy_multi_gpu

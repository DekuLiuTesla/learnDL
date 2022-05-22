import torch
from d2l import torch as d2l

devices = d2l.try_all_gpus()


def run(x):
    return [x.mm(x) for _ in range(50)]


def copy_to_cpu(x, non_blocking=False):
    return [y.to('cpu', non_blocking=non_blocking) for y in x]


x_gpu1 = torch.rand((4000, 4000), device=devices[0])
x_gpu2 = torch.rand((4000, 4000), device=devices[1])


# 预热设备，确保缓存不会影响最终结果
run(x_gpu1)
run(x_gpu2)
# 分别等待各个设备上的所有流中所有核心的计算完成
torch.cuda.synchronize(device=devices[0])
torch.cuda.synchronize(device=devices[1])
# 分别计时
with d2l.Benchmark('GPU1 time'):
    run(x_gpu1)
    torch.cuda.synchronize(devices[0])

with d2l.Benchmark('GPU2 time'):
    run(x_gpu2)
    torch.cuda.synchronize(devices[1])
# 自动并行计算并及时
with d2l.Benchmark('GPU1 & GPU2'):
    run(x_gpu1)
    run(x_gpu2)
    torch.cuda.synchronize()

# 数据在设备间迁移，等待所有参数可用时再传
with d2l.Benchmark('在GPU1上运行'):
    y = run(x_gpu1)
    torch.cuda.synchronize(devices[0])
with d2l.Benchmark('复制到CPU'):
    y_cpu = copy_to_cpu(y)
    torch.cuda.synchronize()
# 设置non_blocking=True，在不需要同步时可使得调用方绕过同步来传递数据
with d2l.Benchmark('在GPU1上运行并复制到CPU'):
    y = run(x_gpu1)
    y_cpu = copy_to_cpu(y, non_blocking=True)
    torch.cuda.synchronize()

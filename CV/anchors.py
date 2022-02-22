import torch
from d2l import torch as d2l


# torch.set_printoptions(2)  # 精简浮点数输出精度


def multibox_prior(data, sizes, ratios):
    """生成以每个像素为中心具有不同形状的锚框"""
    in_height, in_width = data.shape[-2:]
    device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)
    size_tensor = torch.tensor(sizes, device=device)
    ratio_tensor = torch.tensor(ratios, device=device)  # 对应宽高比
    num_pixels = in_height * in_width
    boxes_per_pixel = num_sizes + num_ratios - 1

    # 为了将锚点移动到像素的中心，需要设置偏移量。
    # 因为一个像素的的高为1且宽为1，我们选择偏移我们的中心0.5
    offset_h, offset_w = 0.5, 0.5
    height_steps = 1 / in_height
    width_steps = 1 / in_width

    # 生成锚框的所有中心点
    center_h = (torch.arange(in_height, device=device) + offset_h) * height_steps
    center_w = (torch.arange(in_width, device=device) + offset_w) * width_steps
    shift_y, shift_x = torch.meshgrid(center_h, center_w)
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)

    # 生成所有的宽和高
    s = torch.cat([size_tensor, size_tensor[0].repeat(num_ratios - 1)])
    r = torch.cat([ratio_tensor[0].repeat(num_ratios - 1), ratio_tensor])
    # 锚框原长采用图片高度
    w = s * torch.sqrt(r) * in_height / in_width
    h = s / torch.sqrt(r)

    # 生成所有锚框的左上和右下点坐标的相对偏移量
    anchor_manipulations = torch.stack([-w, -h, w, h]).T.repeat(num_pixels, 1) / 2

    # 生成锚框坐标
    out_grid = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1). \
        repeat_interleave(boxes_per_pixel, dim=0)
    output = out_grid + anchor_manipulations

    # 还原batch的维度
    return output.unsqueeze(0)


def show_bboxes(axes, bboxes, labels=None, colors=None):
    def _make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj

    labels = _make_list(labels)
    colors = _make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = d2l.bbox_to_rect(bbox.detach().numpy(), color)
        axes.add_patch(rect)
        if labels and i < len(labels):
            text_color = 'k' if colors == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i], color=text_color, fontsize=9,
                      ha='center', va='center', bbox=dict(facecolor=color, lw=0))


def box_iou(boxes1, boxes2):
    """计算两个锚框或边界框列表中成对的交并比"""
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) *
                              (boxes[:, 3] - boxes[:, 1]))
    # boxes1,boxes2,areas1,areas2的形状:
    # boxes1：(boxes1的数量,4),
    # boxes2：(boxes2的数量,4),
    # areas1：(boxes1的数量,),
    # areas2：(boxes2的数量,)
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)
    # inter_upperlefts,inter_lowerrights,inters的形状:
    # (boxes1的数量,boxes2的数量,2)
    # 借助broadcast机制，boxes1中的每一个box都与boxes2中的box逐一匹配计算
    inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)
    # inter_areasandunion_areas的形状:(boxes1的数量,boxes2的数量)
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas


def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    """将最接近的真实边界框分配给锚框"""
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    # 位于第i行和第j列的元素x_ij是锚框i和真实边界框j的IoU
    jaccard = box_iou(anchors, ground_truth)
    # 对于每个锚框，分配的真实边界框的张量
    anchor_bbox_map = torch.full((num_anchors,), -1, dtype=torch.long, device=device)
    # 根据阈值，确定每个anchor是否分配真实边界框
    max_ious, indices = torch.max(jaccard, dim=1)
    anc_i = torch.nonzero(max_ious > iou_threshold).reshape(-1)
    print(anc_i)
    box_j = indices[max_ious > iou_threshold]
    anchor_bbox_map[anc_i] = box_j
    # 预先设定丢弃后行和列的形式
    col_discard = torch.full((num_anchors,), -1, device=device)
    row_discard = torch.full((num_gt_boxes,), -1, device=device)
    # 为每个ground-truth找到对应IoU最大的anchor box
    # 由于只需要每个最大值的坐标，因此使用torch.argmax
    for _ in range(num_gt_boxes):
        max_idx = torch.argmax(jaccard)
        box_idx = (max_idx % num_gt_boxes).long()
        anc_idx = (max_idx / num_gt_boxes).long()
        anchor_bbox_map[anc_idx] = box_idx
        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = row_discard

    return anchor_bbox_map


def offset_boxes(anchors, assigned_bb, eps=1e-6):
    """对锚框偏移量的转换"""
    c_anc = d2l.box_corner_to_center(anchors)
    c_assigned_bb = d2l.box_corner_to_center(assigned_bb)
    offset_xy = (c_assigned_bb[:, :2] - c_anc[:, :2]) / (c_anc[:, 2:] * 0.1)
    offset_wh = torch.log(c_assigned_bb[:, 2:] / c_anc[:, 2:] + eps) / 0.2
    offset = torch.cat((offset_xy, offset_wh), dim=1)
    return offset


def multibox_target(anchors, labels):
    """使用真实边界框标记锚框"""
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    num_anchors, device = anchors.shape[0], anchors.device
    batch_offset, batch_mask, batch_class_labels = [], [], []
    for i in range(batch_size):
        label = labels[i]
        anchor_bbox_map = assign_anchor_to_bbox(label[:, 1:], anchors, device)
        bbox_mask = (anchor_bbox_map >= 0).float().unsqueeze(-1).repeat(1, 4)
        # 默认的类别标签为背景，标记为0
        class_labels = torch.zeros(num_anchors, dtype=torch.long, device=device)
        assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32, device=device)
        indices_true = torch.nonzero(anchor_bbox_map >= 0)
        bbx_idx = anchor_bbox_map[indices_true]
        class_labels[indices_true] = label[bbx_idx, 0].long() + 1
        assigned_bb[indices_true] = label[bbx_idx, 1:]
        # 偏移量转换
        offset = offset_boxes(anchors, assigned_bb)
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)

    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    class_labels = torch.stack(batch_class_labels)
    return bbox_offset, bbox_mask, class_labels


def offset_inverse(anchors, offset_preds):
    """将相对偏移转化为实际预测框的坐标"""
    c_anc = d2l.box_corner_to_center(anchors)
    c_pred_bbox = torch.zeros(c_anc.shape, device=anchors.device)
    c_pred_bbox[:, :2] = offset_preds[:, :2] * 0.1 * c_anc[:, 2:] + c_anc[:, :2]
    c_pred_bbox[:, 2:] = torch.exp(offset_preds[:, 2:] * 0.2) * c_anc[:, 2:]
    pred_bbox = d2l.box_center_to_corner(c_pred_bbox)
    return pred_bbox


def nms(boxes, scores, iou_threshold):
    """对预测边界框的置信度进行排序"""
    # 边界框按照降序进行排列
    B = torch.argsort(scores, dim=-1, descending=True)
    keep = []  # 保留预测边界框的指标
    # 将边界框按照置信度降序排列
    while B.numel() > 0:
        max_idx = B[0]
        keep.append(max_idx)
        if B.numel() == 1:
            break
        # 用list作为索引，只提取包含在list中的序号所对应位置的值
        iou = box_iou(boxes[max_idx, :].reshape(-1, 4), boxes[B[1:], :].reshape(-1, 4)).reshape(-1)
        inds = torch.nonzero(iou <= iou_threshold).reshape(-1)
        B = B[inds + 1]

    return torch.tensor(keep, device=boxes.device)


# @save
def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5,
                       pos_threshold=0.009999999):
    """使用非极大值抑制来预测边界框"""
    device, batch_size = cls_probs.device, cls_probs.shape[0]
    anchors = anchors.squeeze(0)
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    out = []
    for i in range(batch_size):
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)
        # 这里只考虑了前景的分类，通过后续nms后再将排除掉的锚框调整为背景
        conf, class_id = torch.max(cls_prob[1:], 0)
        predicted_bb = offset_inverse(anchors, offset_pred)
        # 用NMS获取需要保留的anchor box序号
        keep = nms(predicted_bb, conf, nms_threshold)

        # 找到所有的non_keep索引，并将类设置为背景
        all_idx = torch.arange(num_anchors, dtype=torch.long, device=device)
        combined = torch.cat((keep, all_idx))
        # 级联后的序列中，只出现过一次的序号即对应被排除的背景锚框
        uniques, counts = combined.unique(return_counts=True)
        non_keep = uniques[counts == 1]
        print(non_keep)
        all_id_sorted = torch.cat((keep, non_keep))
        class_id[non_keep] = -1
        # 将返回锚框的顺序进行调整，前景在前，背景在后
        class_id = class_id[all_id_sorted]
        conf, predicted_bb = conf[all_id_sorted], predicted_bb[all_id_sorted]
        # pos_threshold是一个用于非背景预测的阈值
        # 置信度太低的锚框同样会被划为背景，置信度需要相应反转
        below_min_idx = (conf < pos_threshold)
        class_id[below_min_idx] = -1
        conf[below_min_idx] = 1 - conf[below_min_idx]
        pred_info = torch.cat((class_id.unsqueeze(1),
                               conf.unsqueeze(1),
                               predicted_bb), dim=1)
        out.append(pred_info)
    return torch.stack(out)


# 预载图片
img = d2l.plt.imread('../img/catdog.jpg')
# 获取图像尺寸
h, w = img.shape[:2]

# 测试锚框生成
print(h, w)
X = torch.randn((2, 3, h, w))
Y = multibox_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
print(Y.shape)

# 调整数据维数，方便特定位置的锚框提取
boxes = Y.reshape(h, w, -1, 4)
print(boxes[250, 250, 0, :])

# 获得(250, 250)处的锚框并显示
d2l.set_figsize()
fig = d2l.plt.imshow(img)
bbox_scale = torch.tensor([w, h, w, h])
s = boxes[250, 250, :, :]
show_bboxes(fig.axes, boxes[250, 250, :, :] * bbox_scale,
            ['s=0.75, r=1', 's=0.5, r=1', 's=0.25, r=1', 's=0.75, r=2',
             's=0.75, r=0.5'])
d2l.plt.show()

# 测试真实边界框的分配
ground_truth = torch.tensor([[0, 0.1, 0.08, 0.52, 0.92],
                             [1, 0.55, 0.2, 0.9, 0.88]])
anchors = torch.tensor([[0, 0.1, 0.2, 0.3], [0.15, 0.2, 0.4, 0.4],
                        [0.63, 0.05, 0.88, 0.98], [0.66, 0.45, 0.8, 0.8],
                        [0.57, 0.3, 0.92, 0.9]])
fig = d2l.plt.imshow(img)
show_bboxes(fig.axes, ground_truth[:, 1:] * bbox_scale, ['dog', 'cat'], 'k')
show_bboxes(fig.axes, anchors * bbox_scale, ['0', '1', '2', '3', '4'])
d2l.plt.show()
labels = multibox_target(anchors.unsqueeze(dim=0),
                         ground_truth.unsqueeze(dim=0))
print(labels)

# 测试非极大值抑制
anchors = torch.tensor([[0.1, 0.08, 0.52, 0.92], [0.08, 0.2, 0.56, 0.95],
                        [0.15, 0.3, 0.62, 0.91], [0.55, 0.2, 0.9, 0.88]])
offset_preds = torch.tensor([0] * anchors.numel())
cls_probs = torch.tensor([[0] * 4,  # 背景的预测概率
                          [0.9, 0.8, 0.7, 0.1],  # 狗的预测概率
                          [0.1, 0.2, 0.3, 0.9]])  # 猫的预测概率
fig = d2l.plt.imshow(img)
show_bboxes(fig.axes, anchors * bbox_scale,
            ['dog=0.9', 'dog=0.8', 'dog=0.7', 'cat=0.9'])
output = multibox_detection(cls_probs.unsqueeze(dim=0),
                            offset_preds.unsqueeze(dim=0),
                            anchors.unsqueeze(dim=0),
                            nms_threshold=0.5)
print(output)
d2l.plt.show()

# 显示非极大值抑制后的结果
fig = d2l.plt.imshow(img)
for i in output[0].detach().numpy():
    # i中从低位到高位依次是类别，置信度，锚框坐标
    if i[0] != -1:
        label = ('dog=', 'cat=')[int(i[0])] + str(i[1])
        show_bboxes(fig.axes, [torch.tensor(i[2:]) * bbox_scale], label)
d2l.plt.show()

"""作业题解答"""
# 1. 改变sizes，生成锚框的大小会相应地变大或缩小；改变ratios，生成锚框的宽高比也会相应发生改变

# 2. 略

# 3. 对13.4.3节中的锚框的相对坐标全部加上0.1，使得锚框整体向图片右下角移动，此时第三个锚框由于与真实边界框相交部分面积减小，
# 交并比降低到阈值以下，因此不再被分配相应的真实边界框；对13.4.4节中的锚框的相对坐标全部减去0.1，使得候选框整体向图片左上角移动，
# 但由于相对位置不变，因此非极大值抑制后保留的边界框与原来相同，区别仅在于位置偏移

# 4. 是有可能的，比如密集目标的情况，两个目标可能距离很近，导致真实的边界框交并比很高，这种情况下即便将两个目标对应的边界框都准确预测出来了，
# 但仍有可能在NMS阶段由于交并比太高导致其中一个正确的边界框被错误地排除。为了更柔和地抑制，可以将将非极大值抑制变成随机过程，以交并比作为概率，
# 按照这个概率随机地舍弃边界框，这样交并比大的更可能被舍弃，但同时也给予了保留的余地。SoftNMS中不再舍弃边界框，而是根据IoU对置信度进行指数
# 衰减，这样抑制的过程就体现在了最终返回的置信度当中，最后再对每个置信度进行检验，低于阈值的就会剔除。

# 5. 传统NMS只含有iou_threshold一个参数且并不可导，所以难以学习。要想让NMS变成可学习的模块，需要推广其算法流程，
# 相关工作参见文章"Learning non-maximum suppression"

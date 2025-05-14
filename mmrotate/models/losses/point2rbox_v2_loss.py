# Copyright (c) OpenMMLab. All rights reserved.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from mmdet.models.losses.utils import weighted_loss

from mmrotate.registry import MODELS
from mmrotate.models.losses.gaussian_dist_loss import postprocess


@weighted_loss
def gwd_sigma_loss(pred, target, fun='log1p', tau=1.0, alpha=1.0, normalize=True):
    """Gaussian Wasserstein distance loss.
    Modified from gwd_loss. 
    gwd_sigma_loss only involves sigma in Gaussian, with mu ignored.

    Args:
        pred (torch.Tensor): Predicted bboxes.
        target (torch.Tensor): Corresponding gt bboxes.
        fun (str): The function applied to distance. Defaults to 'log1p'.
        tau (float): Defaults to 1.0.
        alpha (float): Defaults to 1.0.
        normalize (bool): Whether to normalize the distance. Defaults to True.

    Returns:
        loss (torch.Tensor)

    """
    Sigma_p = pred
    Sigma_t = target

    whr_distance = Sigma_p.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    whr_distance = whr_distance + Sigma_t.diagonal(
        dim1=-2, dim2=-1).sum(dim=-1)

    _t_tr = (Sigma_p.bmm(Sigma_t)).diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    _t_det_sqrt = (Sigma_p.det() * Sigma_t.det()).clamp(1e-7).sqrt()
    whr_distance = whr_distance + (-2) * (
        (_t_tr + 2 * _t_det_sqrt).clamp(1e-7).sqrt())

    distance = (alpha * alpha * whr_distance).clamp(1e-7).sqrt()

    if normalize:
        scale = 2 * (
            _t_det_sqrt.clamp(1e-7).sqrt().clamp(1e-7).sqrt()).clamp(1e-7)
        distance = distance / scale

    return postprocess(distance, fun=fun, tau=tau)


def bhattacharyya_coefficient(pred, target):
    """Calculate bhattacharyya coefficient between 2-D Gaussian distributions.

    Args:
        pred (Tuple): tuple of (xy, sigma).
            xy (torch.Tensor): center point of 2-D Gaussian distribution
                with shape (N, 2).
            sigma (torch.Tensor): covariance matrix of 2-D Gaussian distribution
                with shape (N, 2, 2).
        target (Tuple): tuple of (xy, sigma).

    Returns:
        coef (Tensor): bhattacharyya coefficient with shape (N,).
    """
    xy_p, Sigma_p = pred
    xy_t, Sigma_t = target

    _shape = xy_p.shape

    xy_p = xy_p.reshape(-1, 2)
    xy_t = xy_t.reshape(-1, 2)
    Sigma_p = Sigma_p.reshape(-1, 2, 2)
    Sigma_t = Sigma_t.reshape(-1, 2, 2)

    Sigma_M = (Sigma_p + Sigma_t) / 2
    dxy = (xy_p - xy_t).unsqueeze(-1)
    t0 = torch.exp(-0.125 * dxy.permute(0, 2, 1).bmm(torch.linalg.solve(Sigma_M, dxy)))
    t1 = (Sigma_p.det() * Sigma_t.det()).clamp(1e-7).sqrt()
    t2 = Sigma_M.det()

    coef = t0 * (t1 / t2).clamp(1e-7).sqrt()[..., None, None]
    coef = coef.reshape(_shape[:-1])
    return coef


@weighted_loss
def gaussian_overlap_loss(pred, target, alpha=0.01, beta=0.6065):
    """Calculate Gaussian overlap loss based on bhattacharyya coefficient.

    Args:
        pred (Tuple): tuple of (xy, sigma).
            xy (torch.Tensor): center point of 2-D Gaussian distribution
                with shape (N, 2).
            sigma (torch.Tensor): covariance matrix of 2-D Gaussian distribution
                with shape (N, 2, 2).

    Returns:
        loss (Tensor): overlap loss with shape (N, N).
    """
    mu, sigma = pred
    B = mu.shape[0]
    mu0 = mu[None].expand(B, B, 2)
    sigma0 = sigma[None].expand(B, B, 2, 2)
    mu1 = mu[:, None].expand(B, B, 2)
    sigma1 = sigma[:, None].expand(B, B, 2, 2)
    loss = bhattacharyya_coefficient((mu0, sigma0), (mu1, sigma1))
    loss[torch.eye(B, dtype=bool)] = 0
    loss = F.leaky_relu(loss - beta, negative_slope=alpha) + beta * alpha
    loss = loss.sum(-1)
    return loss


@MODELS.register_module()
class GaussianOverlapLoss(nn.Module):
    """Gaussian Overlap Loss.

    Args:
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'. Options are "none", "mean" and
            "sum".
        loss_weight (float, optional): Weight of loss. Defaults to 1.0.

    Returns:
        loss (torch.Tensor)
    """

    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0,
                 lamb=1e-4):
        super(GaussianOverlapLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.lamb = lamb

    def forward(self,
                pred,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (Tuple): tuple of (xy, sigma).
                xy (torch.Tensor): center point of 2-D Gaussian distribution
                    with shape (N, 2).
                sigma (torch.Tensor): covariance matrix of 2-D Gaussian distribution
                    with shape (N, 2, 2).
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        assert len(pred[0]) == len(pred[1])

        sigma = pred[1]
        L = torch.linalg.eigh(sigma)[0].clamp(1e-7).sqrt()
        loss_lamb = F.l1_loss(L, torch.zeros_like(L), reduction='none')
        loss_lamb = self.lamb * loss_lamb.log1p().mean()
        
        return self.loss_weight * (loss_lamb + gaussian_overlap_loss(
            pred,
            None,
            weight,
            reduction=reduction,
            avg_factor=avg_factor))


def plot_gaussian_voronoi_watershed(*images):
    """Plot figures for debug."""
    import matplotlib.pyplot as plt
    plt.figure(dpi=300, figsize=(len(images) * 4, 4))
    plt.tight_layout()
    fileid = np.random.randint(0, 20)
    for i in range(len(images)):
        img = images[i]
        img = (img - img.min()) / (img.max() - img.min())
        if img.dim() == 3:
            img = img.permute(1, 2, 0)
        img = img.detach().cpu().numpy()
        plt.subplot(1, len(images), i + 1)
        if i == 3:
            plt.imshow(img)
            x = np.linspace(0, 1024, 1024)
            y = np.linspace(0, 1024, 1024)
            X, Y = np.meshgrid(x, y)
            plt.contourf(X, Y, img, levels=8, cmap=plt.get_cmap('magma'))
        else:
            plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
    plt.savefig(f'debug/Gaussian-Voronoi-{fileid}.png')
    plt.close()


def gaussian_2d(xy, mu, sigma, normalize=False):
    dxy = (xy - mu).unsqueeze(-1)
    t0 = torch.exp(-0.5 * dxy.permute(0, 2, 1).bmm(torch.linalg.solve(sigma, dxy)))
    if normalize:
        t0 = t0 / (2 * np.pi * sigma.det().clamp(1e-7).sqrt())
    return t0

def gaussian_voronoi_watershed_loss(mu, sigma,
                                    label, image, 
                                    pos_thres, neg_thres,
                                    size=None,uncertainty=None,min_ratio_threshold=None,max_ratio_threshold=None,
                                    down_sample=2, topk=0.95, 
                                    default_sigma=4096,
                                    voronoi='prior_guide',
                                    alpha=0.1,
                                    debug=False):

    
    J = len(sigma)
    if J == 0:
        return sigma.sum()
    
    D = down_sample
    H, W = image.shape[-2:]
    h, w = H // D, W // D
    x = torch.linspace(0, h, h, device=mu.device)
    y = torch.linspace(0, w, w, device=mu.device)
    xy = torch.stack(torch.meshgrid(x, y, indexing='xy'), -1)
    vor = mu.new_zeros(J, h, w)
    # Get distribution for each instance
    mm = (mu.detach() / D).round()
    if voronoi == 'standard':
        sg = sigma.new_tensor((default_sigma, 0, 0, default_sigma)).reshape(2, 2)
        sg = sg / D ** 2
        for j, m in enumerate(mm):
            vor[j] = gaussian_2d(xy.view(-1, 2), m[None], sg[None]).view(h, w)
    elif voronoi == 'gaussian-orientation':
        L, V = torch.linalg.eigh(sigma)
        L = L.detach().clone()
        L = L / (L[:, 0:1] * L[:, 1:2]).sqrt() * default_sigma
        sg = V.matmul(torch.diag_embed(L)).matmul(V.permute(0, 2, 1)).detach()
        sg = sg / D ** 2
        for j, (m, s) in enumerate(zip(mm, sg)):
            vor[j] = gaussian_2d(xy.view(-1, 2), m[None], s[None]).view(h, w)
    elif voronoi == 'gaussian-full':
        sg = sigma.detach() / D ** 2
        for j, (m, s) in enumerate(zip(mm, sg)):
            vor[j] = gaussian_2d(xy.view(-1, 2), m[None], s[None]).view(h, w)
    elif voronoi == 'prior_guide':
        # 加权维诺图，按照size参数进行加权
        L, V = torch.linalg.eigh(sigma)
        L = L.detach().clone()
        L = L / (L[:, 0:1] * L[:, 1:2]).sqrt() * default_sigma
        sg = V.matmul(torch.diag_embed(L)).matmul(V.permute(0, 2, 1)).detach()
        sg = sg / D ** 2
        
        # 获取每个目标的类别，以便使用对应的size进行加权
        classes = label.detach().cpu().numpy()
        weights = torch.tensor([size[int(cls)] for cls in classes], device=mu.device)
        
        for j, (m, s) in enumerate(zip(mm, sg)):
            # 根据类别对应的size进行权重加成
            weight = weights[j]
            # 用权重调整高斯分布半径
            vor[j] = gaussian_2d(xy.view(-1, 2), m[None], s[None] / weight).view(h, w) * weight
            
    # val: max prob, vor: belong to which instance, cls: belong to which class
    val, vor = torch.max(vor, 0)
    if D > 1:
        vor = vor[:, None, :, None].expand(-1, D, -1, D).reshape(H, W)
        val = F.interpolate(
            val[None, None], (H, W), mode='bilinear', align_corners=True)[0, 0]
    cls = label[vor]
    kernel = val.new_ones((1, 1, 3, 3))
    kernel[0, 0, 1, 1] = -8
    ridges = torch.conv2d(vor[None].float(), kernel, padding=1)[0] != 0
    vor += 1
    pos_thres = val.new_tensor(pos_thres)
    neg_thres = val.new_tensor(neg_thres)
    vor[val < pos_thres[cls]] = 0
    vor[val < neg_thres[cls]] = J + 1
    vor[ridges] = J + 1

    cls_bg = torch.where(vor == J + 1, 15, cls)
    cls_bg = torch.where(vor == 0, -1, cls_bg)

    # PyTorch does not support watershed, use cv2
    img_uint8 = (image - image.min()) / (image.max() - image.min()) * 255
    img_uint8 = img_uint8.permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
    img_uint8 = cv2.medianBlur(img_uint8, 3)
    markers = vor.detach().cpu().numpy().astype(np.int32)
    markers = vor.new_tensor(cv2.watershed(img_uint8, markers))
    
    # 应用先验约束进行分水岭结果后处理
    if voronoi == 'prior_guide':
        ori_markers= markers.detach()
        markers = apply_prior_constraints(markers, label, size, uncertainty, 
                                          min_ratio_threshold, max_ratio_threshold, 
                                          image, J)
        

    if debug:
        # plot_gaussian_voronoi_watershed(image, cls_bg, markers)
        plot_watershed_result(image, ori_markers,markers, label)

    L, V = torch.linalg.eigh(sigma)
    L_target = []
    for j in range(J):
        xy = (markers == j + 1).nonzero()[:, (1, 0)].float()
        if len(xy) == 0:
            L_target.append(L[j].detach())
            continue
        xy = xy - mu[j]
        xy = V[j].T.matmul(xy[:, :, None])[:, :, 0]
        max_x = torch.max(torch.abs(xy[:, 0]))
        max_y = torch.max(torch.abs(xy[:, 1]))
        L_target.append(torch.stack((max_x, max_y)) ** 2)
    L_target = torch.stack(L_target)
    L = torch.diag_embed(L)
    L_target = torch.diag_embed(L_target)
    loss = gwd_sigma_loss(L, L_target.detach(), reduction='none')
    loss = torch.topk(loss, int(np.ceil(len(loss) * topk)), largest=False)[0].mean()
    return loss, (vor, markers)


def apply_prior_constraints(markers, label, size, uncertainty, 
                           min_ratio_threshold, max_ratio_threshold, 
                           image, J):
    """
    根据先验约束调整分水岭的结果
    
    Args:
        markers: 分水岭的结果
        label: 每个目标的类别
        size: 每个类别的面积大小指数
        uncertainty: 每个类别的不可信度
        min_ratio_threshold: 与平均值的最小比例阈值
        max_ratio_threshold: 与平均值的最大比例阈值
        image: 原始图像
        J: 目标的数量
    """
    # 从GPU转到CPU处理
    markers_np = markers.detach().cpu().numpy()
    label_np = label.detach().cpu().numpy()
    
    # 按类别分组目标
    class_groups = {}
    for j in range(J):
        cls = int(label_np[j])
        if cls not in class_groups:
            class_groups[cls] = []
        area = np.sum(markers_np == (j + 1))
        if area > 0:  # 确保分水岭结果中目标存在
            class_groups[cls].append((j, area))
    
    # 创建一个标记掩码的副本，用于修改
    modified_markers = markers_np.copy()
    
    # 情况1: 处理同一图片内多种目标类别的情况
    if len(class_groups) > 1:
        # 先处理同一类别有多个目标的情况
        for cls, objects in class_groups.items():
            if len(objects) > 1:
                # 计算该类别目标的平均面积
                areas = [area for _, area in objects]
                mean_area = np.mean(areas)
                
                # 根据阈值确定允许的面积范围
                min_area = mean_area * min_ratio_threshold[cls]
                max_area = mean_area * max_ratio_threshold[cls]
                
                # 调整不符合范围的目标
                for j, area in objects:
                    if area < min_area:
                        # 区域生长直到达到阈值
                        modified_markers = region_grow(modified_markers, j+1, min_area, image)
                    elif area > max_area:
                        # 形态学腐蚀
                        modified_markers = region_erode(modified_markers, j+1, max_area)
        
        # 处理只有一个实例的类别，它们需要与其他类别进行比较
        single_instance_classes = {cls: objs[0] for cls, objs in class_groups.items() if len(objs) == 1}
        
        if len(single_instance_classes) > 0:
            # 如果有多个单实例类别，两两配对处理
            classes = list(single_instance_classes.keys())
            for i in range(len(classes)):
                for j in range(i+1, len(classes)):
                    cls1, cls2 = classes[i], classes[j]
                    j1, area1 = single_instance_classes[cls1]
                    j2, area2 = single_instance_classes[cls2]
                    
                    # 根据不可信度计算贡献权重
                    s1 = uncertainty[cls1]
                    s2 = uncertainty[cls2]
                    
                    # 理想面积比例
                    c1 = size[cls1]
                    c2 = size[cls2]
                    
                    # 检查当前面积比例是否在合理范围内
                    current_ratio = area1 / area2
                    target_ratio = c1 / c2
                    
                    # 如果比例不合理，则调整
                    if abs(current_ratio - target_ratio) > 0.2 * target_ratio:
                        # 使用公式计算调整量
                        t = (c1 * area2 - c2 * area1) / (c2 * s1 + c1 * s2)
                        delta_area1 = s1 * t
                        delta_area2 = s2 * t
                        
                        # 根据计算结果调整区域
                        if delta_area1 > 0:
                            # 区域1需要扩大
                            modified_markers = region_grow(modified_markers, j1+1, area1 + delta_area1, image)
                        elif delta_area1 < 0:
                            # 区域1需要缩小
                            modified_markers = region_erode(modified_markers, j1+1, area1 + delta_area1)
                            
                        if delta_area2 > 0:
                            # 区域2需要扩大
                            modified_markers = region_grow(modified_markers, j2+1, area2 + delta_area2, image)
                        elif delta_area2 < 0:
                            # 区域2需要缩小
                            modified_markers = region_erode(modified_markers, j2+1, area2 + delta_area2)
    
    # 情况2: 处理同一图片只有一种目标类别的情况
    elif len(class_groups) == 1:
        cls = list(class_groups.keys())[0]
        objects = class_groups[cls]
        
        # 只处理不确定度大于0.5的类别
        if uncertainty[cls] >= 0.5:
            if len(objects) > 1:
                # 计算平均面积
                areas = [area for _, area in objects]
                mean_area = np.mean(areas)
                
                # 根据阈值确定允许的面积范围
                min_area = mean_area * min_ratio_threshold[cls]
                max_area = mean_area * max_ratio_threshold[cls]
                
                # 调整不符合范围的目标
                for j, area in objects:
                    if area < min_area:
                        # 区域生长直到达到阈值
                        modified_markers = region_grow(modified_markers, j+1, min_area, image)
                    elif area > max_area:
                        # 形态学腐蚀
                        modified_markers = region_erode(modified_markers, j+1, max_area)
    
    # 将处理后的结果转回PyTorch tensor
    return torch.tensor(modified_markers, device=markers.device, dtype=markers.dtype)


def region_grow(markers, label_id, target_area, image):
    """
    基于原图的区域生长算法，扩展目标区域直到达到目标面积
    """
    # 将图像转换为灰度图（如果它是彩色的）
    if len(image.shape) > 2 and image.shape[2] > 1:
        img_gray = cv2.cvtColor(image.permute(1, 2, 0).cpu().numpy().astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        img_gray = image.cpu().numpy().astype(np.uint8)
    
    # 获取当前目标区域
    current_mask = markers == label_id
    current_area = np.sum(current_mask)
    
    if current_area >= target_area:
        return markers
    
    # 初始化队列（包含当前区域的边界像素）
    seed_points = []
    
    # 找出边界像素（使用膨胀和减法来找出边界）
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(current_mask.astype(np.uint8), kernel, iterations=1)
    boundary = dilated - current_mask.astype(np.uint8)
    boundary_points = np.where(boundary > 0)
    
    # 将边界点添加到队列中，并计算其与当前区域的平均像素值差异
    region_mean = np.mean(img_gray[current_mask])
    for y, x in zip(boundary_points[0], boundary_points[1]):
        # 检查点是否在图像边界内且不属于其他目标
        if (0 <= y < markers.shape[0] and 0 <= x < markers.shape[1] and 
            markers[y, x] <= 0):  # 只添加背景点
            pixel_value = img_gray[y, x]
            diff = abs(pixel_value - region_mean)
            # 添加点及其与均值的差异作为优先级
            seed_points.append((diff, (y, x)))
    
    # 按照与区域均值差异排序（差异小的先添加）
    seed_points.sort()
    
    # 当前区域掩码的副本
    new_mask = current_mask.copy()
    
    # 开始区域生长，直到达到目标面积或没有更多可添加的点
    while current_area < target_area and seed_points:
        # 取出差异最小的点
        _, (y, x) = seed_points.pop(0)
        
        # 如果该点已被添加或已被其他标签占用，则跳过
        if new_mask[y, x] or markers[y, x] > 0:
            continue
        
        # 添加该点到区域中
        new_mask[y, x] = True
        current_area += 1
        
        # 若达到目标面积，则停止
        if current_area >= target_area:
            break
        
        # 检查该点的邻居，将符合条件的邻居添加到种子点列表
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            
            # 检查邻居是否在图像边界内且未被添加过且不属于其他目标
            if (0 <= ny < markers.shape[0] and 0 <= nx < markers.shape[1] and 
                not new_mask[ny, nx] and markers[ny, nx] <= 0):
                pixel_value = img_gray[ny, nx]
                diff = abs(pixel_value - region_mean)
                
                # 插入新的种子点，保持列表排序
                i = 0
                while i < len(seed_points) and seed_points[i][0] < diff:
                    i += 1
                seed_points.insert(i, (diff, (ny, nx)))
    
    # 更新标记图
    markers_copy = markers.copy()
    markers_copy[new_mask] = label_id
    
    return markers_copy

def region_erode(markers, label_id, target_area):
    """
    形态学腐蚀算法，减小目标区域直到达到目标面积
    """
    current_mask = markers == label_id
    current_area = np.sum(current_mask)
    
    if current_area <= target_area:
        return markers
    
    # 转换为OpenCV格式
    mask = current_mask.astype(np.uint8) * 255
    
    # 创建不同尺度的结构元素用于腐蚀
    kernel_small = np.ones((3, 3), np.uint8)
    kernel_medium = np.ones((5, 5), np.uint8)
    
    # 逐步腐蚀直到达到目标面积
    iterations = 0
    max_iterations = 50  # 防止无限循环
    
    while current_area > target_area and iterations < max_iterations:
        if iterations < 10:
            # 前10次使用小内核
            mask_eroded = cv2.erode(mask, kernel_small, iterations=1)
        else:
            # 之后使用中等内核
            mask_eroded = cv2.erode(mask, kernel_medium, iterations=1)
            
        # 更新当前掩码和面积
        new_mask = mask_eroded > 0
        new_area = np.sum(new_mask)
        
        if new_area < current_area:
            # 更新标记
            markers_copy = markers.copy()
            markers_copy[current_mask & ~new_mask] = 0  # 移除腐蚀掉的部分
            markers = markers_copy
            current_mask = new_mask
            current_area = new_area
        else:
            # 无法进一步减小
            break
            
        iterations += 1
        
        # 如果已经接近目标面积，可以提前结束
        if current_area <= 1.05 * target_area:
            break
    
    return markers

@MODELS.register_module()
class VoronoiWatershedLoss(nn.Module):
    """Gaussian Overlap Loss.

    Args:
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'. Options are "none", "mean" and
            "sum".
        loss_weight (float, optional): Weight of loss. Defaults to 1.0.

    Returns:
        loss (torch.Tensor)
    """

    def __init__(self,
                 down_sample=2,
                 reduction='mean',
                 loss_weight=1.0,
                 topk=0.95,
                 alpha=0.1,
                 debug=False):
        super(VoronoiWatershedLoss, self).__init__()
        self.down_sample = down_sample
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.topk = topk
        self.alpha = alpha
        self.debug = debug

    def forward(self, pred, label, image, pos_thres, neg_thres,
                voronoi='prior_guide',
                size=None,uncertainty=None,min_ratio_threshold=None,max_ratio_threshold=None):
        """Forward function.

        Args:
            pred (Tuple): Tuple of (xy, sigma).
                xy (torch.Tensor): Center point of 2-D Gaussian distribution
                    with shape (N, 2).
                sigma (torch.Tensor): Covariance matrix of 2-D Gaussian distribution
                    with shape (N, 2, 2).
            image (torch.Tensor): The image for watershed with shape (3, H, W).
            standard_voronoi (bool, optional): Use standard or Gaussian voronoi.

        Returns:
            torch.Tensor: The calculated loss
        """
        # Use ** to prevent duplicate parameters
        loss, self.vis = gaussian_voronoi_watershed_loss(
            *pred,
            label,
            image,
            pos_thres,
            neg_thres,
            size=size,
            uncertainty=uncertainty,
            min_ratio_threshold=min_ratio_threshold,
            max_ratio_threshold=max_ratio_threshold,
            down_sample=self.down_sample,
            topk=self.topk,
            voronoi=voronoi,
            alpha=self.alpha,
            debug=self.debug
        )
        return self.loss_weight * loss


def rbbox2roi(bbox_list):
    """Convert a list of bboxes to roi format.

    Args:
        bbox_list (list[Tensor]): a list of bboxes corresponding to a batch
            of images.

    Returns:
        Tensor: shape (n, 6), [batch_ind, cx, cy, w, h, a]
    """
    rois_list = []
    for img_id, bboxes in enumerate(bbox_list):
        if bboxes.size(0) > 0:
            img_inds = bboxes.new_full((bboxes.size(0), 1), img_id)
            rois = torch.cat([img_inds, bboxes[:, :5]], dim=-1)
        else:
            rois = bboxes.new_zeros((0, 6))
        rois_list.append(rois)
    rois = torch.cat(rois_list, 0)
    return rois


def plot_edge_map(feat, edgex, edgey):
    """Plot figures for debug."""
    import matplotlib.pyplot as plt
    plt.figure(dpi=300, figsize=(4, 4))
    plt.tight_layout()
    fileid = np.random.randint(0, 20)
    for i in range(len(feat)):
        img0 = feat[i, :3]
        img0 = (img0 - img0.min()) / (img0.max() - img0.min())
        img1 = edgex[i, :3]
        img1 = (img1 - img1.min()) / (img1.max() - img1.min())
        img2 = edgey[i, :3]
        img2 = (img2 - img2.min()) / (img2.max() - img2.min())
        img3 = img1 + img2
        img3 = (img3 - img3.min()) / (img3.max() - img3.min())
        img = torch.cat((torch.cat((img0, img2), -1), 
                         torch.cat((img1, img3), -1)), -2
                         ).permute(1, 2, 0).detach().cpu().numpy()
        N = int(np.ceil(np.sqrt(len(feat))))
        plt.subplot(N, N, i + 1)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
    plt.savefig(f'debug/Edge-Map-{fileid}.png')
    plt.close()
def plot_watershed_result(image, original_markers, optimized_markers, labels, edgex=None, edgey=None):
    """
    绘制原始分水岭结果与优化后结果的对比，并标注类别
    
    Args:
        image (torch.Tensor): 原始图像，形状为 [C, H, W]
        original_markers (torch.Tensor): 原始分水岭结果，形状为 [H, W]
        optimized_markers (torch.Tensor): 优化后的分水岭结果，形状为 [H, W]
        labels (torch.Tensor): 每个目标的类别，形状为 [N]
        edgex (torch.Tensor, optional): X方向边缘图
        edgey (torch.Tensor, optional): Y方向边缘图
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    from matplotlib.colors import ListedColormap
    
    # 类别名称列表
    class_names = [
        'plane', 'baseball-diamond', 'bridge', 'ground-track-field',
        'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
        'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout',
        'harbor', 'swimming-pool', 'helicopter'
    ]
    
    plt.figure(dpi=300, figsize=(15, 10))
    
    fileid = labels.detach().cpu().numpy().sum()
    
    # 准备颜色映射 - 使用不同颜色表示不同标签
    np.random.seed(42)  # 固定随机种子以保持颜色一致
    colors = np.random.rand(256, 3)  # 生成随机颜色
    colors[0] = [0, 0, 0]  # 背景为黑色
    cmap = ListedColormap(colors)
    
    # 将图像转换为numpy数组
    if isinstance(image, torch.Tensor):
        img = image.permute(1, 2, 0).cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min())  # 归一化到[0,1]
    else:
        img = image
    
    if isinstance(original_markers, torch.Tensor):
        orig_markers_np = original_markers.cpu().numpy()
    else:
        orig_markers_np = original_markers
        
    if isinstance(optimized_markers, torch.Tensor):
        opt_markers_np = optimized_markers.cpu().numpy()
    else:
        opt_markers_np = optimized_markers
    
    if isinstance(labels, torch.Tensor):
        labels_np = labels.cpu().numpy()
    else:
        labels_np = labels
    
    # 创建3x2的子图布局
    # 原始图像
    plt.subplot(2, 3, 1)
    plt.title("Original Image")
    plt.imshow(img)
    plt.axis('off')
    
    # 原始分水岭结果
    plt.subplot(2, 3, 2)
    plt.title("Original Watershed")
    plt.imshow(orig_markers_np, cmap=cmap)
    plt.axis('off')
    
    # 带标签的原始分水岭结果
    plt.subplot(2, 3, 3)
    plt.title("Original Watershed (Labeled)")
    plt.imshow(orig_markers_np, cmap=cmap, alpha=0.7)
    plt.imshow(img, alpha=0.3)
    
    # 为原始分水岭结果添加标签
    used_labels = set()
    for j in range(1, orig_markers_np.max() + 1):
        if j - 1 < len(labels_np):
            y_indices, x_indices = np.where(orig_markers_np == j)
            if len(y_indices) > 0:
                y_center = int(np.mean(y_indices))
                x_center = int(np.mean(x_indices))
                
                class_id = int(labels_np[j - 1])
                if class_id < len(class_names):
                    class_name = class_names[class_id]
                else:
                    class_name = f"Class {class_id}"
                
                plt.text(x_center, y_center, class_name, 
                         color='white', fontsize=8, 
                         ha='center', va='center',
                         bbox=dict(boxstyle="round,pad=0.3", fc='black', alpha=0.7))
                used_labels.add(class_id)
    plt.axis('off')
    
    # 优化后的分水岭结果
    plt.subplot(2, 3, 5)
    plt.title("Optimized Watershed")
    plt.imshow(opt_markers_np, cmap=cmap)
    plt.axis('off')
    
    # 带标签的优化分水岭结果
    plt.subplot(2, 3, 6)
    plt.title("Optimized Watershed (Labeled)")
    plt.imshow(opt_markers_np, cmap=cmap, alpha=0.7)
    plt.imshow(img, alpha=0.3)
    
    # 为优化分水岭结果添加标签
    for j in range(1, opt_markers_np.max() + 1):
        if j - 1 < len(labels_np):
            y_indices, x_indices = np.where(opt_markers_np == j)
            if len(y_indices) > 0:
                y_center = int(np.mean(y_indices))
                x_center = int(np.mean(x_indices))
                
                class_id = int(labels_np[j - 1])
                if class_id < len(class_names):
                    class_name = class_names[class_id]
                else:
                    class_name = f"Class {class_id}"
                
                plt.text(x_center, y_center, class_name, 
                         color='white', fontsize=8, 
                         ha='center', va='center',
                         bbox=dict(boxstyle="round,pad=0.3", fc='black', alpha=0.7))
                used_labels.add(class_id)
    plt.axis('off')
    
    # 如果有边缘图，则显示
    if edgex is not None and edgey is not None:
        plt.subplot(2, 3, 4)
        plt.title("Edge Map")
        
        if isinstance(edgex, torch.Tensor) and isinstance(edgey, torch.Tensor):
            img1 = edgex[0, :3]
            img1 = (img1 - img1.min()) / (img1.max() - img1.min())
            img2 = edgey[0, :3]
            img2 = (img2 - img2.min()) / (img2.max() - img2.min())
            img3 = img1 + img2
            img3 = (img3 - img3.min()) / (img3.max() - img3.min())
            
            edge_img = img3.permute(1, 2, 0).detach().cpu().numpy()
            plt.imshow(edge_img)
        else:
            # 如果没有提供边缘图，则显示面积变化对比图
            area_diff = np.zeros_like(opt_markers_np, dtype=float)
            
            # 计算每个目标的面积变化比例
            for j in range(1, max(orig_markers_np.max(), opt_markers_np.max()) + 1):
                orig_area = np.sum(orig_markers_np == j)
                opt_area = np.sum(opt_markers_np == j)
                
                if orig_area > 0 and opt_area > 0:
                    # 标记优化后区域的面积变化比例
                    change_ratio = (opt_area - orig_area) / orig_area
                    # 面积增加显示为红色，减少显示为蓝色
                    area_diff[opt_markers_np == j] = change_ratio
            
            plt.title("Area Change (Red: +, Blue: -)")
            plt.imshow(area_diff, cmap='coolwarm', vmin=-1, vmax=1)
            plt.colorbar(label='Area Change Ratio')
        
        plt.axis('off')
    
    # 创建图例
    if used_labels:
        handles = []
        for class_id in sorted(used_labels):
            if class_id < len(class_names):
                patch = mpatches.Patch(color=colors[class_id+1], label=class_names[class_id])
                handles.append(patch)
        
        if handles:
            plt.figlegend(handles=handles, loc='lower center', ncol=min(5, len(handles)))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12 if used_labels else 0.05)  # 为图例留出空间
    
    # 保存图像
    plt.savefig(f'debug/Watershed-Comparison-{fileid}.png')
    plt.close()

    print(f"Comparison visualization saved to debug/Watershed-Comparison-{fileid}.png")
    
    return fileid


@MODELS.register_module()
class EdgeLoss(nn.Module):
    """Edge Loss.

    Args:
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'. Options are "none", "mean" and
            "sum".
        loss_weight (float, optional): Weight of loss. Defaults to 1.0.

    Returns:
        loss (torch.Tensor)
    """

    def __init__(self,
                 resolution=24,
                 max_scale=1.6,
                 sigma=6,
                 reduction='mean',
                 loss_weight=1.0,
                 debug=False):
        super(EdgeLoss, self).__init__()
        self.resolution = resolution
        self.max_scale = max_scale
        self.sigma = sigma
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.center_idx = self.resolution / self.max_scale
        self.debug = debug

        self.roi_extractor = MODELS.build(dict(
            type='RotatedSingleRoIExtractor',
                roi_layer=dict(
                    type='RoIAlignRotated',
                    out_size=(2 * self.resolution + 1),
                    sample_num=2,
                    clockwise=True),
            out_channels=1,
            featmap_strides=[1],
            finest_scale=1024))

        edge_idx = torch.arange(0, self.resolution + 1)
        edge_distribution = torch.exp(-((edge_idx - self.center_idx) ** 2) / (2 * self.sigma ** 2))
        edge_distribution[0] = edge_distribution[-1] = 0
        self.register_buffer('edge_idx', edge_idx)
        self.register_buffer('edge_distribution', edge_distribution)

    def forward(self, pred, edge):
        """Forward function.

        Args:
            pred (Tuple): Batched predicted rboxes
            edge (torch.Tensor): The edge map with shape (B, 1, H, W).

        Returns:
            torch.Tensor: The calculated loss
        """
        G = self.resolution
        C = self.center_idx
        roi = rbbox2roi(pred)
        roi[:, 3:5] *= self.max_scale
        feat = self.roi_extractor([edge], roi)
        if len(feat) == 0:
            return pred[0].new_tensor(0)
        featx = feat.sum(1).abs().sum(1)
        featy = feat.sum(1).abs().sum(2)
        featx2 = torch.flip(featx[:, :G + 1], (-1,)) + featx[:, G:]
        featy2 = torch.flip(featy[:, :G + 1], (-1,)) + featy[:, G:]  # (N, 25)
        ex = ((featx2 * self.edge_distribution).softmax(1) * self.edge_idx).sum(1) / C
        ey = ((featy2 * self.edge_distribution).softmax(1) * self.edge_idx).sum(1) / C
        exy = torch.stack((ex, ey), -1)
        rbbox_concat = torch.cat(pred, 0)
        
        if self.debug:
            edgex = featx[:, None, None, :].expand(-1, 1, 2 * self.resolution + 1, -1)
            edgey = featy[:, None, :, None].expand(-1, 1, -1, 2 * self.resolution + 1)
            plot_edge_map(feat, edgex, edgey)

        return self.loss_weight * F.smooth_l1_loss(rbbox_concat[:, 2:4], 
                                      (rbbox_concat[:, 2:4] * exy).detach(),
                                      beta=8)


@MODELS.register_module()
class Point2RBoxV2ConsistencyLoss(nn.Module):
    """Consistency Loss.

    Args:
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'. Options are "none", "mean" and
            "sum".
        loss_weight (float, optional): Weight of loss. Defaults to 1.0.

    Returns:
        loss (torch.Tensor)
    """

    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0):
        super(Point2RBoxV2ConsistencyLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, ori_pred, trs_pred, square_mask, aug_type, aug_val):
        """Forward function.

        Args:
            ori_pred (Tuple): (Sigma, theta)
            trs_pred (Tuple): (Sigma, theta)
            square_mask: When True, the angle is ignored
            aug_type: 'rot', 'flp', 'sca'
            aug_val: Rotation or scale value

        Returns:
            torch.Tensor: The calculated loss
        """
        ori_gaus, ori_angle = ori_pred
        trs_gaus, trs_angle = trs_pred

        if aug_type == 'rot':
            rot = ori_gaus.new_tensor(aug_val)
            cos_r = torch.cos(rot)
            sin_r = torch.sin(rot)
            R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
            ori_gaus = R.matmul(ori_gaus).matmul(R.permute(0, 2, 1))
            d_ang = trs_angle - ori_angle - aug_val
        elif aug_type == 'flp':
            ori_gaus = ori_gaus * ori_gaus.new_tensor((1, -1, -1, 1)).reshape(2, 2)
            d_ang = trs_angle + ori_angle
        else:
            sca = ori_gaus.new_tensor(aug_val)
            ori_gaus = ori_gaus * sca
            d_ang = trs_angle - ori_angle
        
        loss_ssg = gwd_sigma_loss(ori_gaus.bmm(ori_gaus), trs_gaus.bmm(trs_gaus))
        d_ang = (d_ang + math.pi / 2) % math.pi - math.pi / 2
        loss_ssa = F.smooth_l1_loss(d_ang, torch.zeros_like(d_ang), reduction='none', beta=0.1)
        loss_ssa = loss_ssa[~square_mask].sum() / max(1, (~square_mask).sum())

        return self.loss_weight * (loss_ssg + loss_ssa)

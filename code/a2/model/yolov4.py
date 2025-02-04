import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from model.ciou import bbox_ciou, xywh2xyxy


class YOLOv4Loss(nn.Module):
    def __init__(self, anchors, num_classes=80):
        super(YOLOv4Loss, self).__init__()
        self.anchors = anchors
        self.num_classes = num_classes
        self.ignore_thres = 0.5
        self.obj_scale = 1
        self.noobj_scale = 100
        
    def forward(self, predictions, targets):
        """
        Args:
            predictions: tensor of shape (batch_size, num_boxes, 5 + num_classes)
            targets: list of dicts containing boxes and labels
        """
        device = predictions.device
        batch_size = predictions.size(0)
        total_loss = torch.tensor(0., requires_grad=True, device=device)
        
        for i in range(batch_size):
            pred = predictions[i]
            target = targets[i]
            
            # Get target boxes and labels and move them to the correct device
            target_boxes = target['boxes'].to(device)
            target_labels = target['labels'].to(device)
            
            if len(target_boxes) == 0:
                continue
            
            # Get prediction components
            pred_boxes = pred[..., :4]  # [x, y, w, h]
            pred_conf = pred[..., 4]    # objectness
            pred_cls = pred[..., 5:]    # class scores
            
            # Calculate IoU for each predicted box with each target box
            num_pred = pred_boxes.size(0)
            num_target = target_boxes.size(0)
            
            # Expand dimensions for broadcasting
            pred_boxes = pred_boxes.unsqueeze(1).repeat(1, num_target, 1)
            target_boxes = target_boxes.unsqueeze(0).repeat(num_pred, 1, 1)
            
            # Calculate CIoU loss
            ciou = bbox_ciou(pred_boxes.view(-1, 4), target_boxes.view(-1, 4))
            ciou = ciou.view(num_pred, num_target)
            
            # For each target, find the best matching prediction
            best_ious, best_idx = ciou.max(dim=0)
            
            # Calculate box loss using CIoU
            box_loss = (1.0 - best_ious).mean()
            
            # Calculate objectness loss
            obj_mask = torch.zeros_like(pred_conf)
            obj_mask[best_idx] = 1
            obj_loss = F.binary_cross_entropy_with_logits(pred_conf, obj_mask)
            
            # Calculate classification loss
            target_cls = torch.zeros_like(pred_cls)
            for j, label in enumerate(target_labels):
                target_cls[best_idx[j], label] = 1
            cls_loss = F.binary_cross_entropy_with_logits(pred_cls, target_cls)
            
            # Combine losses
            batch_loss = box_loss + obj_loss + cls_loss
            total_loss = total_loss + batch_loss
        
        return total_loss / batch_size


class YOLOLayer(nn.Module):
    def __init__(self, anchors, nc, img_size, yolo_index, layers, stride):
        super(YOLOLayer, self).__init__()
        self.anchors = torch.Tensor(anchors)
        self.index = yolo_index  # index of this layer in layers
        self.layers = layers  # model output layer indices
        self.stride = stride  # layer stride
        self.nl = len(layers)  # number of output layers (3)
        self.na = len(anchors)  # number of anchors (3)
        self.nc = nc  # number of classes (80)
        self.no = nc + 5  # number of outputs (85)
        self.nx, self.ny, self.ng = 0, 0, 0  # initialize number of x, y gridpoints
        self.anchor_vec = self.anchors / self.stride
        self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2)

        if ONNX_EXPORT:
            self.training = False
            self.create_grids((img_size[1] // stride, img_size[0] // stride))  # number x, y grid points

    def create_grids(self, ng=(13, 13), device='cpu'):
        self.nx, self.ny = ng  # x and y grid size
        self.ng = torch.tensor(ng, dtype=torch.float)

        # build xy offsets
        if not self.training:
            yv, xv = torch.meshgrid([torch.arange(self.ny, device=device), torch.arange(self.nx, device=device)])
            self.grid = torch.stack((xv, yv), 2).view((1, 1, self.ny, self.nx, 2)).float()

        if self.anchor_vec.device != device:
            self.anchor_vec = self.anchor_vec.to(device)
            self.anchor_wh = self.anchor_wh.to(device)

    def forward(self, p, out):
        ASFF = False  # https://arxiv.org/abs/1911.09516
        if ASFF:
            i, n = self.index, self.nl  # index in layers, number of layers
            p = out[self.layers[i]]
            bs, _, ny, nx = p.shape  # bs, 255, 13, 13
            if (self.nx, self.ny) != (nx, ny):
                self.create_grids((nx, ny), p.device)

            # outputs and weights
            # w = F.softmax(p[:, -n:], 1)  # normalized weights
            w = torch.sigmoid(p[:, -n:]) * (2 / n)  # sigmoid weights (faster)
            # w = w / w.sum(1).unsqueeze(1)  # normalize across layer dimension

            # weighted ASFF sum
            p = out[self.layers[i]][:, :-n] * w[:, i:i + 1]
            for j in range(n):
                if j != i:
                    p += w[:, j:j + 1] * \
                         F.interpolate(out[self.layers[j]][:, :-n], size=[ny, nx], mode='bilinear', align_corners=False)

        elif ONNX_EXPORT:
            bs = 1  # batch size
        else:
            bs, _, ny, nx = p.shape  # bs, 255, 13, 13
            if (self.nx, self.ny) != (nx, ny):
                self.create_grids((nx, ny), p.device)

        # p.view(bs, 255, 13, 13) -- > (bs, 3, 13, 13, 85)  # (bs, anchors, grid, grid, classes + xywh)
        p = p.view(bs, self.na, self.no, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()  # prediction

        if self.training:
            return p

        elif ONNX_EXPORT:
            # Avoid broadcasting for ANE operations
            m = self.na * self.nx * self.ny
            ng = 1. / self.ng.repeat(m, 1)
            grid = self.grid.repeat(1, self.na, 1, 1, 1).view(m, 2)
            anchor_wh = self.anchor_wh.repeat(1, 1, self.nx, self.ny, 1).view(m, 2) * ng

            p = p.view(m, self.no)
            xy = torch.sigmoid(p[:, 0:2]) + grid  # x, y
            wh = torch.exp(p[:, 2:4]) * anchor_wh  # width, height
            p_cls = torch.sigmoid(p[:, 4:5]) if self.nc == 1 else \
                torch.sigmoid(p[:, 5:self.no]) * torch.sigmoid(p[:, 4:5])  # conf
            return p_cls, xy * ng, wh

        else:  # inference
            io = p.sigmoid()
            io[..., :2] = (io[..., :2] * 2. - 0.5 + self.grid)
            io[..., 2:4] = (io[..., 2:4] * 2) ** 2 * self.anchor_wh
            io[..., :4] *= self.stride
            #io = p.clone()  # inference output
            #io[..., :2] = torch.sigmoid(io[..., :2]) + self.grid  # xy
            #io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh  # wh yolo method
            #io[..., :4] *= self.stride
            #torch.sigmoid_(io[..., 4:])
            return io.view(bs, -1, self.no), p  # view [1, 3, 13, 13, 85] as [1, 507, 85]
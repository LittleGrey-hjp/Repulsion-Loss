import math
import torch
import torch.nn as nn
import numpy as np
from utils.metrics import bbox_iou
import time


class RepLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, sigma=1):
        super(RepLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma
        self.eps = 1e-7

    def forward(self, gt_boxes, pre_boxes):
        # bbox_iou利用广播机制，得到所有目标和所有预测框的iou，维度和预测框维度一致
        box_iou = self.bbox_iou(gt_boxes, pre_boxes)
        proposal_overlaps = self.bbox_iou(pre_boxes, pre_boxes, xywh=False)
        # box_iou.max得到最大IOU的index和值，按行索引，取出当前行IOU最大的索引，因此得到的维度是行数
        max_attr, max_attr_index = box_iou.max(dim=0)
        # 得到IOU最大的目标框
        GT_attr = gt_boxes[max_attr_index]
        # 将iou最大的值赋值为0，方便后面计算次IOU最大，range 按行遍历
        box_iou[max_attr_index, range(pre_boxes.shape[0])] = 0
        # 判断若只有一个预测目标和真实目标匹配，不存在次IOU最大框则第二项不开启
        if not box_iou.sum == 0:
            max_rep, max_rep_index = box_iou.max(dim=0)
            GT_rep = gt_boxes[max_rep_index]
            rep_loss = self.Attr(pre_boxes, GT_attr, max_attr) + \
                       self.alpha * self.RepGT(pre_boxes, GT_rep, max_attr)+\
                       self.beta * self.RepBox(proposal_overlaps)

        else:
            rep_loss = self.Attr(GT_attr, pre_boxes, max_attr)+self.beta * self.RepBox(proposal_overlaps)

        return rep_loss

    def Attr(self, gt_boxes, pre_boxes, max_iou):
        Attr_loss = 0
        for index, (gt_box, pre_box) in enumerate(zip(gt_boxes, pre_boxes)):
            # if max_iou[index] > self.sigma:
            Attr_loss += self.SmoothL1(gt_box, pre_box)
        Attr_loss = Attr_loss.sum() / len(gt_boxes)
        return Attr_loss

    def RepGT(self, gt_boxes, pre_boxes, max_iou):
        # RepGT_loss = 0
        # count = 0
        # t1=time.time()
        # for index, (gt_box, pre_box) in enumerate(zip(gt_boxes, pre_boxes)):
        #     count += 1
        #     IOG = self.RepGT_iog(gt_box, pre_box)
        #     if IOG > self.sigma:
        #         RepGT_loss += ((IOG - self.sigma) / (1 - self.sigma) - math.log(1 - self.sigma)).sum()
        #     else:
        #         RepGT_loss += -(1 - IOG).clamp(min=self.eps).log().sum()
        # RepGT_loss = RepGT_loss.sum() / count
        # t2=time.time()
        # print(t2-t1) # Time :0.3487117290496826

        RepGT_iog = self.RepGT_iog(gt_boxes, pre_boxes, List=False)
        Re_loss_tensor = torch.where(RepGT_iog > self.sigma,
                                     (RepGT_iog - self.sigma) / (1 - self.sigma) - math.log(1 - self.sigma),
                                     - torch.clamp(torch.log(1 - RepGT_iog), min=self.eps))
        RepGT_loss = Re_loss_tensor.mean()  # Time.00598731232
        return RepGT_loss

        # Replace with cIOU
        # t3= time.time()
        # reploss=bbox_iou(gt_boxes, pre_boxes, CIoU=True)
        # reploss.mean()
        # t4=time.time()
        # print(t4-t3)  #Time for one iteration:0.0012631416320800781

    def RepBox(self, overlaps):
        # RepBox_loss = 0
        # overlap_loss = 0
        # count = 0
        # result = overlaps.triu(1)
        # for i in range(0, overlaps.shape[0]):
        #     for j in range(1 + i, overlaps.shape[0]):
        #         count += 1
        #         if overlaps[i][j] > self.sigma:
        #             RepBox_loss += ((overlaps[i][j] - self.sigma) / (1 - self.sigma) - math.log(1 - self.sigma)).sum()
        #         else:
        #             RepBox_loss += -(1 - overlaps[i][j]).clamp(min=self.eps).log().sum()
        # RepBox_loss = RepBox_loss / count # time:10.72557178497

        Re_loss_tensor = torch.where(overlaps > self.sigma,
                                   (overlaps - self.sigma) / (1 - self.sigma) - math.log(1 - self.sigma),
                                   torch.clamp(-torch.log(1-overlaps), min=self.eps))
        up_triangular = torch.triu(Re_loss_tensor, diagonal=1)
        non_zereos_elements = up_triangular[up_triangular != 0]
        RepBox_loss = non_zereos_elements.mean()  # time:0.00598731232
        return RepBox_loss

    def SmoothL1(self, pred, target, beta=1.0):
        diff = torch.abs(pred - target)
        cond = torch.lt(diff, beta)
        loss = torch.where(cond, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
        return loss

    def RepGT_iog(self, box1, box2, List=True):
        if List:  # transform from xywh to xyxy
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
        else:  # x1, y1, x2, y2 = box1
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

        # Intersection area
        inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
        # g_area Area
        g_area = torch.abs(b2_x2 - b2_x1) * torch.abs(b2_y2 - b2_y1)
        # IoU
        iog = inter / g_area
        return iog

    def bbox_iou(self, bboxes1, bboxes2, xywh=True, eps=1e-7):
        if xywh:
            # chunk(4, -1)表示在最后一个维度上切块
            (x1, y1, w1, h1), (x2, y2, w2, h2) = bboxes1.chunk(4, -1), bboxes2.chunk(4, -1)
            bboxes1[:, 0:1], bboxes1[:, 1:2], bboxes1[:, 2:3], bboxes1[:,
                                                               3:4] = x1 - w1 / 2, y1 - h1 / 2, x1 + w1 / 2, y1 + h1 / 2
            bboxes2[:, 0:1], bboxes2[:, 1:2], bboxes2[:, 2:3], bboxes2[:,
                                                               3:4] = x2 - w2 / 2, y2 - h2 / 2, x2 + w2 / 2, y2 + h2 / 2

        lt = torch.max(bboxes1[:, None, :2], bboxes2[:, :2])  # [rows, cols, 2]
        rb = torch.min(bboxes1[:, None, 2:], bboxes2[:, 2:])  # [rows, cols, 2]

        wh = (rb - lt + 1).clamp(min=0)  # [rows, cols, 2]
        overlap = wh[:, :, 0] * wh[:, :, 1]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
                bboxes1[:, 3] - bboxes1[:, 1] + 1)

        area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
                bboxes2[:, 3] - bboxes2[:, 1] + 1)
        ious = overlap / (area1[:, None] + area2 - overlap).clamp(min=eps)  # 会产生大于1的值？？？

        return ious.clamp(min=eps, max=1)


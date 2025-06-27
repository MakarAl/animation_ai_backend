# modified from https://github.com/hzwer/ECCV2022-RIFE/blob/main/model/IFNet.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import itertools
from model.warplayer import warp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
        )

def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes, kernel_size=4, stride=2, padding=1, bias=True),
        nn.PReLU(out_planes)
        )
            
class Conv2(nn.Module):
    def __init__(self, in_planes, out_planes, stride=2):
        super(Conv2, self).__init__()
        self.conv1 = conv(in_planes, out_planes, 3, stride, 1)
        self.conv2 = conv(out_planes, out_planes, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
    
c = 16
class Contextnet(nn.Module):
    def __init__(self):
        super(Contextnet, self).__init__()
        self.conv1 = Conv2(3, c)
        self.conv2 = Conv2(c, 2*c)
        self.conv3 = Conv2(2*c, 4*c)
        self.conv4 = Conv2(4*c, 8*c)
    
    def forward(self, x, flow):
        x = self.conv1(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 0.5
        f1 = warp(x, flow)        
        x = self.conv2(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 0.5
        f2 = warp(x, flow)
        x = self.conv3(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 0.5
        f3 = warp(x, flow)
        x = self.conv4(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 0.5
        f4 = warp(x, flow)
        return [f1, f2, f3, f4]
    
class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.down0 = Conv2(17, 2*c)
        self.down1 = Conv2(4*c, 4*c)
        self.down2 = Conv2(8*c, 8*c)
        self.down3 = Conv2(16*c, 16*c)
        self.up0 = deconv(32*c, 8*c)
        self.up1 = deconv(16*c, 4*c)
        self.up2 = deconv(8*c, 2*c)
        self.up3 = deconv(4*c, c)
        self.conv = nn.Conv2d(c, 3, 3, 1, 1)

    def forward(self, img0, img1, warped_img0, warped_img1, mask, flow, c0, c1):
        # Down path
        d0_inputs = crop_to_min_size(img0, img1, warped_img0, warped_img1, mask, flow)
        s0 = self.down0(torch.cat(d0_inputs, 1))
        d1_inputs = crop_to_min_size(s0, c0[0], c1[0])
        s1 = self.down1(torch.cat(d1_inputs, 1))
        d2_inputs = crop_to_min_size(s1, c0[1], c1[1])
        s2 = self.down2(torch.cat(d2_inputs, 1))
        d3_inputs = crop_to_min_size(s2, c0[2], c1[2])
        s3 = self.down3(torch.cat(d3_inputs, 1))
        # Up path
        u0_inputs = crop_to_min_size(s3, c0[3], c1[3])
        x = self.up0(torch.cat(u0_inputs, 1))
        u1_inputs = crop_to_min_size(x, s2)
        x = self.up1(torch.cat(u1_inputs, 1))
        u2_inputs = crop_to_min_size(x, s1)
        x = self.up2(torch.cat(u2_inputs, 1))
        u3_inputs = crop_to_min_size(x, s0)
        x = self.up3(torch.cat(u3_inputs, 1))
        x = self.conv(x)
        return torch.sigmoid(x)


class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64):
        super(IFBlock, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c//2, 3, 2, 1),
            conv(c//2, c, 3, 2, 1),
            )
        self.convblock = nn.Sequential(
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
        )
        self.lastconv = nn.ConvTranspose2d(c, 5, 4, 2, 1)

    def forward(self, x, flow, scale):
        if scale != 1:
            x = F.interpolate(x, scale_factor = 1. / scale, mode="bilinear", align_corners=False)
        if flow != None:
            flow = F.interpolate(flow, scale_factor = 1. / scale, mode="bilinear", align_corners=False) * 1. / scale
            # Crop x and flow to min size before concatenation
            x, flow = crop_to_min_size(x, flow)
            x = torch.cat((x, flow), 1)
        x = self.conv0(x)
        x = self.convblock(x) + x
        tmp = self.lastconv(x)
        tmp = F.interpolate(tmp, scale_factor = scale * 2, mode="bilinear", align_corners=False)
        flow = tmp[:, :4] * scale * 2
        mask = tmp[:, 4:5]
        return flow, mask
    
def crop_to_min_size(*tensors):
    """Crop all tensors to the minimum common height and width."""
    min_h = min(t.shape[2] for t in tensors)
    min_w = min(t.shape[3] for t in tensors)
    return [t[:, :, :min_h, :min_w] for t in tensors]

class IFNet(nn.Module):
    def __init__(self):
        super(IFNet, self).__init__()
        self.block0 = IFBlock(6, c=240)
        self.block1 = IFBlock(13+4, c=150)
        self.block2 = IFBlock(13+4, c=90)
        self.block_tea = IFBlock(16+4, c=90)
        self.contextnet = Contextnet()
        self.unet = Unet()

    def forward(self, I0t, I1t, scale=[4,2,1], timestep=0.5):
        x = torch.cat((I0t.repeat(1, 3, 1, 1), I1t.repeat(1, 3, 1, 1)), dim=1)
        img0 = x[:, :3]
        img1 = x[:, 3:6]
        gt = x[:, 6:] # In inference time, gt is None
        flow_list = []
        merged = []
        mask_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = None 
        loss_distill = 0
        stu = [self.block0, self.block1, self.block2]
        for i in range(3):
            if flow != None:
                # Crop all tensors to min size before concatenation
                img0_c, img1_c, warped_img0_c, warped_img1_c, mask_c = crop_to_min_size(img0, img1, warped_img0, warped_img1, mask)
                flow_d, mask_d = stu[i](torch.cat((img0_c, img1_c, warped_img0_c, warped_img1_c, mask_c), 1), flow, scale=scale[i])
                # Crop flow and flow_d, mask and mask_d to min size before addition
                flow, flow_d = crop_to_min_size(flow, flow_d)
                mask, mask_d = crop_to_min_size(mask, mask_d)
                flow = flow + flow_d
                mask = mask + mask_d
            else:
                img0_c, img1_c = crop_to_min_size(img0, img1)
                flow, mask = stu[i](torch.cat((img0_c, img1_c), 1), None, scale=scale[i])
            mask_list.append(torch.sigmoid(mask))
            flow_list.append(flow)
            # Crop for warping
            img0_w, flow_w0 = crop_to_min_size(img0, flow[:, :2])
            img1_w, flow_w1 = crop_to_min_size(img1, flow[:, 2:4])
            warped_img0 = warp(img0_w, flow_w0)
            warped_img1 = warp(img1_w, flow_w1)
            merged_student = (warped_img0, warped_img1)
            merged.append(merged_student)
        if gt.shape[1] == 3:
            img0_c, img1_c, warped_img0_c, warped_img1_c, mask_c, gt_c = crop_to_min_size(img0, img1, warped_img0, warped_img1, mask, gt)
            flow_d, mask_d = self.block_tea(torch.cat((img0_c, img1_c, warped_img0_c, warped_img1_c, mask_c, gt_c), 1), flow, scale=1)
            img0_w, flow_w0 = crop_to_min_size(img0, flow_d[:, :2])
            img1_w, flow_w1 = crop_to_min_size(img1, flow_d[:, 2:4])
            flow_teacher = flow + flow_d
            warped_img0_teacher = warp(img0_w, flow_w0)
            warped_img1_teacher = warp(img1_w, flow_w1)
            mask_teacher = torch.sigmoid(mask + mask_d)
            merged_teacher = warped_img0_teacher * mask_teacher + warped_img1_teacher * (1 - mask_teacher)
        else:
            flow_teacher = None
            merged_teacher = None
        for i in range(3):
            m0, m1, mask_c = crop_to_min_size(merged[i][0], merged[i][1], mask_list[i])
            merged[i] = m0 * mask_c + m1 * (1 - mask_c)
            if gt.shape[1] == 3:
                m_merged, m_teacher, m_gt = crop_to_min_size(merged[i], merged_teacher, gt)
                loss_mask = ((m_merged - m_gt).abs().mean(1, True) > (m_teacher - m_gt).abs().mean(1, True) + 0.01).float().detach()
                loss_distill += (((flow_teacher.detach() - flow_list[i]) ** 2).mean(1, True) ** 0.5 * loss_mask).mean()
        c0 = self.contextnet(img0, flow[:, :2])
        c1 = self.contextnet(img1, flow[:, 2:4])
        # Crop context features to min size
        c0 = [f[:, :, :min(c0[0].shape[2], c1[0].shape[2]), :min(c0[0].shape[3], c1[0].shape[3])] for f in c0]
        c1 = [f[:, :, :min(c0[0].shape[2], c1[0].shape[2]), :min(c0[0].shape[3], c1[0].shape[3])] for f in c1]
        img0_c, img1_c, warped_img0_c, warped_img1_c, mask_c, flow_c = crop_to_min_size(img0, img1, warped_img0, warped_img1, mask, flow)
        tmp = self.unet(img0_c, img1_c, warped_img0_c, warped_img1_c, mask_c, flow_c, c0, c1)
        res = tmp[:, :3] * 2 - 1
        m2, r2 = crop_to_min_size(merged[2], res)
        merged[2] = torch.clamp(m2 + r2, 0, 1)
        return flow_list[2], mask_list[2], merged[2], flow_teacher, merged_teacher, loss_distill
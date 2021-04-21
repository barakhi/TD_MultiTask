# Adapted from: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

import torch 
import torch.nn as nn
import torch.nn.functional as F
import math


class VGGclassifier(nn.Module):
    def __init__(self, num_classes=2, init_weights=True):
        super(VGGclassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(128 * 7 * 7, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        x = torch.flatten(x, 1)
        x = self.classifier(x)
        out = F.log_softmax(x, dim=1)
        return out


class BUTD_comp_CLEVR_vgg(nn.Module):
    def __init__(self, num_tasks, block, num_blocks):
        super(BUTD_comp_CLEVR_vgg, self).__init__()
        self.in_planes = 64
        self.in_planes_TD = 512

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 128, kernel_size=3, padding=1)
        self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.avgpool = nn.AvgPool2d(7, stride=1)

        self.emb = nn.Linear(num_tasks, 32*49) #=128*4*4
        self.conv_emb = nn.Conv2d(32, 128, kernel_size=1, stride=1) #was 32 128
        # enlarge map
        self.TDlayer4 = nn.Sequential(nn.Conv2d(128, 512, kernel_size=3, padding=1), nn.ReLU(True), nn.Conv2d(512, 512, kernel_size=3, padding=1))
        self.TDlayer3 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(True), nn.Conv2d(512, 256, kernel_size=3, padding=1))
        self.TDlayer2 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(True), nn.Conv2d(256, 128, kernel_size=3, padding=1))
        self.TDlayer1 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(True), nn.Conv2d(128, 64, kernel_size=3, padding=1))
        self.TDlayer0 = nn.Conv2d(64, 32, kernel_size=3, padding=1)

        #self.conv_seg = nn.Conv2d(32, 2, kernel_size=1)
        self.conv_seg = nn.Conv2d(32, 1, kernel_size=1)

        #self.lat1_0 = nn.Conv2d(64, 32, kernel_size=1)
        self.lat1_1 = nn.Conv2d(128, 128, kernel_size=1)
        self.lat1_2 = nn.Conv2d(256, 256, kernel_size=1)
        self.lat1_3 = nn.Conv2d(512, 512, kernel_size=1)
        self.lat1_4 = nn.Conv2d(128, 128, kernel_size=1)

        #self.lat2_0 = nn.Conv2d(32, 64, kernel_size=1)
        self.lat2_1 = nn.Conv2d(64, 64, kernel_size=1)
        self.lat2_2 = nn.Conv2d(128, 128, kernel_size=1)
        self.lat2_3 = nn.Conv2d(256, 256, kernel_size=1)
        self.lat2_4 = nn.Conv2d(512, 512, kernel_size=1)

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = VGGclassifier(num_classes=8)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, image, task):
        bs = image.size(0)

        out_stem = nn.ReLU(inplace=True)(self.conv1(image))
        out_stem = self.maxpool1(out_stem)
        out_l1 = self.conv2(out_stem)
        out_l1 = self.maxpool2(out_l1)
        out_l2 = self.conv3_2(nn.ReLU(inplace=True)(self.conv3_1(nn.ReLU(inplace=True)(out_l1))))
        out_l2 = self.maxpool3(out_l2)
        out_l3 = self.conv4_2(nn.ReLU(inplace=True)(self.conv4_1(nn.ReLU(inplace=True)(out_l2))))
        out_l3 = self.maxpool4(out_l3)
        out_l4 = self.conv5_2(nn.ReLU(inplace=True)(self.conv5_1(nn.ReLU(inplace=True)(out_l3))))
        out_l4 = self.maxpool5(out_l4)

        task_emb = self.emb(task)

        td_first = self.conv_emb(task_emb.view(bs, 32, 7, 7))
        td_d4 = td_first + self.lat1_4(out_l4)
        td_c4 = F.interpolate(td_d4, scale_factor=2, mode='nearest')
        td_e4 = self.TDlayer4(td_c4)
        td_e4_l = self.lat2_4(td_e4)

        td_d3 = td_e4 + self.lat1_3(out_l3)  # + self.lat1_2(bu1_e2)
        td_c3 = F.interpolate(td_d3, scale_factor=2, mode='nearest')
        td_e3 = self.TDlayer3(td_c3)  # + self.lat1_2(bu1_e2)
        td_e3_l = self.lat2_3(td_e3)

        td_d2 = td_e3 + self.lat1_2(out_l2)  # ==================> loss
        td_c2 = F.interpolate(td_d2, scale_factor=2, mode='nearest')
        td_e2 = self.TDlayer2(td_c2)  # ==================> loss
        td_e2_l = self.lat2_2(td_e2)

        td_d1 = td_e2 + self.lat1_1(out_l1)  # ==================> loss
        td_c1 = F.interpolate(td_d1, scale_factor=2, mode='nearest')
        td_e1 = self.TDlayer1(td_c1)  # ==================> loss
        td_e1_l = self.lat2_1(td_e1)

        td_e0 = F.interpolate(td_e1, scale_factor=2, mode='nearest')
        td_e0 = self.TDlayer0(td_e0)
        td_e0 = self.conv_seg(td_e0)

        out2_stem = nn.ReLU(inplace=True)(self.conv1(image))
        out2_stem = self.maxpool1(out2_stem)
        out2_stem_i = out2_stem * td_e1_l
        out2_l1 = self.conv2(out2_stem_i)
        out2_l1 = self.maxpool2(out2_l1)
        out2_l1_i = out2_l1 * td_e2_l
        out2_l2 = self.conv3_2(nn.ReLU(inplace=True)(self.conv3_1(nn.ReLU(inplace=True)(out2_l1_i))))
        out2_l2 = self.maxpool3(out2_l2)
        out2_l2_i = out2_l2 * td_e3_l
        out2_l3 = self.conv4_2(nn.ReLU(inplace=True)(self.conv4_1(nn.ReLU(inplace=True)(out2_l2_i))))
        out2_l3 = self.maxpool4(out2_l3)
        out2_l3_i = out2_l3 * td_e4_l
        out2_l4 = self.conv5_2(nn.ReLU(inplace=True)(self.conv5_1(nn.ReLU(inplace=True)(out2_l3_i))))
        out2_l4 = self.maxpool5(out2_l4)


        rep = self.avgpool(out2_l4)

        out = self.classifier(rep)

        return out, td_e0


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        #self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        #self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        #out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        #out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(int(planes))
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(int(planes))
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(int(planes)*4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


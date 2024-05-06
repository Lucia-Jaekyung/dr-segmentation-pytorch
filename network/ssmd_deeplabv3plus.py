import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from network.xception import Xception

class ASPP(nn.Module):
    def __init__(self, in_planes, out_planes, rate):
        super(ASPP, self).__init__()
        if rate == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = rate

        self.atrous_convolution = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=1, padding=padding, dilation=rate, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU()

        self.__init_weight()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class Encoder(nn.Module):
    def __init__(self, in_planes, os):
        super(Encoder, self).__init__()

        self.xception_features = Xception(in_planes, os)

        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        # ASPP
        self.aspp1 = ASPP(2048, 256, rate=rates[0])
        self.aspp2 = ASPP(2048, 256, rate=rates[1])
        self.aspp3 = ASPP(2048, 256, rate=rates[2])
        self.aspp4 = ASPP(2048, 256, rate=rates[3])

        self.relu = nn.ReLU()

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(2048, 256, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(256),
                                             nn.ReLU())

        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)

        self.__init_weight()

    def forward(self, x):
        x, low_level_features = self.xception_features(x)

        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.upsample(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return x, low_level_features

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class Decoder(nn.Module):
    def __init__(self, in_planes, n_classes, recon=False):
        super(Decoder, self).__init__()

        self.conv2 = nn.Conv2d(128, 48, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)
        self.relu = nn.ReLU()

        if recon:
            self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                           nn.BatchNorm2d(256),
                                           nn.ReLU(),
                                           nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                           nn.BatchNorm2d(256),
                                           nn.ReLU(),
                                           nn.Conv2d(256, in_planes, kernel_size=1, stride=1))
        else:
            self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                             nn.BatchNorm2d(256),
                                             nn.ReLU(),
                                             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                             nn.BatchNorm2d(256),
                                             nn.ReLU(),
                                             nn.Conv2d(256, n_classes, kernel_size=1, stride=1))


        self.__init_weight()

    def forward(self, x, low_level_features):
        x = F.upsample(x, size=(int(math.ceil(512/4)),
                                int(math.ceil(512/4))),
                                mode='bilinear', align_corners=True) # x upsample

        low_level_features = self.conv2(low_level_features)
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)

        x = torch.cat((x, low_level_features), dim=1)
        x = self.last_conv(x)
        x = F.upsample(x, size=(512, 512), mode='bilinear', align_corners=True)

        return x

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class SSMDDeepLabv3plus(nn.Module):
    def __init__(self, n_classes, in_planes=3, os=16):
        super(SSMDDeepLabv3plus, self).__init__()

        self.encoder = Encoder(in_planes, os)
        self.reconstruction = Decoder(in_planes, n_classes=None, recon=True)

        self.microaneurysms = Decoder(in_planes, n_classes, recon=False)
        self.hemohedges = Decoder(in_planes, n_classes, recon=False)
        self.hardexudates = Decoder(in_planes, n_classes, recon=False)
        self.softexudates = Decoder(in_planes, n_classes, recon=False)

        self.__init_weight()

    def forward(self, x):
        x, low_level_features = self.encoder(x)
        x0 = self.reconstruction(x, low_level_features)
        x1 = self.microaneurysms(x, low_level_features)
        x2 = self.hemohedges(x, low_level_features)
        x3 = self.hardexudates(x, low_level_features)
        x4 = self.softexudates(x, low_level_features)
        return x0, x1, x2, x3, x4

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
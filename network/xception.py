import torch
import torch.nn as nn
import torch.nn.functional as F


def fixed_padding(inputs, kernel_size, rate):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs

class SeparableConvoluton(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(SeparableConvoluton, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size, stride, 0, dilation, groups=in_planes, bias=bias)
        self.pointwise = nn.Conv2d(in_planes, out_planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = fixed_padding(x, self.conv1.kernel_size[0], rate=self.conv1.dilation[0])
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

class Block(nn.Module):
    # reps(몇 번 반복할지, entry flow에서는 1번(입력2), middle flow에서는 2번(입력3)), grow_firs & is_last(블록마다 확인하여 설정)
    def __init__(self, in_planes, out_planes, reps, stride=1, dilation=1, start_with_relu=True, grow_first=True, is_last=True):
        super(Block, self).__init__()

        # skip connection
        if out_planes != in_planes or stride != 1:
            self.skip = nn.Conv2d(in_planes, out_planes, 1, stride=stride, bias=False)
            self.skipbn = nn.BatchNorm2d(out_planes)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        # entry flow & middel flow
        filters = in_planes # 필터 개수를 맞추기 위함
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConvoluton(in_planes, out_planes, 3, stride=1, dilation=dilation))
            rep.append(nn.BatchNorm2d(out_planes))
            filters = out_planes

        # 반복
        for _ in range(reps-1):
            rep.append(self.relu)
            rep.append(SeparableConvoluton(filters, filters, 3, stride=1, dilation=dilation))
            rep.append(nn.BatchNorm2d(filters))

        # entry flow에서 사용
        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConvoluton(in_planes, out_planes, 3, stride=1, dilation=dilation))
            rep.append(nn.BatchNorm2d(out_planes))

        # relu가 시작이 아닐 경우 (entry flow에서 사용)
        if not start_with_relu:
            rep = rep[1:]

        if stride != 1: # 각 블록에서 마지막에 stride가 2일 경우
            rep.append(SeparableConvoluton(out_planes, out_planes, 3, stride=2))

        if stride == 1 and is_last: # exit flow의 마지막에 사용
            rep.append(SeparableConvoluton(out_planes, out_planes, 3, stride=1))

        self.rep = nn.Sequential(*rep)

    def forward(self, input):
        x = self.rep(input)

        # skip connection
        if self.skip is not None:
            skip = self.skip(input)
            skip = self.skipbn(skip)
        else:
            skip = input

        x += skip

        return x

class Xception(nn.Module):
    def __init__(self, in_planes=3, os=16):
        super(Xception, self).__init__()

        if os == 16:
            entry_block3_stride = 2
            middle_block_rate = 1
            exit_block_rates = (1, 2)
        elif os == 8:
            entry_block3_stride = 1
            middle_block_rate = 2
            exit_block_rates = (2, 4)
        else:
            raise NotImplementedError

        # Entry Flow
        self.conv1 = nn.Conv2d(in_planes, 32, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.block1 = Block(64, 128, reps=2, stride=2, start_with_relu=False)
        self.block2 = Block(128, 256, reps=2, stride=2, start_with_relu=True, grow_first=True)
        self.block3 = Block(256, 728, reps=2, stride=entry_block3_stride, start_with_relu=True, grow_first=True, is_last=True)

        # Middle Flow
        self.block4  = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True)
        self.block5  = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True)
        self.block6  = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True)
        self.block7  = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True)
        self.block8  = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True)
        self.block9  = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True)
        self.block10 = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True)
        self.block11 = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True)
        self.block12 = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True)
        self.block13 = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True)
        self.block14 = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True)
        self.block15 = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True)
        self.block16 = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True)
        self.block17 = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True)
        self.block18 = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True)
        self.block19 = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True)

        # Exit Flow
        self.block20 = Block(728, 1024, reps=2, stride=1, dilation=exit_block_rates[0], start_with_relu=True, grow_first=False, is_last=True)

        self.conv3 = SeparableConvoluton(1024, 1536, 3, stride=1, dilation=exit_block_rates[1])
        self.bn3 = nn.BatchNorm2d(1536)

        self.conv4 = SeparableConvoluton(1536, 1536, 3, stride=1, dilation=exit_block_rates[1])
        self.bn4 = nn.BatchNorm2d(1536)

        self.conv5 = SeparableConvoluton(1536, 2048, 3, stride=1, dilation=exit_block_rates[1])
        self.bn5 = nn.BatchNorm2d(2048)

        # Init weights
        self._init_weight()

    def forward(self, x):
        # Entry Flow
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        # low_level_features: block1의 결과값으로, deeplabv3+ 구조에서 사용
        low_level_features = x
        x = self.block2(x)
        x = self.block3(x)

        # Middle Flow
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)
        x = self.block17(x)
        x = self.block18(x)
        x = self.block19(x)

        # Exit Flow
        x = self.block20(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)

        return x, low_level_features

    def _init_weight(self):
        for m in self.modules():
            # modules에서 각 레이어가 Conv2d이면 weight를 초기화, BatchNorm2d이면 weight를 1로, bias를 0으로 초기화
            if isinstance(m, nn.Conv2d): # isinstance(확인하고자 하는 데이터 값, 확인하고자 하는 데이터 타입), 서로 타입이 같으면 True, 아니면 False
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight) # input tensor가 He초기값으로 N(0,std^2)의 정규분포를 갖는다.
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
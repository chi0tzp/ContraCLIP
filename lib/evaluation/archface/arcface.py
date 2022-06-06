from torch import nn
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout, MaxPool2d, \
    AdaptiveAvgPool2d, Sequential, Module
import torch
from collections import namedtuple


class IDComparator(nn.Module):
    def __init__(self):
        super(IDComparator, self).__init__()
        self.backbone = SE_IR(50, drop_ratio=0.4, mode='ir_se')
        self.backbone.load_state_dict(torch.load('models/pretrained/arcface/model_ir_se50.pth'))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.criterion = nn.CosineSimilarity(dim=1, eps=1e-6)

    def extract_feats(self, x):
        # Crop interesting region
        x = x[:, :, 35:223, 32:220]
        return self.backbone(self.face_pool(x))

    def forward(self, x, x_prime):
        return self.criterion(self.extract_feats(x), self.extract_feats(x_prime)).mean()


########################################################################################################################
##                                                                                                                    ##
##                                             [ Original Arcface Model ]                                             ##
##                                                                                                                    ##
########################################################################################################################
class Flatten(Module):
    @staticmethod
    def forward(x):
        return x.view(x.size(0), -1)


def l2_norm(x, axis=1):
    norm = torch.norm(x, 2, axis, True)
    output = torch.div(x, norm)
    return output


class SEModule(Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(
            channels, channels // reduction, kernel_size=(1, 1), padding=0, bias=False)
        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(
            channels // reduction, channels, kernel_size=(1, 1), padding=0, bias=False)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class bottleneck_IR(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False), BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False), PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False), BatchNorm2d(depth))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


class bottleneck_IR_SE(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR_SE, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth),
            SEModule(depth, 16)
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    """A named tuple describing a ResNet block."""


def get_block(in_channel, depth, num_units, stride=2):
    return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for _ in range(num_units - 1)]


def get_blocks(num_layers):
    if num_layers == 50:
        return [get_block(in_channel=64, depth=64, num_units=3),
                get_block(in_channel=64, depth=128, num_units=4),
                get_block(in_channel=128, depth=256, num_units=14),
                get_block(in_channel=256, depth=512, num_units=3)]

    elif num_layers == 100:
        return [get_block(in_channel=64, depth=64, num_units=3),
                get_block(in_channel=64, depth=128, num_units=13),
                get_block(in_channel=128, depth=256, num_units=30),
                get_block(in_channel=256, depth=512, num_units=3)]
    elif num_layers == 152:
        return [get_block(in_channel=64, depth=64, num_units=3),
                get_block(in_channel=64, depth=128, num_units=8),
                get_block(in_channel=128, depth=256, num_units=36),
                get_block(in_channel=256, depth=512, num_units=3)]


class SE_IR(Module):
    def __init__(self, num_layers, drop_ratio=0.4, mode='ir_se'):
        super(SE_IR, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'

        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), (1, 1), 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        self.output_layer = Sequential(BatchNorm2d(512),
                                       Dropout(drop_ratio),
                                       Flatten(),
                                       Linear(512 * 7 * 7, 512),
                                       BatchNorm1d(512))
        modules = []
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            for block in blocks:
                for bottleneck in block:
                    modules.append(bottleneck_IR(bottleneck.in_channel, bottleneck.depth, bottleneck.stride))
        elif mode == 'ir_se':
            for block in blocks:
                for bottleneck in block:
                    modules.append(bottleneck_IR_SE(bottleneck.in_channel, bottleneck.depth, bottleneck.stride))

        self.body = Sequential(*modules)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)
        return l2_norm(x)

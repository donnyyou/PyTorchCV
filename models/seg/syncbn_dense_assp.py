# SynBN_version of DenseAspp
import torch
from torch import nn

from torchvision.models import DenseNet
import re
from encoding.nn import BatchNorm2d

from models.backbones.backbone_selector import BackboneSelector


class SyncBNDenseASPP(nn.Module):
    """
    * output_scale can only set as 8 or 16
    """
    def __init__(self, configer):
        super(SyncBNDenseASPP, self).__init__()
        self.configer = configer
        dropout0 = 0.1
        dropout1 = 0.1

        self.features = BackboneSelector(configer).get_backbone()

        num_features = self.features.get_num_features()

        self.trans = _Transition(num_input_features=self.num_features, num_output_features=self.num_features // 2)

        self.num_features = self.num_features // 2

        self.ASPP_3 = _DenseAsppBlock(input_num=num_features, num1=256, num2=64,
                                      dilation_rate=3, drop_out=dropout0)

        self.ASPP_6 = _DenseAsppBlock(input_num=num_features + 64 * 1, num1=256, num2=64,
                                      dilation_rate=6, drop_out=dropout0)

        self.ASPP_12 = _DenseAsppBlock(input_num=num_features + 64 * 2, num1=256, num2=64,
                                       dilation_rate=12, drop_out=dropout0)

        self.ASPP_18 = _DenseAsppBlock(input_num=num_features + 64 * 3, num1=256, num2=64,
                                       dilation_rate=18, drop_out=dropout0)

        self.ASPP_24 = _DenseAsppBlock(input_num=num_features + 64 * 4, num1=256, num2=64,
                                       dilation_rate=24, drop_out=dropout0)

        num_features = num_features + 5 * 64

        self.classification = nn.Sequential(
            nn.Dropout2d(p=dropout1),
            nn.Conv2d(in_channels=num_features,
                      out_channels=self.configer.get('network', 'out_channels'), kernel_size=1, padding=0)
        )

        self.upsample = nn.Sequential(nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        feature = self.features(x)

        aspp3 = self.ASPP_3(feature)
        feature = torch.cat((aspp3, feature), dim=1)

        aspp6 = self.ASPP_6(feature)
        feature = torch.cat((aspp6, feature), dim=1)

        aspp12 = self.ASPP_12(feature)
        feature = torch.cat((aspp12, feature), dim=1)

        aspp18 = self.ASPP_18(feature)
        feature = torch.cat((aspp18, feature), dim=1)

        aspp24 = self.ASPP_24(feature)
        feature = torch.cat((aspp24, feature), dim=1)

        cls = self.classification(feature)

        out = self.upsample(cls)

        res = []
        res.append(out)
        return tuple(res)


class _DenseAsppBlock(nn.Sequential):
    """ ConvNet block for building DenseASPP. """

    def __init__(self, input_num, num1, num2, dilation_rate, drop_out):
        super(_DenseAsppBlock, self).__init__()

        self.add_module('relu1', nn.ReLU(inplace=False)),
        self.add_module('conv1', nn.Conv2d(in_channels=input_num, out_channels=num1, kernel_size=1)),

        self.add_module('norm2', BatchNorm2d(num_features=num1)),
        self.add_module('relu2', nn.ReLU(inplace=False)),
        self.add_module('conv2', nn.Conv2d(in_channels=num1, out_channels=num2, kernel_size=3,
                                            dilation=dilation_rate, padding=dilation_rate)),
        self.add_module('norm2', BatchNorm2d(num_features=input_num)),

        self.drop_rate = drop_out

    def forward(self, _input):
        feature = super(_DenseAsppBlock, self).forward(_input)
        return feature


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('relu', nn.ReLU(inplace=False))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module('norm', BatchNorm2d(num_features=num_output_features)),


def densenet121_pretrained(pretrained_dir=None, pretrained= True, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16),
                     **kwargs)
    if pretrained:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = torch.load(pretrained_dir)
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
    return model

if __name__ == "__main__":
    model = SyncBNDenseASPP(12)
    image = torch.autograd.Variable(torch.randn(1, 3, 512, 512))
    out = model(image)
    print(out.size())

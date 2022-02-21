from functools import partial
import torch.nn as nn 


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvNormActivation(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, dilation=1, groups=1,
            norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU, inplace=True):
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, groups=groups, bias=norm_layer is None)]
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
        if activation_layer is not None:
            layers.append(activation_layer(inplace=True))

        super().__init__(*layers)


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, squeeze_channels, activation=nn.ReLU, scale_activation=nn.Sigmoid):
        super().__init__()
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.fc2 = nn.Conv2d(squeeze_channels, in_channels, kernel_size=1)
        self.relu = activation()
        self.scale_activation = scale_activation()

    def forward(self, x):
        scale = self.avgpool(x)
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        scale = self.scale_activation(scale)

        return scale * x 


class InvertedResidualConfig:

    def __init__(self, in_channels, expand_channels, out_channels, kernel_size, stride, dilation, use_se, activation, width_mult=1.):
        self.in_channels = self._adjust_channels(in_channels, width_mult)
        self.expand_channels = self._adjust_channels(expand_channels, width_mult)
        self.out_channels = self._adjust_channels(out_channels, width_mult)
        self.kernel_size = kernel_size
        self.stride = stride 
        self.dilation = dilation 
        self.use_se = use_se 
        self.use_hs = activation == 'HS'
        
    @staticmethod
    def _adjust_channels(channels, width_mult):
        return _make_divisible(channels * width_mult, 8)


class InvertedResidual(nn.Module):

    def __init__(self, cfg, norm_layer, se_layer=partial(SqueezeExcitation, scale_activation=nn.Hardsigmoid)):
        super().__init__()

        self.use_res_connect = cfg.stride == 1 and cfg.in_channels == cfg.out_channels 
        layers = []
        activation_layer = nn.Hardswish if cfg.use_hs else nn.ReLU 

        if cfg.expand_channels != cfg.in_channels:
            layers.append(
                ConvNormActivation(cfg.in_channels, cfg.expand_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=activation_layer)
            )

        # dw 
        stride = 1 if cfg.dilation > 1 else cfg.stride 
        layers.append(ConvNormActivation(
            cfg.expand_channels, cfg.expand_channels, kernel_size=cfg.kernel_size, stride=stride, 
            dilation=cfg.dilation, groups=cfg.expand_channels, norm_layer=norm_layer, activation_layer=activation_layer
        ))

        if cfg.use_se:
            squeeze_channels = _make_divisible(cfg.expand_channels // 4, 8)
            layers.append(se_layer(cfg.expand_channels, squeeze_channels))

        layers.append(ConvNormActivation(
            cfg.expand_channels, cfg.out_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=None
        ))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        if self.use_res_connect:
            out += x

        return out 


class MobileNetV3(nn.Module):

    def __init__(self, inverted_residual_setting, layer_getter_idx=[3, 6, 12, 16], block=None, norm_layer=None):
        super().__init__()

        self.layer_getter_idx = layer_getter_idx

        if block is None:
            block = InvertedResidual 
        
        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        layers = []
        in_channels = inverted_residual_setting[0].in_channels 
        layers.append(ConvNormActivation(3, in_channels, kernel_size=3, stride=2, norm_layer=norm_layer, activation_layer=nn.Hardswish))

        for cfg in inverted_residual_setting:
            layers.append(block(cfg, norm_layer))

        lastconv_input_channels = inverted_residual_setting[-1].out_channels 
        lastconv_output_channels = 6 * lastconv_input_channels
        layers.append(ConvNormActivation(
            lastconv_input_channels, lastconv_output_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.Hardswish
        ))

        self.num_channels = lastconv_output_channels

        self.features = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.features(x)


class MobileNetV3Large(MobileNetV3):
    def __init__(self):
        bneck_conf = partial(InvertedResidualConfig, width_mult=1.)
        inverted_residual_setting = [
            # in_channels, expand_channels, out_channels, kernel_size, stride, dilation, use_se, activation
            bneck_conf(16,  16,  16,  3, 1, 1, False, "RE"),
            bneck_conf(16,  64,  24,  3, 2, 1, False, "RE"),
            bneck_conf(24,  72,  24,  3, 1, 1, False, 'RE'),
            bneck_conf(24,  72,  40,  5, 2, 1, True,  "RE"),
            bneck_conf(40,  120, 40,  5, 1, 1, True,  "RE"),
            bneck_conf(40,  120, 40,  5, 1, 1, True,  "RE"),
            bneck_conf(40,  240, 80,  3, 2, 1, False, "HS"),
            bneck_conf(80,  200, 80,  3, 1, 1, False, "HS"),
            bneck_conf(80,  184, 80,  3, 1, 1, False, "HS"),
            bneck_conf(80,  184, 80,  3, 1, 1, False, "HS"),
            bneck_conf(80,  480, 112, 3, 1, 1, True,  "HS"),
            bneck_conf(112, 672, 112, 3, 1, 1, True,  "HS"),
            bneck_conf(112, 672, 160, 5, 2, 1, True,  "HS"),
            bneck_conf(160, 960, 160, 5, 1, 1, True,  "HS"),
            bneck_conf(160, 960, 160, 5, 1, 1, True,  "HS")
        ]
        layer_getter_idx = [3, 6, 12, 16]
        super().__init__(inverted_residual_setting, layer_getter_idx)


class MobileNetV3Small(MobileNetV3):
    def __init__(self):
        bneck_conf = partial(InvertedResidualConfig, width_mult=1.)
        inverted_residual_setting = [
            bneck_conf(16, 16,  16, 3, 2, 1, True,  "RE"),
            bneck_conf(16, 72,  24, 3, 2, 1, False, "RE"), 
            bneck_conf(24, 88,  24, 3, 1, 1, False, "RE"), 
            bneck_conf(24, 96,  40, 5, 2, 1, True,  "HS"), 
            bneck_conf(40, 240, 40, 5, 1, 1, True,  "HS"), 
            bneck_conf(40, 240, 40, 5, 1, 1, True,  "HS"), 
            bneck_conf(40, 120, 48, 5, 1, 1, True,  "HS"), 
            bneck_conf(48, 144, 48, 5, 1, 1, True,  "HS"), 
            bneck_conf(48, 288, 96, 5, 2, 1, True,  "HS"),
            bneck_conf(96, 576, 96, 5, 1, 1, True,  "HS"), 
            bneck_conf(96, 576, 96, 5, 1, 1, True,  "HS")
        ]
        layer_getter_idx = [1, 3, 8, 12]
        super().__init__(inverted_residual_setting, layer_getter_idx)

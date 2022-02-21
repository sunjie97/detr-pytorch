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

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, 
            dilation=1, groups=1, norm_layer=nn.BatchNorm2d, activation=nn.ReLU6):
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation

        conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, 
            padding=padding, dilation=dilation, groups=groups, bias=norm_layer is None)
        norm = norm_layer(out_channels)
        relu = activation(inplace=True)
        layers = [conv, norm, relu]

        super().__init__(*layers)


class InvertedResidual(nn.Module):

    def __init__(self, in_channels, out_channels, stride, expand_ratio, norm_layer=None):
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_res_connect = stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            layers.append(
                ConvNormActivation(in_channels, hidden_dim, kernel_size=1, norm_layer=norm_layer, activation=nn.ReLU6)
            )

        layers.extend([
            # dw 
            ConvNormActivation(hidden_dim, hidden_dim, kernel_size=3, stride=stride, groups=hidden_dim, norm_layer=norm_layer, activation=nn.ReLU6),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels)
        ])

        self.conv = nn.Sequential(*layers)
        self.out_channels = out_channels 

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobilenetV2(nn.Module):

    def __init__(self, width_mult=1., inverted_residual_setting=None, round_nearest=8, block=None, norm_layer=None):
        super().__init__()

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        in_channels = 32 
        out_channels = 1280
        self.block = block

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        in_channels = _make_divisible(in_channels * width_mult, round_nearest)
        self.num_channels = _make_divisible(out_channels * max(1., width_mult), round_nearest)

        features = [ConvNormActivation(3, in_channels, stride=2, norm_layer=norm_layer, activation=nn.ReLU6)]
        for t, c, n, s in inverted_residual_setting:
            out_channels = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1 
                features.append(block(in_channels, out_channels, stride=stride, expand_ratio=t, norm_layer=norm_layer))
                in_channels = out_channels 

        features.append(ConvNormActivation(in_channels, self.num_channels, kernel_size=1, norm_layer=norm_layer, activation=nn.ReLU6))
        self.features = nn.Sequential(*features)

        self._init_weights()
    
    def _init_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.features(x)

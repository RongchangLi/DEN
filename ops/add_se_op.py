import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class Addse_block(nn.Module):
    def __init__(self, block, n_segment, net_out_channels=256):
        super(Addse_block, self).__init__()
        self.block = block
        self.n_segment = n_segment

        num_squeezed_channels = net_out_channels // 16
        self._se_reduce_conv = nn.Conv3d(in_channels=net_out_channels, out_channels=num_squeezed_channels,
                                         kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False)
        self._se_reduce_bn=nn.BatchNorm3d(num_squeezed_channels)

        self._se_expand_conv = nn.Conv3d(in_channels=num_squeezed_channels, out_channels=net_out_channels,
                                         kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False)
        self._se_expand_bn=nn.BatchNorm3d(net_out_channels)


        # self._se_reduce_conv = nn.Conv2d(in_channels=net_out_channels, out_channels=num_squeezed_channels,
        #                                  kernel_size=1, padding=0, bias=True)
        #
        # self._ST_conv = nn.Conv3d(in_channels=num_squeezed_channels, out_channels=num_squeezed_channels,
        #                                  kernel_size=(3,1,1), padding=(1,0,0), bias=True)
        #
        # self._se_expand_conv = nn.Conv2d(in_channels=num_squeezed_channels, out_channels=net_out_channels,
        #                                  kernel_size=1, padding=0, bias=True)

        nn.init.constant_(self._se_reduce_conv.weight, 0)
        nn.init.constant_(self._se_reduce_bn.weight,0)
        nn.init.constant_(self._se_reduce_bn.bias, 1)

        nn.init.constant_(self._se_expand_conv.weight, 0)
        nn.init.constant_(self._se_expand_bn.weight, 0)
        nn.init.constant_(self._se_expand_bn.bias, 1)

        # nn.init.constant_(self._ST_conv.weight, 0)
        # nn.init.constant_(self._ST_conv.bias, 1)

        # self._swish = MemoryEfficientSwish()
        self.relu=nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.block(x)

        # squeezze
        x_squeezed = F.adaptive_avg_pool2d(x, 1)

        # reshape
        nt, c, h, w = x_squeezed.size()
        n_batch = nt // self.n_segment
        x_squeezed = x_squeezed.view(n_batch, self.n_segment, c, h, w).permute(0, 2, 1, 3, 4).contiguous()

        # reduce
        x_squeezed = self.relu(self._se_reduce_bn(self._se_reduce_conv(x_squeezed)))

        # expand
        x_squeezed = torch.sigmoid(self._se_expand_bn(self._se_expand_conv(x_squeezed)))

        # reshape
        x_squeezed = x_squeezed.permute(0, 2, 1, 3, 4).contiguous().view(nt, c, h, w)
        x = x_squeezed * x

        # x = self.block(x)

        # # squeezze
        # x_squeezed = F.adaptive_avg_pool2d(x, 1)
        #
        # # reduce
        # x_squeezed = self.relu(self._se_reduce_conv(x_squeezed))
        #
        # # reshape
        # nt, c, h, w = x_squeezed.size()
        # n_batch = nt // self.n_segment
        # x_squeezed = x_squeezed.view(n_batch, self.n_segment, c, h, w).permute(0, 2, 1, 3, 4).contiguous()
        #
        # #ST_conv
        # x_squeezed = self.relu(self._ST_conv(x_squeezed))
        #
        # # reshape
        # x_squeezed = x_squeezed.permute(0, 2, 1, 3, 4).contiguous().view(nt, c, h, w)
        #
        # # expand
        # x_squeezed = torch.sigmoid(self._se_expand_conv(x_squeezed))
        #
        #
        # x = x_squeezed * x

        return x


def add_se_layer(layer, n_segment):
    blocks = list(layer.children())
    print('=> Processing stage with {} blocks residual'.format(len(blocks)))
    for i, b in enumerate(blocks):
        blocks[i]= Addse_block(b, n_segment=n_segment, net_out_channels=b.conv3.out_channels)
    return nn.Sequential(*blocks)

    return block


def add_se(net, n_segment):
    net.layer1 = add_se_layer(net.layer1, n_segment)
    net.layer2 = add_se_layer(net.layer2, n_segment)
    net.layer3 = add_se_layer(net.layer3, n_segment)
    net.layer4 = add_se_layer(net.layer4, n_segment)


if __name__ == '__main__':
    model = EfficientNet.from_pretrained('efficientnet-b0')

    model._blocks = make_block_lstm(model._blocks, n_segment=8)

    total_params = sum(p.numel() for p in model.parameters())
    from torchvision.models import inception_v3

    a = 0

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.cuda import FloatTensor as ftens


class LSTM_FusionBlock(nn.Module):
    def __init__(self, in_channels, n_segment, n_div, n_layers=1):
        super(LSTM_FusionBlock, self).__init__()

        self.n_div = n_div
        self.fold = in_channels // n_div
        self.n_segment = n_segment

        self.lstm = nn.LSTM(input_size=2 * self.fold, hidden_size=2 * self.fold, num_layers=n_layers,
                            bidirectional=True)

    def forward(self, x):
        '''
        :param x: (nt, c, h, w)
        :return:(nt, c, h, w)
        '''
        self.lstm.flatten_parameters()
        # Reshaping to tensor of size [batch, frames, channels, H, W]
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        fold = self.fold

        x = x.view(n_batch, self.n_segment, c, h, w)

        # Squeeze and Recurrsion block
        tensor_shape = list(x.size())

        # Perform GlobalAvgPool operation frame-wise (for non-vector inputs)
        # x.size: n, t, c, h, w
        pool = F.avg_pool3d(x[:, :, :2 * fold], kernel_size=(1, tensor_shape[3], tensor_shape[4]), stride=1)

        # Reshaping to tensor of size [batch, frames, channels_fold]
        squeezed = pool.squeeze(-1).squeeze(-1)

        # [batch, frames, channels] -> [frames, batch, channels_fold]
        squeezed_temp = squeezed.permute(1, 0, 2)

        # use squeezed tensor as (1-layer) LSTM input
        lstm_out, _ = self.lstm(squeezed_temp)

        # [frames, batch, 4*channels_div] -> [batch, frames, 4*channels_fold]
        lstm_out = lstm_out.permute(1, 0, 2).unsqueeze(-1).unsqueeze(-1)

        out = torch.zeros_like(x)

        # comunicate with left, size of lstm_out: [batch, frames, 4*channels_div]
        # left_indces_sub1 = lstm_out[:, :-1, 2 * fold:3 * fold]
        # left_indeces_self = lstm_out[:, :-1, 3 * fold:]
        left_indces_sub1 = lstm_out[:, :, 2 * fold:3 * fold]
        left_indeces_self = lstm_out[:, :, 3 * fold:]
        next_value = torch.zeros_like(x[:, -1, :fold])
        for i in range(self.n_segment):
            now_index = self.n_segment - 1 - i
            now_value = x[:, now_index, :fold]
            replace_value = next_value * left_indeces_self[:, now_index] + now_value * left_indces_sub1[:, now_index]
            out[:, now_index, :fold] = replace_value.detach()
            next_value = replace_value

        # out[:, :-1, :fold] = x[:, 1:, :fold] * left_indeces_self + x[:, :-1, :fold] * left_indces_sub1  # shift left

        # comunicate with right
        # right_indces_plus1 = lstm_out[:, 1:, fold:2 * fold]
        # right_indeces_self = lstm_out[:, 1:, :fold]
        right_indces_plus1 = lstm_out[:, :, fold:2 * fold]
        right_indeces_self = lstm_out[:, :, :fold]
        last_value = torch.zeros_like(x[:, 0, fold:2 * fold])
        for i in range(self.n_segment):
            now_index = i
            now_value = x[:, now_index, fold:2 * fold]
            replace_value = last_value * right_indeces_self[:, now_index] + now_value * right_indces_plus1[:, now_index]
            out[:, now_index, fold:2 * fold] = replace_value
            last_value = replace_value

        # out[:, 1:, fold:2 * fold] = x[:, :-1, :fold] * right_indeces_self + x[:, 1:, :fold] * right_indces_plus1

        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift

        return out.view(nt, c, h, w)


class CONV1d_FusionBlock(nn.Module):
    def __init__(self, in_channels, n_segment, n_div):
        super(CONV1d_FusionBlock, self).__init__()

        self.n_div = n_div
        self.fold = in_channels // n_div
        self.n_segment = n_segment

        self.temporal_conv = nn.Conv3d(in_channels=2 * self.fold, out_channels=2 * self.fold, kernel_size=(3, 1, 1),
                                       padding=(1, 0, 0), stride=1, bias=True)

        # self.temporal_bn = nn.BatchNorm3d(2 * self.fold)
        #
        # self.relu = nn.ReLU(inplace=True)

        nn.init.constant_(self.temporal_conv.weight, 0)
        nn.init.constant_(self.temporal_conv.bias, 0)

    def forward(self, x):
        '''
        :param x: (nt, c, h, w)
        :return:(nt, c, h, w)
        '''

        # Reshaping to tensor of size [batch, frames, channels, H, W]
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment

        x = x.view(n_batch, self.n_segment, c, h, w).transpose(1, 2)  # n, c, t, h, w
        out_part = x[:, :2 * self.fold]

        out_part = self.temporal_conv(out_part)  # n, 2*fold, t, h, w
        # out_part = self.temporal_bn(out_part)
        # out_part = self.relu(out_part)

        out = torch.zeros_like(x)
        out[:, :2 * self.fold] = out_part
        out[:, 2 * self.fold:] = x[:, 2 * self.fold:]

        out = out.transpose(1, 2).contiguous().view(nt, c, h, w)

        return out


class CONV3d_FusionBlock(nn.Module):
    def __init__(self, in_channels, n_segment, n_div):
        super(CONV3d_FusionBlock, self).__init__()

        self.n_div = n_div
        self.fold = in_channels // n_div
        self.n_segment = n_segment

        self.temporal_conv = nn.Conv3d(in_channels=2 * self.fold, out_channels=2 * self.fold, kernel_size=(3, 3, 3),
                                       padding=(1, 1, 1), stride=1, bias=False)

        self.temporal_bn = nn.BatchNorm3d(2 * self.fold)

        self.relu = nn.ReLU(inplace=True)

        nn.init.constant_(self.temporal_conv.weight, 0)
        # nn.init.constant_(self.temporal_conv.bias, 0)

    def forward(self, x):
        '''
        :param x: (nt, c, h, w)
        :return:(nt, c, h, w)
        '''

        # Reshaping to tensor of size [batch, frames, channels, H, W]
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment

        x = x.view(n_batch, self.n_segment, c, h, w).transpose(1, 2)  # n, c, t, h, w
        out_part = x[:, :2 * self.fold]

        out_part = self.temporal_conv(out_part)  # n, 2*fold, t, h, w
        out_part = self.temporal_bn(out_part)
        out_part = self.relu(out_part)

        out = torch.zeros_like(x)
        out[:, :2 * self.fold] = out_part
        out[:, 2 * self.fold:] = x[:, 2 * self.fold:]

        out = out.transpose(1, 2).contiguous().view(nt, c, h, w)

        return out


class CONV1d_Channel_FusionBlock(nn.Module):
    def __init__(self, in_channels, n_segment, n_div):
        super(CONV1d_Channel_FusionBlock, self).__init__()

        self.n_div = n_div
        self.fold = in_channels // n_div
        self.n_segment = n_segment

        self.temporal_conv = nn.Conv3d(in_channels=2 * self.fold, out_channels=2 * self.fold, kernel_size=(3, 1, 1),
                                       padding=(1, 0, 0), stride=1, bias=True, groups=2 * self.fold)

        self.temporal_bn = nn.BatchNorm3d(2 * self.fold)
        #
        # self.relu = nn.ReLU(inplace=True)

        nn.init.constant_(self.temporal_conv.weight, 0)
        nn.init.constant_(self.temporal_conv.bias, 0)

    def forward(self, x):
        '''
        :param x: (nt, c, h, w)
        :return:(nt, c, h, w)
        '''

        # Reshaping to tensor of size [batch, frames, channels, H, W]
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment

        x = x.view(n_batch, self.n_segment, c, h, w).transpose(1, 2)  # n, c, t, h, w
        out_part = x[:, :2 * self.fold]

        out_part = self.temporal_conv(out_part)  # n, 2*fold, t, h, w
        out_part = self.temporal_bn(out_part)
        # out_part = self.relu(out_part)

        out = torch.zeros_like(x)
        out[:, :2 * self.fold] = out_part
        out[:, 2 * self.fold:] = x[:, 2 * self.fold:]

        out = out.transpose(1, 2).contiguous().view(nt, c, h, w)

        return out


class MOTION_ReplaceBlock(nn.Module):
    def __init__(self, in_channels, n_segment, n_div):
        super(MOTION_ReplaceBlock, self).__init__()

        self.n_div = n_div
        self.fold = in_channels // n_div
        self.n_segment = n_segment

        self.next_frame_conv = nn.Conv2d(in_channels=self.fold, out_channels=self.fold, kernel_size=3,
                                         padding=1, stride=1, bias=True)

        self.last_frame_conv = nn.Conv2d(in_channels=self.fold, out_channels=self.fold, kernel_size=3,
                                         padding=1, stride=1, bias=True)

        self.temporal_bn = nn.BatchNorm2d(in_channels)

        self.relu = nn.ReLU(inplace=True)

        nn.init.constant_(self.next_frame_conv.weight, 0)
        nn.init.constant_(self.next_frame_conv.bias, 0)
        nn.init.constant_(self.last_frame_conv.weight, 0)
        nn.init.constant_(self.last_frame_conv.bias, 0)

    def forward(self, x):
        '''
        :param x: (nt, c, h, w)
        :return:(nt, c, h, w)
        '''

        # Reshaping to tensor of size [batch, frames, channels, H, W]
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment

        x = x.view(n_batch, self.n_segment, c, h, w)  # n, t, c, h, w
        out = torch.zeros_like(x)

        out_part = x.view(nt, c, h, w)[:, :self.fold]  # nt,fold, h, w
        out_part = self.next_frame_conv(out_part)  # nt,fold, h, w
        out_part = out_part.view(n_batch, self.n_segment, self.fold, h, w)
        out[:, :-1, :self.fold] = out_part[:, 1:, :self.fold] - x[:, :-1, :self.fold]

        out_part = x.view(nt, c, h, w)[:, self.fold:2 * self.fold]  # nt,fold, h, w
        out_part = self.last_frame_conv(out_part)  # nt,fold, h, w
        out_part = out_part.view(n_batch, self.n_segment, self.fold, h, w)
        out[:, 1:, self.fold:2 * self.fold] = x[:, 1:, self.fold:2 * self.fold] - out_part[:, :-1, :self.fold]

        out[:, :, 2 * self.fold:] = x[:, :, 2 * self.fold:]

        out = out.view(nt, c, h, w)

        out = self.temporal_bn(out)

        return out


class MOTION_ReplaceBlock_B(nn.Module):
    """
    using diff
    """

    def __init__(self, in_channels, n_segment, n_div):
        super(MOTION_ReplaceBlock_B, self).__init__()

        self.n_div = n_div
        self.fold = in_channels // n_div
        self.n_segment = n_segment

        # self.next_frame_conv = nn.Conv2d(in_channels=self.fold, out_channels=self.fold // 2, kernel_size=3,
        #                                  padding=1, stride=1, bias=True)
        #
        # self.last_frame_conv = nn.Conv2d(in_channels=self.fold, out_channels=self.fold // 2, kernel_size=3,
        #                                  padding=1, stride=1, bias=True)

        # self.temporal_bn1 = nn.BatchNorm2d(self.fold)
        # self.temporal_bn2 = nn.BatchNorm2d(self.fold)

        # nn.init.constant_(self.next_frame_conv.weight, 0)
        # nn.init.constant_(self.next_frame_conv.bias, 0)
        # nn.init.constant_(self.last_frame_conv.weight, 0)
        # nn.init.constant_(self.last_frame_conv.bias, 0)

    def forward(self, x):
        '''
        :param x: (nt, c, h, w)
        :return:(nt, c, h, w)
        '''
        #
        # Reshaping to tensor of size [batch, frames, channels, H, W]
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment

        x = x.view(n_batch, self.n_segment, c, h, w)  # n, t, c, h, w
        out = torch.zeros_like(x)

        # out_part = x[:, :, :self.fold]  # n,t,fold, h, w
        out[:, :-1, :self.fold ] = x[:, 1:, :self.fold] - x[:, :-1, :self.fold]

        # out_part = x.view(nt, c, h, w)[:, :self.fold]  # nt,fold, h, w
        # out_part = self.last_frame_conv(out_part)  # nt,fold, h, w
        # out_part = out_part.view(n_batch, self.n_segment, self.fold // 2, h, w)
        out[:, 1:, self.fold :2*self.fold] = x[:, 1:, self.fold :2*self.fold] - x[:, :-1,  self.fold :2*self.fold]

        out[:, :, 2*self.fold:] = x[:, :, 2*self.fold:]
        #
        out = out.view(nt, c, h, w)

        return out


class MOTION_ReplaceBlock_C(nn.Module):
    """
    r(t)+m(t+1)

    """

    def __init__(self, in_channels, n_segment, n_div):
        super(MOTION_ReplaceBlock_C, self).__init__()

        self.n_div = n_div
        self.fold = in_channels // n_div
        self.n_segment = n_segment

        self.frame_conv = nn.Conv2d(in_channels=2 * self.fold, out_channels=2 * self.fold, kernel_size=3,
                                    padding=1, stride=1, bias=True, groups=2)

        nn.init.constant_(self.frame_conv.weight, 0)
        nn.init.constant_(self.frame_conv.bias, 0)

    def lshift_zeroPad(self, x):
        # X:n_batch, self.n_segment, self.fold, h, w
        return torch.cat((x[:, 1:], ftens(x.size(0), 1, x.size(2), x.size(3), x.size(4)).fill_(0)), dim=1)

    def rshift_zeroPad(self, x):
        return torch.cat((ftens(x.size(0), 1, x.size(2), x.size(3), x.size(4)).fill_(0), x[:, :-1]), dim=1)

    def forward(self, x):
        '''
        :param x: (nt, c, h, w)
        :return:(nt, c, h, w)
        '''

        # Reshaping to tensor of size [batch, frames, channels, H, W]
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment

        out_part = x[:, :2 * self.fold]  # nt,fold, h, w
        out_part = self.frame_conv(out_part)  # nt,2*fold, h, w

        x_group1 = x[:, : self.fold]
        x_group2 = x[:, self.fold:2 * self.fold]
        y_group1 = out_part[:, : self.fold]
        y_group2 = out_part[:, self.fold:2 * self.fold]
        r_group1 = x_group1 - y_group1
        r_group2 = x_group2 - y_group2

        y_group1 = self.lshift_zeroPad(y_group1.view(n_batch, self.n_segment, self.fold, h, w)).view(nt, self.fold, h,
                                                                                                     w) + r_group1
        y_group2 = self.rshift_zeroPad(y_group2.view(n_batch, self.n_segment, self.fold, h, w)).view(nt, self.fold, h,
                                                                                                     w) + r_group2

        y = torch.cat((y_group1, y_group2, x[:, 2 * self.fold:]), dim=1)

        return y


# class MOTION_ReplaceBlock_C(nn.Module):
#     """
#     r(t)+m(t+1)
#
#     """
#
#     def __init__(self, in_channels, n_segment, n_div):
#         super(MOTION_ReplaceBlock_C, self).__init__()
#
#         self.n_div = n_div
#         self.fold = in_channels // n_div
#         self.n_segment = n_segment
#
#         self.frame_conv = nn.Conv2d(in_channels=2 * self.fold, out_channels=2 * self.fold, kernel_size=3,
#                                     padding=1, stride=1, bias=True, groups=2)
#
#         nn.init.constant_(self.frame_conv.weight, 0)
#         nn.init.constant_(self.frame_conv.bias, 0)
#
#     def lshift_zeroPad(self, x):
#         return torch.cat((x[:, :, 1:], ftens(x.size(0), x.size(1), 1, x.size(3), x.size(4)).fill_(0)), dim=2)
#
#     def rshift_zeroPad(self, x):
#         return torch.cat((ftens(x.size(0), x.size(1), 1, x.size(3), x.size(4)).fill_(0), x[:, :, :-1]), dim=2)
#
#     def forward(self, x):
#         '''
#         :param x: (nt, c, h, w)
#         :return:(nt, c, h, w)
#         '''
#
#         # Reshaping to tensor of size [batch, frames, channels, H, W]
#         nt, c, h, w = x.size()
#         n_batch = nt // self.n_segment
#
#         out_part = x[:, :2 * self.fold]  # nt,fold, h, w
#         out_part = self.frame_conv(out_part)  # nt,2*fold, h, w
#
#         x = x.view(n_batch, self.n_segment, c, h, w).permute(0, 2, 1, 3, 4).contiguous()
#         out_part = out_part.view(n_batch, self.n_segment, 2 * self.fold, h, w).permute(0, 2, 1, 3, 4).contiguous()
#
#         x_group1 = x[:, : self.fold]
#         x_group2 = x[:, self.fold:2 * self.fold]
#         y_group1 = out_part[:, : self.fold]
#         y_group2 = out_part[:, self.fold:2 * self.fold]
#         r_group1 = x_group1 - y_group1
#         r_group2 = x_group2 - y_group2
#
#         y_group1 = self.lshift_zeroPad(y_group1) + r_group1
#         y_group2 = self.rshift_zeroPad(y_group2) + r_group2
#
#         # y_group1 = y_group1.view(n_batch, 2, self.fold // 2, self.n_segment, h, w).permute(0, 2, 1, 3, 4,
#         #                                                                                       5)
#         # y_group2 = y_group2.view(n_batch, 2, self.fold // 2, self.n_segment, h, w).permute(0, 2, 1, 3, 4,
#         #                                                                                       5)
#
#         y = torch.cat((y_group1.contiguous().view(n_batch, self.fold, self.n_segment, h, w),
#                        y_group2.contiguous().view(n_batch, self.fold, self.n_segment, h, w),
#                        x[:, 2 * self.fold:]), dim=1)
#
#         return y.permute(0, 2, 1, 3, 4).contiguous().view(nt, c, h, w)
#

# class MOTION_ReplaceBlock_C(nn.Module):
#
#     """
#     r(t)+m(t+1)
#
#     """
#     def __init__(self, in_channels, n_segment, n_div):
#         super(MOTION_ReplaceBlock_C, self).__init__()
#
#         self.n_div = n_div
#         self.fold = in_channels // n_div
#         self.n_segment = n_segment
#
#         self.next_frame_conv = nn.Conv2d(in_channels=self.fold, out_channels=self.fold, kernel_size=3,
#                                          padding=1, stride=1, bias=True)
#
#         self.last_frame_conv = nn.Conv2d(in_channels=self.fold, out_channels=self.fold, kernel_size=3,
#                                          padding=1, stride=1, bias=True)
#
#         nn.init.constant_(self.next_frame_conv.weight, 0)
#         nn.init.constant_(self.next_frame_conv.bias, 0)
#         nn.init.constant_(self.last_frame_conv.weight, 0)
#         nn.init.constant_(self.last_frame_conv.bias, 0)
#
#     def forward(self, x):
#         '''
#         :param x: (nt, c, h, w)
#         :return:(nt, c, h, w)
#         '''
#
#         # Reshaping to tensor of size [batch, frames, channels, H, W]
#         nt, c, h, w = x.size()
#         n_batch = nt // self.n_segment
#
#         x = x.view(n_batch, self.n_segment, c, h, w)  # n, t, c, h, w
#         out = torch.zeros_like(x)
#
#         out_part = x.view(nt, c, h, w)[:, :self.fold]  # nt,fold, h, w
#         out_part = self.next_frame_conv(out_part)  # nt,fold, h, w
#         out_part = out_part.view(n_batch, self.n_segment, self.fold, h, w)
#         out[:, :, :self.fold] = x[:, :, :self.fold] - out_part[:, :, :self.fold]  # r(t)
#         out[:, :-1, :self.fold] = out[:, :-1, :self.fold] + out_part[:, 1:, :self.fold]  # r(t)+x_mean(t+1)
#
#         out_part = x.view(nt, c, h, w)[:, self.fold:2 * self.fold]  # nt,fold, h, w
#         out_part = self.last_frame_conv(out_part)  # nt,fold, h, w
#         out_part = out_part.view(n_batch, self.n_segment, self.fold, h, w)
#         out[:, :, self.fold:2 * self.fold] = x[:, :, self.fold:2 * self.fold] - out_part[:, :, :self.fold]
#         out[:, 1:, self.fold:2 * self.fold] = out[:, 1:, self.fold:2 * self.fold] + out_part[:, :-1,
#                                                                                     :self.fold]  # r(t)+x_mean(t-1)
#
#         out[:, :, 2 * self.fold:] = x[:, :, 2 * self.fold:]
#
#         out = out.view(nt, c, h, w)
#
#         return out

class MOTION_ReplaceBlock_D(nn.Module):
    """
    reuse conv

    """

    def __init__(self, in_channels, n_segment, n_div):
        super(MOTION_ReplaceBlock_D, self).__init__()

        self.n_div = n_div
        self.fold = in_channels // n_div
        self.n_segment = n_segment

        self.frame_conv = nn.Conv2d(in_channels=self.fold, out_channels=self.fold, kernel_size=3,
                                    padding=1, stride=1, bias=True)

        # self.temporal_bn=nn.BatchNorm2d(in_channels)

        self.relu = nn.ReLU(inplace=True)

        nn.init.constant_(self.frame_conv.weight, 0)
        nn.init.constant_(self.frame_conv.bias, 0)

    def forward(self, x):
        '''
        :param x: (nt, c, h, w)
        :return:(nt, c, h, w)
        '''

        # Reshaping to tensor of size [batch, frames, channels, H, W]
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment

        x = x.view(n_batch, self.n_segment, c, h, w)  # n, t, c, h, w
        out = torch.zeros_like(x)

        out_part = x.view(nt, c, h, w)[:, :self.fold]  # nt,fold, h, w
        out_part = self.frame_conv(out_part)  # nt,fold, h, w
        out_part = out_part.view(n_batch, self.n_segment, self.fold, h, w)
        out[:, :-1, :self.fold] = out_part[:, 1:, :self.fold] - x[:, :-1, :self.fold]

        out_part = x.view(nt, c, h, w)[:, self.fold:2 * self.fold]  # nt,fold, h, w
        out_part = self.frame_conv(out_part)  # nt,fold, h, w
        out_part = out_part.view(n_batch, self.n_segment, self.fold, h, w)
        out[:, 1:, self.fold:2 * self.fold] = x[:, 1:, self.fold:2 * self.fold] - out_part[:, :-1, :self.fold]

        out[:, :, 2 * self.fold:] = x[:, :, 2 * self.fold:]

        out = out.view(nt, c, h, w)
        # out=self.temporal_bn(out)

        return out


class MOTION_Channel_ReplaceBlock(nn.Module):
    def __init__(self, in_channels, n_segment, n_div):
        super(MOTION_Channel_ReplaceBlock, self).__init__()

        self.n_div = n_div
        self.fold = in_channels // n_div
        self.n_segment = n_segment

        # self.next_fusion = nn.Conv2d(in_channels=self.fold, out_channels=self.fold, kernel_size=1,
        #                              stride=1, bias=False)
        # self.next_bn = nn.BatchNorm2d(self.fold)
        self.next_frame_conv = nn.Conv2d(in_channels=self.fold, out_channels=self.fold, kernel_size=3,
                                         padding=1, stride=1, bias=False, groups=self.fold)

        # self.last_fusion = nn.Conv2d(in_channels=self.fold, out_channels=self.fold, kernel_size=1,
        #                              stride=1, bias=False)
        # self.last_bn = nn.BatchNorm2d(self.fold)
        self.last_frame_conv = nn.Conv2d(in_channels=self.fold, out_channels=self.fold, kernel_size=3,
                                         padding=1, stride=1, bias=False, groups=self.fold)

        # self.temporal_bn = nn.BatchNorm2d(2 * self.fold)
        #
        # self.relu = nn.ReLU(inplace=True)

        # nn.init.constant_(self.next_frame_conv.weight, 0)
        # nn.init.constant_(self.next_frame_conv.bias, 0)
        # nn.init.constant_(self.last_frame_conv.weight, 0)
        # nn.init.constant_(self.last_frame_conv.bias, 0)

    def forward(self, x):
        '''
        :param x: (nt, c, h, w)
        :return:(nt, c, h, w)
        '''

        # Reshaping to tensor of size [batch, frames, channels, H, W]
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment

        x = x.view(n_batch, self.n_segment, c, h, w)  # n, t, c, h, w
        out = torch.zeros_like(x)

        out_part = x.view(nt, c, h, w)[:, :self.fold]  # nt,fold, h, w

        # out_part = self.next_fusion(out_part)
        # out_part = self.next_bn(out_part)
        # out_part = self.relu(out_part)

        out_part = self.next_frame_conv(out_part)  # nt,fold, h, w
        out_part = out_part.view(n_batch, self.n_segment, self.fold, h, w)
        out[:, :-1, :self.fold] = out_part[:, 1:, :self.fold] - x[:, :-1, :self.fold]

        out_part = x.view(nt, c, h, w)[:, self.fold:2 * self.fold]  # nt,fold, h, w

        # out_part = self.last_fusion(out_part)
        # out_part = self.last_bn(out_part)
        # out_part = self.relu(out_part)

        out_part = self.last_frame_conv(out_part)  # nt,fold, h, w
        out_part = out_part.view(n_batch, self.n_segment, self.fold, h, w)
        out[:, 1:, self.fold:2 * self.fold] = x[:, 1:, self.fold:2 * self.fold] - out_part[:, :-1, :self.fold]

        out[:, :, 2 * self.fold:] = x[:, :, 2 * self.fold:]

        out = out.view(nt, c, h, w)

        return out


class SELECT_fusion_block(nn.Module):
    def __init__(self, in_channels, n_segment, n_div):
        super(SELECT_fusion_block, self).__init__()

        self.n_div = n_div
        self.fold = in_channels // n_div
        self.n_segment = n_segment

        self.select_op = nn.MaxPool2d(kernel_size=3, padding=1, stride=1)

        self.fusion_conv = nn.Conv2d(in_channels=3 * self.fold, out_channels=self.fold, kernel_size=1,
                                     padding=0, stride=1, bias=True)

        # self.temporal_bn = nn.BatchNorm2d(2 * self.fold)
        #
        # self.relu = nn.ReLU(inplace=True)

        nn.init.constant_(self.fusion_conv.weight, 0)
        nn.init.constant_(self.fusion_conv.bias, 0)

    def forward(self, x):
        '''
        :param x: (nt, c, h, w)
        :return:(nt, c, h, w)
        '''

        # Reshaping to tensor of size [batch, frames, channels, H, W]
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment

        x = x.view(n_batch, self.n_segment, c, h, w)  # n, t, c, h, w
        out = torch.zeros_like(x)

        out_part = x.view(nt, c, h, w)[:, :self.fold]  # nt,fold, h, w
        out_part_select = self.select_op(out_part)  # nt,fold, h, w
        out_part_select = out_part_select.view(n_batch, self.n_segment, self.fold, h, w)
        out_part = out_part.view(n_batch, self.n_segment, self.fold, h, w)

        select_left = torch.zeros_like(out_part_select)
        select_right = torch.zeros_like(out_part_select)
        select_left[:, 1:] = out_part_select[:, :-1]
        select_right[:, :-1] = out_part_select[:, 1:]
        out_part = torch.cat([select_left, out_part, select_right], dim=2)

        out_part = out_part.view(nt, -1, h, w)
        out_part = self.fusion_conv(out_part)
        out_part = out_part.view(n_batch, self.n_segment, self.fold, h, w)

        out[:, :, :self.fold] = out_part[:, :, :self.fold]

        out[:, :, self.fold:] = x[:, :, self.fold:]

        out = out.view(nt, c, h, w)

        return out


class LIGHT_MOTION_ReplaceBlock(nn.Module):
    def __init__(self, in_channels, n_segment, n_div):
        super(LIGHT_MOTION_ReplaceBlock, self).__init__()

        self.n_div = n_div
        self.fold = in_channels // n_div
        self.n_segment = n_segment

        self.channel_down = nn.Conv2d(in_channels=in_channels, out_channels=2 * self.fold, kernel_size=1, bias=True)
        self.channel_down_bn = nn.BatchNorm2d(2 * self.fold)
        self.relu = nn.ReLU(inplace=True)

        self.next_frame_conv = nn.Conv2d(in_channels=self.fold, out_channels=self.fold, kernel_size=3,
                                         padding=1, stride=1, bias=True)

        self.last_frame_conv = nn.Conv2d(in_channels=self.fold, out_channels=self.fold, kernel_size=3,
                                         padding=1, stride=1, bias=True)

        nn.init.constant_(self.next_frame_conv.weight, 0)
        nn.init.constant_(self.next_frame_conv.bias, 0)
        nn.init.constant_(self.last_frame_conv.weight, 0)
        nn.init.constant_(self.last_frame_conv.bias, 0)

    def forward(self, x):
        '''
        :param x: (nt, c, h, w)
        :return:(nt, c, h, w)
        '''

        # save the x
        identity = x

        # reduce the channels
        x = self.channel_down(x)
        x = self.channel_down_bn(x)
        x = self.relu(x)

        # Reshaping to tensor of size [batch, frames, channels, H, W]
        nt, c, h, w = identity.size()
        n_batch = nt // self.n_segment

        identity = identity.view(n_batch, self.n_segment, c, h, w)  # n, t, c, h, w
        out = torch.zeros_like(identity)

        out_part = x[:, :self.fold]  # nt,fold, h, w
        out_part = self.next_frame_conv(out_part)  # nt,fold, h, w
        out_part = out_part.view(n_batch, self.n_segment, self.fold, h, w)
        out[:, :-1, :self.fold] = out_part[:, 1:, :self.fold] - x.view(n_batch, self.n_segment, -1, h, w)[:, :-1,
                                                                :self.fold]

        out_part = x[:, self.fold:2 * self.fold]  # nt,fold, h, w
        out_part = self.last_frame_conv(out_part)  # nt,fold, h, w
        out_part = out_part.view(n_batch, self.n_segment, self.fold, h, w)
        out[:, 1:, self.fold:2 * self.fold] = x.view(n_batch, self.n_segment, -1, h, w)[:, 1:,
                                              self.fold:2 * self.fold] - out_part[:, :-1, :self.fold]

        out[:, :, 2 * self.fold:] = identity[:, :, 2 * self.fold:]

        out = out.view(nt, c, h, w)
        # out=self.temporal_bn(out)

        return out

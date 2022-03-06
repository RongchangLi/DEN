# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import torch
import torch.nn as nn
import torch.nn.functional as F

from ops.part_fusion import LSTM_FusionBlock, CONV1d_FusionBlock, CONV1d_Channel_FusionBlock, MOTION_ReplaceBlock, \
    CONV3d_FusionBlock, MOTION_Channel_ReplaceBlock, SELECT_fusion_block, MOTION_ReplaceBlock_C, \
    MOTION_ReplaceBlock_B, LIGHT_MOTION_ReplaceBlock, MOTION_ReplaceBlock_D


class TemporalShift(nn.Module):
    def __init__(self, net, n_segment=3, n_div=8, inplace=False, comu_type='replace', type='A', borrow_BN=False):
        super(TemporalShift, self).__init__()
        self.net = net
        self.n_segment = n_segment
        self.fold_div = n_div
        self.inplace = inplace
        self.type = type

        self.comu_type = comu_type
        self.borrow_BN = borrow_BN

        if self.borrow_BN:
            inchannels = net.out_channels
            print('=> Using Borrow BN layers')
        else:
            inchannels = net.in_channels

        if self.comu_type == 'replace':
            print('=> Using TSM')
        elif self.comu_type == 'lstm_inblock':
            print('=> Using LSTM_FusionBlock after res-convolution, layer number is 2')
            self.shift_block = LSTM_FusionBlock(in_channels=net.out_channels, n_segment=n_segment, n_div=n_div,
                                                n_layers=2)
        elif self.comu_type == 'lstm_outblock':
            print('=> Using LSTM_FusionBlock before res-convolution, layer number is 1')
            self.shift_block = LSTM_FusionBlock(in_channels=net.in_channels, n_segment=n_segment, n_div=n_div,
                                                n_layers=1)
        elif self.comu_type == 'conv1d_block':
            print('=> Using COV1d_FusionBlock')
            self.shift_block = CONV1d_FusionBlock(in_channels=net.in_channels, n_segment=n_segment, n_div=n_div)

        elif self.comu_type == 'conv1d_channel_block':
            print('=> Using COV1d_FusionBlock')
            self.shift_block = CONV1d_Channel_FusionBlock(in_channels=net.in_channels, n_segment=n_segment, n_div=n_div)

        elif self.comu_type == 'conv3d_block':
            print('=> Using COV3d_FusionBlock')
            self.shift_block = CONV3d_FusionBlock(in_channels=net.in_channels, n_segment=n_segment, n_div=n_div)

        elif 'motion_replace' in self.comu_type and 'light' not in self.comu_type and 'depthwise' not in self.comu_type:
            if self.type == 'A':
                print('=> Using MOTION_ReplaceBlock_A')
                self.shift_block = MOTION_ReplaceBlock(in_channels=inchannels, n_segment=n_segment, n_div=n_div)
            elif self.type == 'B':
                print('=> Using MOTION_ReplaceBlock_B')
                self.shift_block = MOTION_ReplaceBlock_B(in_channels=inchannels, n_segment=n_segment, n_div=n_div)
            elif self.type == 'C':
                print('=> Using MOTION_ReplaceBlock_C')
                self.shift_block = MOTION_ReplaceBlock_C(in_channels=inchannels, n_segment=n_segment,
                                                         n_div=n_div)
            elif self.type == 'D':
                print('=> Using MOTION_ReplaceBlock_D')
                self.shift_block = MOTION_ReplaceBlock_D(in_channels=inchannels, n_segment=n_segment,
                                                         n_div=n_div)

            else:
                raise NotImplementedError

        elif self.comu_type == 'motion_replace_depthwise':
            print('=> Using MOTION_Channel_ReplaceBlock')
            self.shift_block = MOTION_Channel_ReplaceBlock(in_channels=net.in_channels, n_segment=n_segment,
                                                           n_div=n_div)
        elif self.comu_type == 'motion_and_shift':
            print('=> Using MOTION_Channel_ReplaceBlock')
            self.shift_block = MOTION_Channel_ReplaceBlock(in_channels=net.in_channels, n_segment=n_segment,
                                                           n_div=n_div)
        elif self.comu_type == 'select_fusion':
            print('=> Using SELECT_Fusion')
            self.shift_block = SELECT_fusion_block(in_channels=net.in_channels, n_segment=n_segment, n_div=n_div)

        elif self.comu_type == 'light_motion_replace':
            print('=> Using LIGHT_MOTION_ReplaceBlock')
            self.shift_block = LIGHT_MOTION_ReplaceBlock(in_channels=net.in_channels, n_segment=n_segment, n_div=n_div)

        else:
            raise NotImplementedError

        if inplace:
            print('=> Using in-place shift...')
        print('=> Using fold div: {}'.format(self.fold_div))

    def forward(self, x):
        if self.comu_type == 'replace':
            x = self.shift(x, self.n_segment, fold_div=self.fold_div, inplace=self.inplace)
            return self.net(x)
        if self.borrow_BN:
            x = self.net(x)
            return self.shift_block(x)
        else:
            x = self.shift_block(x)
            return self.net(x)

    @staticmethod
    def shift(x, n_segment, fold_div=3, inplace=False, ):
        nt, c, h, w = x.size()
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, c, h, w)

        fold = c // fold_div
        if inplace:
            # Due to some out of order error when performing parallel computing. 
            # May need to write a CUDA kernel.
            raise NotImplementedError
            # out = InplaceShift.apply(x, fold)
        else:

            out = torch.zeros_like(x)
            out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
            out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
            out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift

        return out.view(nt, c, h, w)

    @staticmethod
    def motion_shift(x, n_segment, fold_div=3, inplace=False, ):
        nt, c, h, w = x.size()
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, c, h, w)

        fold = c // fold_div
        s_fold = fold // 2
        if inplace:
            # Due to some out of order error when performing parallel computing.
            # May need to write a CUDA kernel.
            raise NotImplementedError
            # out = InplaceShift.apply(x, fold)
        else:

            out = torch.zeros_like(x)
            out[:, :-1, 2 * fold:2 * fold + s_fold] = x[:, 1:, 2 * fold:2 * fold + s_fold]  # shift left
            out[:, 1:, 2 * fold + s_fold:2 * fold + 2 * s_fold] = x[:, :-1,
                                                                  2 * fold + s_fold:2 * fold + 2 * s_fold]  # shift right
            out[:, :, 2 * fold + 2 * s_fold:] = x[:, :, 2 * fold + 2 * s_fold:]  # not shift

        return out.view(nt, c, h, w)


def make_temporal_shift(net, n_segment, n_div=8, place='blockres', temporal_pool=False, comu_type='replace'):
    if temporal_pool:
        n_segment_list = [n_segment, n_segment // 2, n_segment // 2, n_segment // 2]
    else:
        n_segment_list = [n_segment] * 4
    assert n_segment_list[-1] > 0
    print('=> n_segment per stage: {}'.format(n_segment_list))

    import torchvision
    if isinstance(net, torchvision.models.ResNet):

        ttype = 'A'
        if 'B' in comu_type:
            ttype = 'B'
        elif 'C' in comu_type:
            ttype = 'C'
        elif 'D' in comu_type:
            ttype = 'D'
        else:
            print("==> using comutype:{}".format(comu_type))

        if 'BB' in place:
            borrow_BN = True
            print("=>Borrow BN!!!")
        else:
            borrow_BN = False

        if 'inblockres' in place and borrow_BN:
            n_round = 1
            if len(list(net.layer3.children())) >= 23:
                n_round = 2
                print('=> Using n_round {} to insert temporal shift'.format(n_round))

            def make_block_temporal(stage, this_segment, comu_type='replace'):
                blocks = list(stage.children())
                print('=> Processing stage with {} blocks residual'.format(len(blocks)))
                for i, b in enumerate(blocks):
                    if i % n_round == 0:
                        blocks[i].conv1 = TemporalShift(b.conv1, n_segment=this_segment, n_div=n_div
                                                        , comu_type=comu_type, type=ttype, borrow_BN=borrow_BN)
                return nn.Sequential(*blocks)

            net.layer1 = make_block_temporal(net.layer1, n_segment_list[0], comu_type=comu_type)
            net.layer2 = make_block_temporal(net.layer2, n_segment_list[1], comu_type=comu_type)
            net.layer3 = make_block_temporal(net.layer3, n_segment_list[2], comu_type=comu_type)
            net.layer4 = make_block_temporal(net.layer4, n_segment_list[3], comu_type=comu_type)
        elif 'inblockres' in place and not borrow_BN:
            n_round = 1
            if len(list(net.layer3.children())) >= 23:
                n_round = 2
                print('=> Using n_round {} to insert temporal shift'.format(n_round))

            def make_block_temporal(stage, this_segment, comu_type='replace'):
                blocks = list(stage.children())
                print('=> Processing stage with {} blocks residual'.format(len(blocks)))
                for i, b in enumerate(blocks):
                    if i % n_round == 0:
                        blocks[i].conv2 = TemporalShift(b.conv2, n_segment=this_segment, n_div=n_div
                                                        , comu_type=comu_type, type=ttype)
                return nn.Sequential(*blocks)

            net.layer1 = make_block_temporal(net.layer1, n_segment_list[0], comu_type=comu_type)
            net.layer2 = make_block_temporal(net.layer2, n_segment_list[1], comu_type=comu_type)
            net.layer3 = make_block_temporal(net.layer3, n_segment_list[2], comu_type=comu_type)
            net.layer4 = make_block_temporal(net.layer4, n_segment_list[3], comu_type=comu_type)

        elif 'blockres' in place:
            n_round = 1
            if len(list(net.layer3.children())) >= 23:
                n_round = 2
                print('=> Using n_round {} to insert temporal shift'.format(n_round))

            def make_block_temporal(stage, this_segment, comu_type='replace'):
                blocks = list(stage.children())
                print('=> Processing stage with {} blocks residual'.format(len(blocks)))
                for i, b in enumerate(blocks):
                    if i % n_round == 0:
                        blocks[i].conv1 = TemporalShift(b.conv1, n_segment=this_segment, n_div=n_div
                                                        , comu_type=comu_type, type=ttype)
                return nn.Sequential(*blocks)

            net.layer1 = make_block_temporal(net.layer1, n_segment_list[0], comu_type=comu_type)
            net.layer2 = make_block_temporal(net.layer2, n_segment_list[1], comu_type=comu_type)
            net.layer3 = make_block_temporal(net.layer3, n_segment_list[2], comu_type=comu_type)
            net.layer4 = make_block_temporal(net.layer4, n_segment_list[3], comu_type=comu_type)
        else:
            raise NotImplementedError(place)
    else:
        raise NotImplementedError(place)


if __name__ == '__main__':
    # test inplace shift v.s. vanilla shift
    tsm1 = TemporalShift(nn.Sequential(), n_segment=8, n_div=8, inplace=False)
    tsm2 = TemporalShift(nn.Sequential(), n_segment=8, n_div=8, inplace=True)

    print('=> Testing CPU...')
    # test forward
    with torch.no_grad():
        for i in range(10):
            x = torch.rand(2 * 8, 3, 224, 224)
            y1 = tsm1(x)
            y2 = tsm2(x)
            assert torch.norm(y1 - y2).item() < 1e-5

    # test backward
    with torch.enable_grad():
        for i in range(10):
            x1 = torch.rand(2 * 8, 3, 224, 224)
            x1.requires_grad_()
            x2 = x1.clone()
            y1 = tsm1(x1)
            y2 = tsm2(x2)
            grad1 = torch.autograd.grad((y1 ** 2).mean(), [x1])[0]
            grad2 = torch.autograd.grad((y2 ** 2).mean(), [x2])[0]
            assert torch.norm(grad1 - grad2).item() < 1e-5

    print('=> Testing GPU...')
    tsm1.cuda()
    tsm2.cuda()
    # test forward
    with torch.no_grad():
        for i in range(10):
            x = torch.rand(2 * 8, 3, 224, 224).cuda()
            y1 = tsm1(x)
            y2 = tsm2(x)
            assert torch.norm(y1 - y2).item() < 1e-5

    # test backward
    with torch.enable_grad():
        for i in range(10):
            x1 = torch.rand(2 * 8, 3, 224, 224).cuda()
            x1.requires_grad_()
            x2 = x1.clone()
            y1 = tsm1(x1)
            y2 = tsm2(x2)
            grad1 = torch.autograd.grad((y1 ** 2).mean(), [x1])[0]
            grad2 = torch.autograd.grad((y2 ** 2).mean(), [x2])[0]
            assert torch.norm(grad1 - grad2).item() < 1e-5
    print('Test passed.')

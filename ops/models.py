# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

from torch import nn

from ops.basic_ops import ConsensusModule
from ops.transforms import *
from torch.nn.init import normal_, constant_

import collections


def load_followed_model(base_model, followed_model_path):
    print(("=> load pretrained model '{}'".format(followed_model_path)))
    sd = torch.load(followed_model_path)
    sd = sd['state_dict']
    model_dict = base_model.state_dict()
    sd_base_model_dict = collections.OrderedDict()

    replace_dict = []
    print('=> Load after remove module.base_model')
    for k, v in sd.items():
        if k in model_dict or k.replace('module.base_model.', '').replace('.net', '') in model_dict:
            replace_dict.append(
                (k, k.replace('module.base_model.', '').replace('.net', '')))

    for k, k_new in replace_dict:
        sd_base_model_dict[k_new] = sd.pop(k)

    model_dict.update(sd_base_model_dict)
    base_model.load_state_dict(model_dict)


class TSN(nn.Module):
    def __init__(self, num_class, num_segments, modality,
                 base_model='resnet101', new_length=None,
                 consensus_type='avg', before_softmax=True,
                 dropout=0.8, img_feature_dim=256,
                 crop_num=1, partial_bn=True, print_spec=True, pretrain='imagenet',
                 is_shift=False, shift_div=8, shift_place='blockres', fc_lr5=False,
                 temporal_pool=False, non_local=False, comu_type='replace', follow_pretrain='',add_se=False):
        super(TSN, self).__init__()
        self.modality = modality
        self.num_segments = num_segments
        self.reshape = True
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.crop_num = crop_num
        self.consensus_type = consensus_type
        self.img_feature_dim = img_feature_dim  # the dimension of the CNN feature to represent each frame
        self.pretrain = pretrain

        self.is_shift = is_shift
        self.shift_div = shift_div
        self.shift_place = shift_place
        self.base_model_name = base_model
        self.fc_lr5 = fc_lr5
        self.temporal_pool = temporal_pool
        self.non_local = non_local
        self.add_se=add_se

        self.comu_type = comu_type
        self.follow_pretrain = follow_pretrain

        if not before_softmax and consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")

        if new_length is None:
            self.new_length = 1 if modality == "RGB" else 5
        else:
            self.new_length = new_length
        if print_spec:
            print(("""
    Initializing TSN with base model: {}.
    TSN Configurations:
        input_modality:     {}
        num_segments:       {}
        new_length:         {}
        consensus_module:   {}
        dropout_ratio:      {}
        img_feature_dim:    {}
            """.format(base_model, self.modality, self.num_segments, self.new_length, consensus_type, self.dropout,
                       self.img_feature_dim)))

        self._prepare_base_model(base_model)

        feature_dim = self._prepare_tsn(num_class)

        if self.modality == 'Flow':
            print("Converting the ImageNet model to a flow init model")
            self.base_model = self._construct_flow_model(self.base_model)
            print("Done. Flow model ready...")
        elif self.modality == 'RGBDiff':
            print("Converting the ImageNet model to RGB+Diff init model")
            self.base_model = self._construct_diff_model(self.base_model)
            print("Done. RGBDiff model ready.")

        self.consensus = ConsensusModule(consensus_type)

        if not self.before_softmax:
            self.softmax = nn.Softmax()

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def _prepare_tsn(self, num_class):
        feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
        if self.dropout == 0:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Linear(feature_dim, num_class))
            self.new_fc = None
        else:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
            self.new_fc = nn.Linear(feature_dim, num_class)

        std = 0.001
        if self.new_fc is None:
            normal_(getattr(self.base_model, self.base_model.last_layer_name).weight, 0, std)
            constant_(getattr(self.base_model, self.base_model.last_layer_name).bias, 0)
        else:
            if hasattr(self.new_fc, 'weight'):
                normal_(self.new_fc.weight, 0, std)
                constant_(self.new_fc.bias, 0)
        return feature_dim

    def _prepare_base_model(self, base_model):
        print('=> base model: {}'.format(base_model))

        if 'resnet' in base_model:
            self.base_model = getattr(torchvision.models, base_model)(True if self.pretrain == 'imagenet' else False)
            # if self.follow_pretrain != '':
            #     load_followed_model(self.base_model, self.follow_pretrain)
            # if self.is_shift:
            #     print('Adding temporal shift...')
            #     from ops.temporal_shift import make_temporal_shift
            #     make_temporal_shift(self.base_model, self.num_segments,
            #                         n_div=self.shift_div, place=self.shift_place, temporal_pool=self.temporal_pool,
            #                         comu_type=self.comu_type)
            # if self.non_local:
            #     print('Adding non-local module...')
            #     from ops.non_local import make_non_local
            #     make_non_local(self.base_model, self.num_segments)
            #
            # if self.add_se:
            #     print('Adding st se...')
            #     from ops.add_se_op import add_se
            #     add_se(self.base_model, self.num_segments)

            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

            self.base_model.avgpool = nn.AdaptiveAvgPool2d(1)

            if self.modality == 'Flow':
                self.input_mean = [0.5]
                self.input_std = [np.mean(self.input_std)]
            elif self.modality == 'RGBDiff':
                self.input_mean = [0.485, 0.456, 0.406] + [0] * 3 * self.new_length
                self.input_std = self.input_std + [np.mean(self.input_std) * 2] * 3 * self.new_length

        elif base_model == 'mobilenetv2':
            from archs.mobilenet_v2 import mobilenet_v2, InvertedResidual
            self.base_model = mobilenet_v2(True if self.pretrain == 'imagenet' else False)

            self.base_model.last_layer_name = 'classifier'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

            self.base_model.avgpool = nn.AdaptiveAvgPool2d(1)
            if self.is_shift:
                from ops.temporal_shift import TemporalShift
                for m in self.base_model.modules():
                    if isinstance(m, InvertedResidual) and len(m.conv) == 8 and m.use_res_connect:
                        if self.print_spec:
                            print('Adding temporal shift... {}'.format(m.use_res_connect))
                        m.conv[0] = TemporalShift(m.conv[0], n_segment=self.num_segments, n_div=self.shift_div)
            if self.modality == 'Flow':
                self.input_mean = [0.5]
                self.input_std = [np.mean(self.input_std)]
            elif self.modality == 'RGBDiff':
                self.input_mean = [0.485, 0.456, 0.406] + [0] * 3 * self.new_length
                self.input_std = self.input_std + [np.mean(self.input_std) * 2] * 3 * self.new_length

        elif base_model == 'BNInception':
            from archs.bn_inception import bninception
            self.base_model = bninception(pretrained=self.pretrain)
            self.input_size = self.base_model.input_size
            self.input_mean = self.base_model.mean
            self.input_std = self.base_model.std
            self.base_model.last_layer_name = 'fc'
            if self.modality == 'Flow':
                self.input_mean = [128]
            elif self.modality == 'RGBDiff':
                self.input_mean = self.input_mean * (1 + self.new_length)
            if self.is_shift:
                print('Adding temporal shift...')
                self.base_model.build_temporal_ops(
                    self.num_segments, is_temporal_shift=self.shift_place, shift_div=self.shift_div)
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(TSN, self).train(mode)
        count = 0
        if self._enable_pbn and mode:
            print("Freezing BatchNorm2D except the first one and new one.")
            # for m in self.base_model.modules():
            for n, m in self.named_modules():
                if 'shift_block' not in n and 'bn' in n and 'bn3' not in n:
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()
                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

                # if isinstance(m, nn.BatchNorm2d):
                #     count += 1
                #     if count >= (2 if self._enable_pbn else 1):
                #         m.eval()
                #         # shutdown update in frozen mode
                #         m.weight.requires_grad = False
                #         m.bias.requires_grad = False

    def partialBN(self, enable):
        self._enable_pbn = enable

    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        lr5_weight = []
        lr10_bias = []
        bn = []
        custom_ops = []

        conv_cnt = 0
        bn_cnt = 0
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv3d):
                ps = list(m.parameters())  # get the parameters
                conv_cnt += 1  # conv_cont +1
                if conv_cnt == 1:  # if is the first
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:  # can justify if there is bias
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                if self.fc_lr5:
                    lr5_weight.append(ps[0])
                else:
                    normal_weight.append(ps[0])
                if len(ps) == 2:
                    if self.fc_lr5:
                        lr10_bias.append(ps[1])
                    else:
                        normal_bias.append(ps[1])

            elif isinstance(m, torch.nn.BatchNorm2d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm1d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm3d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        return [
            {'params': first_conv_weight, 'lr_mult': 5 if self.modality == 'Flow' else 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 10 if self.modality == 'Flow' else 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
            {'params': custom_ops, 'lr_mult': 1, 'decay_mult': 1,
             'name': "custom_ops"},
            # for fc
            {'params': lr5_weight, 'lr_mult': 5, 'decay_mult': 1,
             'name': "lr5_weight"},
            {'params': lr10_bias, 'lr_mult': 10, 'decay_mult': 0,
             'name': "lr10_bias"},
            # for new
        ]

    def get_optim_policies_new(self):

        new_conv_weight = []
        new_conv_bias = []
        new_lstm_weight = []
        new_lstm_bias = []
        new_bn = []

        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        lr5_weight = []
        lr10_bias = []
        bn = []
        custom_ops = []

        bn_cnt = 0
        conv_cnt = 0
        for n, m in self.named_modules():
            if 'shift_block' in n:
                if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv3d):
                    ps = list(m.parameters())  # get the parameters
                    new_conv_weight.append(ps[0])
                    if len(ps) == 2:  # can justify if there is bias
                        new_conv_bias.append(ps[1])

                elif isinstance(m, torch.nn.BatchNorm2d):
                    new_bn.extend(list(m.parameters()))

                elif isinstance(m, torch.nn.BatchNorm1d):
                    new_bn.extend(list(m.parameters()))

                elif isinstance(m, torch.nn.BatchNorm3d):
                    new_bn.extend(list(m.parameters()))

                elif isinstance(m, torch.nn.LSTM):
                    for l_n, l_p in m.named_parameters():
                        if 'weight' in l_n:
                            new_lstm_weight.append(l_p)
                        if 'bias' in l_n:
                            new_lstm_bias.append(l_p)
                elif len(m._modules) == 0:
                    if len(list(m.parameters())) > 0:
                        raise ValueError(
                            "New atomic module type: {}. Need to give it a learning policy".format(type(m)))
            else:
                if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv3d):
                    ps = list(m.parameters())  # get the parameters
                    conv_cnt += 1  # conv_cont +1
                    if conv_cnt == 1:  # if is the first
                        first_conv_weight.append(ps[0])
                        if len(ps) == 2:  # can justify if there is bias
                            first_conv_bias.append(ps[1])
                    else:
                        normal_weight.append(ps[0])
                        if len(ps) == 2:
                            normal_bias.append(ps[1])
                elif isinstance(m, torch.nn.Linear):
                    ps = list(m.parameters())
                    if self.fc_lr5:
                        lr5_weight.append(ps[0])
                    else:
                        normal_weight.append(ps[0])
                    if len(ps) == 2:
                        if self.fc_lr5:
                            lr10_bias.append(ps[1])
                        else:
                            normal_bias.append(ps[1])

                elif isinstance(m, torch.nn.BatchNorm2d):
                    bn_cnt += 1
                    # later BN's are frozen
                    if not self._enable_pbn or bn_cnt == 1:
                        bn.extend(list(m.parameters()))
                elif isinstance(m, torch.nn.BatchNorm1d):
                    bn_cnt += 1
                    # later BN's are frozen
                    if not self._enable_pbn or bn_cnt == 1:
                        bn.extend(list(m.parameters()))
                elif isinstance(m, torch.nn.BatchNorm3d):
                    bn_cnt += 1
                    # later BN's are frozen
                    if not self._enable_pbn or bn_cnt == 1:
                        bn.extend(list(m.parameters()))
                elif len(m._modules) == 0:
                    if len(list(m.parameters())) > 0:
                        raise ValueError(
                            "New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        return [  # for new
            {'params': new_conv_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "new_conv_weight", 'is_backbone': False},
            {'params': new_conv_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "new_conv_bias", 'is_backbone': False},
            {'params': new_lstm_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "new_lstm_weight", 'is_backbone': False},
            {'params': new_lstm_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "new_lstm_bias", 'is_backbone': False},
            {'params': new_bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "new_bn", 'is_backbone': False},
            # for backbone
            {'params': first_conv_weight, 'lr_mult': 5 if self.modality == 'Flow' else 1, 'decay_mult': 1,
             'name': "first_conv_weight", 'is_backbone': True},
            {'params': first_conv_bias, 'lr_mult': 10 if self.modality == 'Flow' else 2, 'decay_mult': 0,
             'name': "first_conv_bias", 'is_backbone': True},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight", 'is_backbone': True},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias", 'is_backbone': True},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift", 'is_backbone': True},
            {'params': custom_ops, 'lr_mult': 1, 'decay_mult': 1,
             'name': "custom_ops", 'is_backbone': True},
            # for fc
            {'params': lr5_weight, 'lr_mult': 5, 'decay_mult': 1,
             'name': "lr5_weight", 'is_backbone': False},
            {'params': lr10_bias, 'lr_mult': 10, 'decay_mult': 0,
             'name': "lr10_bias", 'is_backbone': False}
        ]

    def forward(self, input, no_reshape=False):
        if not no_reshape:
            sample_len = (3 if self.modality == "RGB" else 2) * self.new_length

            if self.modality == 'RGBDiff':
                sample_len = 3 * self.new_length
                input = self._get_diff(input)

            base_out = self.base_model(input.view((-1, sample_len) + input.size()[-2:]))
        else:
            base_out = self.base_model(input)

        if self.dropout > 0:
            base_out = self.new_fc(base_out)

        if not self.before_softmax:
            base_out = self.softmax(base_out)

        if self.reshape:
            if self.is_shift and self.temporal_pool:
                base_out = base_out.view((-1, self.num_segments // 2) + base_out.size()[1:])
            else:
                base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])
            output = self.consensus(base_out)
            return output.squeeze(1)

    def _get_diff(self, input, keep_rgb=False):
        input_c = 3 if self.modality in ["RGB", "RGBDiff"] else 2
        input_view = input.view((-1, self.num_segments, self.new_length + 1, input_c,) + input.size()[2:])
        if keep_rgb:
            new_data = input_view.clone()
        else:
            new_data = input_view[:, :, 1:, :, :, :].clone()

        for x in reversed(list(range(1, self.new_length + 1))):
            if keep_rgb:
                new_data[:, :, x, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]
            else:
                new_data[:, :, x - 1, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]

        return new_data

    def _construct_flow_model(self, base_model):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (2 * self.new_length,) + kernel_size[2:]
        new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()

        new_conv = nn.Conv2d(2 * self.new_length, conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data  # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7]  # remove .weight suffix to get the layer name

        # replace the first convlution layer
        setattr(container, layer_name, new_conv)

        if self.base_model_name == 'BNInception':
            import torch.utils.model_zoo as model_zoo
            sd = model_zoo.load_url('https://www.dropbox.com/s/35ftw2t4mxxgjae/BNInceptionFlow-ef652051.pth.tar?dl=1')
            base_model.load_state_dict(sd)
            print('=> Loading pretrained Flow weight done...')
        else:
            print('#' * 30, 'Warning! No Flow pretrained model is found')
        return base_model

    def _construct_diff_model(self, base_model, keep_rgb=False):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        if not keep_rgb:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
        else:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = torch.cat(
                (params[0].data, params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()),
                1)
            new_kernel_size = kernel_size[:1] + (3 + 3 * self.new_length,) + kernel_size[2:]

        new_conv = nn.Conv2d(new_kernel_size[1], conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data  # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7]  # remove .weight suffix to get the layer name

        # replace the first convolution layer
        setattr(container, layer_name, new_conv)
        return base_model

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    def get_augmentation(self, flip=True):
        if self.modality == 'RGB':
            if flip:
                return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                                                       GroupRandomHorizontalFlip(is_flow=False)])
            else:
                print('#' * 20, 'NO FLIP!!!')
                return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66])])
        elif self.modality == 'Flow':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=True)])
        elif self.modality == 'RGBDiff':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])

    def first_fit(self):
        print('freeze bns in backbone')
        for n, m in self.named_modules():
            if 'shift_block' not in n and 'bn' in n:
                m.eval()
                # shutdown update in frozen mode
                m.weight.requires_grad = False
                m.bias.requires_grad = False
import torch
import torch.nn as nn
from collections import OrderedDict
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule

from mmseg.models.decode_heads.isa_head import ISALayer
from mmseg.ops import resize
from ..builder import HEADS
from .aspp_head import ASPPModule
from .decode_head import BaseDecodeHead
from functools import partial
from .segformer_head import MLP
from .sep_aspp_head import DepthwiseSeparableASPPModule
from .conv_module import stacked_conv


class SinglePanopticDeepLabHead(nn.Module):
    def __init__(self, decoder_channels, head_channels, num_classes, class_key):
        super(SinglePanopticDeepLabHead, self).__init__()
        fuse_conv = partial(stacked_conv, kernel_size=5, num_stack=1, padding=2,
                            conv_type='depthwise_separable_conv')

        self.num_head = len(num_classes)
        assert self.num_head == len(class_key)

        classifier = {}
        for i in range(self.num_head):
            classifier[class_key[i]] = nn.Sequential(
                fuse_conv(
                    decoder_channels,
                    head_channels,
                ),
                nn.Conv2d(head_channels, num_classes[i], 1)
            )
        self.classifier = nn.ModuleDict(classifier)
        self.class_key = class_key

    def forward(self, x):
        # pred = OrderedDict()
        # build classifier
        # for key in self.class_key:
        center = self.classifier['center'](x)
        offset = self.classifier['offset'](x)

        return center, offset


class ASPPWrapper(nn.Module):

    def __init__(self,
                 in_channels,
                 channels,
                 sep,
                 dilations,
                 pool,
                 norm_cfg,
                 act_cfg,
                 align_corners,
                 context_cfg=None):
        super(ASPPWrapper, self).__init__()
        assert isinstance(dilations, (list, tuple))
        self.dilations = dilations
        self.align_corners = align_corners
        if pool:
            self.image_pool = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                ConvModule(
                    in_channels,
                    channels,
                    1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
        else:
            self.image_pool = None
        if context_cfg is not None:
            self.context_layer = build_layer(in_channels, channels,
                                             **context_cfg)
        else:
            self.context_layer = None
        ASPP = {True: DepthwiseSeparableASPPModule, False: ASPPModule}[sep]
        self.aspp_modules = ASPP(
            dilations=dilations,
            in_channels=in_channels,
            channels=channels,
            norm_cfg=norm_cfg,
            conv_cfg=None,
            act_cfg=act_cfg)
        self.bottleneck = ConvModule(
            (len(dilations) + int(pool) + int(bool(context_cfg))) * channels,
            channels,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x):
        """Forward function."""
        aspp_outs = []
        if self.image_pool is not None:
            aspp_outs.append(
                resize(
                    self.image_pool(x),
                    size=x.size()[2:],
                    mode='bilinear',
                    align_corners=self.align_corners))
        if self.context_layer is not None:
            aspp_outs.append(self.context_layer(x))
        aspp_outs.extend(self.aspp_modules(x))
        aspp_outs = torch.cat(aspp_outs, dim=1)

        output = self.bottleneck(aspp_outs)
        return output


def build_layer(in_channels, out_channels, type, **kwargs):
    if type == 'id':
        return nn.Identity()
    elif type == 'mlp':
        return MLP(input_dim=in_channels, embed_dim=out_channels)
    elif type == 'sep_conv':
        return DepthwiseSeparableConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            padding=kwargs['kernel_size'] // 2,
            **kwargs)
    elif type == 'conv':
        return ConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            padding=kwargs['kernel_size'] // 2,
            **kwargs)
    elif type == 'aspp':
        return ASPPWrapper(
            in_channels=in_channels, channels=out_channels, **kwargs)
    elif type == 'rawconv_and_aspp':
        kernel_size = kwargs.pop('kernel_size')
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2),
            ASPPWrapper(
                in_channels=out_channels, channels=out_channels, **kwargs))
    elif type == 'isa':
        return ISALayer(
            in_channels=in_channels, channels=out_channels, **kwargs)
    else:
        raise NotImplementedError(type)


@HEADS.register_module()
class DAFormerHead(BaseDecodeHead):

    def __init__(self, **kwargs):
        super(DAFormerHead, self).__init__(
            input_transform='multiple_select', **kwargs)

        assert not self.align_corners
        decoder_params = kwargs['decoder_params']
        embed_dims = decoder_params['embed_dims']
        if isinstance(embed_dims, int):
            embed_dims = [embed_dims] * len(self.in_index)
        embed_cfg = decoder_params['embed_cfg']
        embed_neck_cfg = decoder_params['embed_neck_cfg']
        if embed_neck_cfg == 'same_as_embed_cfg':
            embed_neck_cfg = embed_cfg
        fusion_cfg = decoder_params['fusion_cfg']
        for cfg in [embed_cfg, embed_neck_cfg, fusion_cfg]:
            if cfg is not None and 'aspp' in cfg['type']:
                cfg['align_corners'] = self.align_corners

        self.embed_layers_semantic = {}
        for i, in_channels, embed_dim in zip(self.in_index, self.in_channels,
                                             embed_dims):
            if i == self.in_index[-1]:
                self.embed_layers_semantic[str(i)] = build_layer(
                    in_channels, embed_dim, **embed_neck_cfg)
            else:
                self.embed_layers_semantic[str(i)] = build_layer(
                    in_channels, embed_dim, **embed_cfg)
        self.embed_layers_semantic = nn.ModuleDict(self.embed_layers_semantic)

        self.fuse_layer_semantic = build_layer(
            sum(embed_dims), self.channels, **fusion_cfg)

        self.embed_layers_instance = {}
        for i, in_channels, embed_dim in zip(self.in_index, self.in_channels,
                                             embed_dims):
            if i == self.in_index[-1]:
                self.embed_layers_instance[str(i)] = build_layer(
                    in_channels, embed_dim, **embed_neck_cfg)
            else:
                self.embed_layers_instance[str(i)] = build_layer(
                    in_channels, embed_dim, **embed_cfg)
        self.embed_layers_instance = nn.ModuleDict(self.embed_layers_instance)

        self.fuse_layer_instance = build_layer(
            sum(embed_dims), self.channels, **fusion_cfg)
        
        instance_head_kwargs = dict(
                decoder_channels=256,
                head_channels=32,
                num_classes=(1, 2),
                class_key=["center", "offset"]
            )
        self.instance_head = SinglePanopticDeepLabHead(**instance_head_kwargs)

    def forward(self, inputs):
        pred = OrderedDict()
        # Semantic decoder
        semantics = inputs
        n, _, h, w = semantics[-1].shape
        # for f in x:
        #     mmcv.print_log(f'{f.shape}', 'mmseg')

        os_size = semantics[0].size()[2:]
        _c = {}
        for i in self.in_index:
            # mmcv.print_log(f'{i}: {x[i].shape}', 'mmseg')
            _c[i] = self.embed_layers_semantic[str(i)](semantics[i])
            if _c[i].dim() == 3:
                _c[i] = _c[i].permute(0, 2, 1).contiguous()\
                    .reshape(n, -1, semantics[i].shape[2], semantics[i].shape[3])
            # mmcv.print_log(f'_c{i}: {_c[i].shape}', 'mmseg')
            if _c[i].size()[2:] != os_size:
                # mmcv.print_log(f'resize {i}', 'mmseg')
                _c[i] = resize(
                    _c[i],
                    size=os_size,
                    mode='bilinear',
                    align_corners=self.align_corners)

        semantics = self.fuse_layer_semantic(torch.cat(list(_c.values()), dim=1))
        semantics = self.cls_seg(semantics)
    
        pred['semantic'] = semantics

        # Instance decoder
        instance = inputs
        n, _, h, w = instance[-1].shape

        os_size = instance[0].size()[2:]
        _c = {}
        for i in self.in_index:
            # mmcv.print_log(f'{i}: {x[i].shape}', 'mmseg')
            _c[i] = self.embed_layers_instance[str(i)](instance[i])
            if _c[i].dim() == 3:
                _c[i] = _c[i].permute(0, 2, 1).contiguous()\
                    .reshape(n, -1, instance[i].shape[2], instance[i].shape[3])
            # mmcv.print_log(f'_c{i}: {_c[i].shape}', 'mmseg')
            if _c[i].size()[2:] != os_size:
                # mmcv.print_log(f'resize {i}', 'mmseg')
                _c[i] = resize(
                    _c[i],
                    size=os_size,
                    mode='bilinear',
                    align_corners=self.align_corners)

        instance = self.fuse_layer_instance(torch.cat(list(_c.values()), dim=1))
        # instance = self.conv_inst(instance)
        # pred['center'] = self.conv_inst_center(instance)
        # pred['offset'] = self.conv_inst_offset(instance)

        center, offset = self.instance_head(instance)
        pred['center'] = center
        pred['offset'] = offset
        # # for key in instance.keys():
        # #     pred[key] = instance[key]
        return pred

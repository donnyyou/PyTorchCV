#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Priorbox layer for Detection.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from torch.nn import functional as F
from torch import nn

from utils.layers.det.fr_priorbox_layer import FRPriorBoxLayer
from utils.layers.det.fr_roi_generator import FRRoiGenerator


class FRRPNLayer(nn.Module):
    """Region Proposal Network introduced in Faster R-CNN.
    This is Region Proposal Network introduced in Faster R-CNN [#]_.
    This takes features extracted from images and propose
    class agnostic bounding boxes around "objects".
    .. [#] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. \
    Faster R-CNN: Towards Real-Time Object Detection with \
    Region Proposal Networks. NIPS 2015.
    Args:
        in_channels (int): The channel size of input.
        mid_channels (int): The channel size of the intermediate tensor.
        ratios (list of floats): This is ratios of width to height of
            the anchors.
        anchor_scales (list of numbers): This is areas of anchors.
            Those areas will be the product of the square of an element in
            :obj:`anchor_scales` and the original area of the reference
            window.
        feat_stride (int): Stride size after extracting features from an
            image.
        initialW (callable): Initial weight value. If :obj:`None` then this
            function uses Gaussian distribution scaled by 0.1 to
            initialize weight.
            May also be a callable that takes an array and edits its values.
        proposal_creator_params (dict): Key valued paramters for
            :class:`model.utils.creator_tools.ProposalCreator`.
    .. seealso::
        :class:`~model.utils.creator_tools.ProposalCreator`
    """

    def __init__(self, configer):
        super(FRRPNLayer, self).__init__()
        self.configer = configer
        self.anchor_bboxes = FRPriorBoxLayer(configer)()
        self.n_anchors = self.configer.get('gt', 'n_anchors_list')
        self.conv1 = nn.Conv2d(512, 512, 3, 1, 1)
        self.score = nn.Conv2d(512, self.n_anchors[0] * 2, 1, 1, 0)
        self.loc = nn.Conv2d(512, self.n_anchors[0] * 4, 1, 1, 0)
        self.fr_roi_genrator = FRRoiGenerator(configer)

    def forward(self, feature_list, img_size, scale=1.):
        """Forward Region Proposal Network.
        Here are notations.
        * :math:`N` is batch size.
        * :math:`C` channel size of the input.
        * :math:`H` and :math:`W` are height and witdh of the input feature.
        * :math:`A` is number of anchors assigned to each pixel.
        Args:
            x (~torch.autograd.Variable): The Features extracted from images.
                Its shape is :math:`(N, C, H, W)`.
            img_size (tuple of ints): A tuple :obj:`height, width`,
                which contains image size after scaling.
            scale (float): The amount of scaling done to the input images after
                reading them from files.
        Returns:
            (~torch.autograd.Variable, ~torch.autograd.Variable, array, array, array):
            This is a tuple of five following values.
            * **rpn_locs**: Predicted bounding box offsets and scales for \
                anchors. Its shape is :math:`(N, H W A, 4)`.
            * **rpn_scores**:  Predicted foreground scores for \
                anchors. Its shape is :math:`(N, H W A, 2)`.
            * **rois**: A bounding box array containing coordinates of \
                proposal boxes.  This is a concatenation of bounding box \
                arrays from multiple images in the batch. \
                Its shape is :math:`(R', 4)`. Given :math:`R_i` predicted \
                bounding boxes from the :math:`i` th image, \
                :math:`R' = \\sum _{i=1} ^ N R_i`.
            * **roi_indices**: An array containing indices of images to \
                which RoIs correspond to. Its shape is :math:`(R',)`.
            * **anchor**: Coordinates of enumerated shifted anchors. \
                Its shape is :math:`(H W A, 4)`.
        """
        assert len(feature_list) == 1
        x = feature_list[0]
        batch_size, _, hh, ww = x.size()
        h = F.relu(self.conv1(x))

        rpn_locs = self.loc(h)
        # UNNOTE: check whether need contiguous
        # A: Yes
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4)
        rpn_scores = self.score(h)
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous()
        rpn_fg_scores = rpn_scores.view(batch_size, hh, ww, self.n_anchors[0], 2)[:, :, :, :, 1].contiguous()
        rpn_fg_scores = rpn_fg_scores.view(batch_size, -1)
        rpn_scores = rpn_scores.view(batch_size, -1, 2)

        rois = list()
        roi_indices = list()
        for i in range(batch_size):
            roi = self.proposal_layer(rpn_locs[i].cpu().numpy(),
                                      rpn_fg_scores[i].cpu().numpy(), self.anchor_bboxes, img_size)
            batch_index = i * np.ones((len(roi),), dtype=np.int32)
            rois.append(roi)
            roi_indices.append(batch_index)

        rois = np.concatenate(rois, axis=0)
        roi_indices = np.concatenate(roi_indices, axis=0)
        return rpn_locs, rpn_scores, rois, roi_indices


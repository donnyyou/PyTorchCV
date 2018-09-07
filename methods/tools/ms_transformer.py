
import random
import torch
import torch.nn.functional as F


class MSTransformer(object):
    """Resize the given numpy.ndarray to random size and aspect ratio.

    Args:
        scale_min: the min scale to resize.
        scale_max: the max scale to resize.
    """

    def __init__(self, configer):
        self.configer = configer

    def __call__(self, batch_img_list, batch_labelmap_list=None, batch_maskmap_list=None,
                 batch_kpts_list=None, batch_bboxes_list=None, batch_labels_list=None,
                 batch_polygons_list=None):

        ms_input_size = self.configer.get('trans_params', 'ms_transformer')['ms_input_size']
        target_width, target_height = ms_input_size[random.randint(0, len(ms_input_size))]

        for i in range(len(batch_img_list)):
            channels, height, width = batch_img_list[i].size()
            scaled_size = [width, height]

            if self.configer.get('trans_params', 'ms_transformer')['method'] in ['only_scale', 'scale_and_pad']:
                w_scale_ratio = target_width / width
                h_scale_ratio = target_height / height
                if self.configer.get('trans_params', 'ms_transformer')['method'] == 'scale_and_pad':
                    w_scale_ratio = min(w_scale_ratio, h_scale_ratio)
                    h_scale_ratio = w_scale_ratio

                if batch_kpts_list is not None and batch_kpts_list.numel() > 0:
                    batch_kpts_list[i][:, :, 0] *= w_scale_ratio
                    batch_kpts_list[i][:, :, 1] *= h_scale_ratio

                if batch_bboxes_list is not None and batch_bboxes_list.numel() > 0:
                    batch_bboxes_list[i][:, 0::2] *= w_scale_ratio
                    batch_bboxes_list[i][:, 1::2] *= h_scale_ratio

                if batch_polygons_list is not None:
                    for object_id in range(len(batch_polygons_list)):
                        for polygon_id in range(len(batch_polygons_list[object_id])):
                            batch_polygons_list[i][object_id][polygon_id][0::2] *= w_scale_ratio
                            batch_polygons_list[i][object_id][polygon_id][1::2] *= h_scale_ratio

                scaled_size = (int(round(width * w_scale_ratio)), int(round(height * h_scale_ratio)))
                batch_img_list[i] = F.interpolate(batch_img_list[i].unsqueeze(0), scaled_size, mode='bilinear')
                if batch_labelmap_list is not None:
                    batch_labelmap_list[i] = F.interpolate(batch_labelmap_list[i].unsqueeze(0).unsqueeze(0).float(),
                                                           scaled_size, mode='nearest').long().squeeze(0)

                if batch_maskmap_list is not None:
                    batch_maskmap_list[i] = F.interpolate(batch_maskmap_list[i].unsqueeze(0).unsqueeze(0).float(),
                                                          scaled_size, mode='nearest').long().squeeze(0)

            pad_width = target_width - scaled_size[0]
            pad_height = target_height - scaled_size[1]
            assert pad_height >= 0 and pad_width >= 0
            if pad_width > 0 or pad_height > 0:
                assert self.configer.get('trans_params', 'ms_transformer')['method'] in ['only_pad', 'scale_and_pad']
                left_pad = random.randint(0, pad_width)  # pad_left
                up_pad = random.randint(0, pad_height)  # pad_up

                expand_image = torch.zeros((1, channels, target_height, target_width))
                expand_image[:, :, up_pad:up_pad + height, left_pad:left_pad + width] = batch_img_list[i]
                batch_img_list[i] = expand_image

                if batch_labelmap_list is not None:
                    expand_labelmap = torch.zeros((1, target_height, target_width)).long()
                    expand_labelmap[:, :, :] = self.configer.get('data', 'num_classes')
                    expand_labelmap[:, up_pad:up_pad + height, left_pad:left_pad + width] =batch_labelmap_list[i]
                    batch_labelmap_list[i] = expand_labelmap

                if batch_maskmap_list is not None:
                    expand_maskmap = torch.zeros((1, target_height, target_width)).long()
                    expand_maskmap[:, :, :] = self.configer.get('data', 'num_classes')
                    expand_maskmap[:, up_pad:up_pad + height, left_pad:left_pad + width] =batch_maskmap_list[i]
                    batch_maskmap_list[i] = expand_maskmap

                if batch_polygons_list is not None:
                    for object_id in range(len(batch_polygons_list)):
                        for polygon_id in range(len(batch_polygons_list[object_id])):
                            batch_polygons_list[i][object_id][polygon_id][0::2] += left_pad
                            batch_polygons_list[i][object_id][polygon_id][1::2] += up_pad

                if batch_kpts_list is not None and batch_kpts_list.numel() > 0:
                    batch_kpts_list[i][:, :, 0] += left_pad
                    batch_kpts_list[i][:, :, 1] += up_pad

                if batch_bboxes_list is not None and batch_bboxes_list.numel() > 0:
                    batch_bboxes_list[i][:, 0::2] += left_pad
                    batch_bboxes_list[i][:, 1::2] += up_pad

            return {'img': torch.cat(batch_img_list, 0),
                    'labelmap': torch.cat(batch_labelmap_list, 0),
                    'maskmap': torch.cat(batch_maskmap_list, 0),
                    'kpts': batch_kpts_list,
                    'bboxes': batch_bboxes_list,
                    'labels': batch_labels_list,
                    'polygons': batch_polygons_list}
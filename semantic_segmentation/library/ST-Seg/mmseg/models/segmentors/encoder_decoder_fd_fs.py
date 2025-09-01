# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Support for seg_weight and forward_with_aux

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

from mmseg.utils.utils import downscale_label_ratio
from mmseg.utils.utils import AdaptiveInstanceNormalization
from mmseg.core import add_prefix
from mmseg.ops import resize
from .. import builder
from ..builder import build_loss
from ..builder import SEGMENTORS
from .base import BaseSegmentor, get_module

@SEGMENTORS.register_module()
class EncoderDecoder_FD_FS(BaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 mat_model = None,
                 mat_feature_dist_lambda=None,
                 mat_feature_dist_classes=None,
                 mat_feature_dist_scale_min_ratio=None,
                 mat_feature_layer = None,
                 wild_model = None,
                 wild_feature_layer = None,
                 kl_loss = None,
                 ):
        super(EncoderDecoder_FD_FS, self).__init__(init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        
        assert self.with_decode_head
        ############# FD ###############
        self.mat = mat_model['backbone']
        self.fdist_lambda = mat_feature_dist_lambda
        self.fdist_classes = mat_feature_dist_classes
        self.fdist_scale_min_ratio = mat_feature_dist_scale_min_ratio
        self.enable_fdist = self.fdist_lambda > 0
        self.mat_layer = mat_feature_layer
        
        if self.enable_fdist:
            self.mat_model = builder.build_backbone(self.mat)
        else:
            self.mat_model = None
        ################################
        ############# WilD #############
        #self.wild = wild_model['backbone']
        self.wild_layer = wild_feature_layer   # wildnet statistics stage number
        self.enable_wild = isinstance(self.wild_layer, list) 
        # if self.enable_wild:
        #     #self.wild_model = builder.build_backbone(self.wild)
        # else:
        #     self.wild_model = None
        
        self.kl_loss = kl_loss
        self.criterion_kl = nn.KLDivLoss(reduction='batchmean').cuda()
        
        ###################################
    def get_mat_model(self):
        return get_module(self.mat_model)
    def get_wild_model(self):
        return get_module(self.wild_model)
    
    def masked_feat_dist(self, f1, f2, mask=None):
        feat_diff = f1 - f2
        # mmcv.print_log(f'fdiff: {feat_diff.shape}', 'mmseg')
        pw_feat_dist = torch.norm(feat_diff, dim=1, p=2)
        # mmcv.print_log(f'pw_fdist: {pw_feat_dist.shape}', 'mmseg')
        if mask is not None:
            # mmcv.print_log(f'fd mask: {mask.shape}', 'mmseg')
            pw_feat_dist = pw_feat_dist[mask.squeeze(1)]
            # mmcv.print_log(f'fd masked: {pw_feat_dist.shape}', 'mmseg')
        return torch.mean(pw_feat_dist)
    
    def calc_feat_dist(self, img, gt, feat=None):
        assert self.enable_fdist
        with torch.no_grad():
            self.get_mat_model().eval()
            feat_mat = self.mat_model(img)
            feat_mat = [f.detach() for f in feat_mat]
        lay = -1
        if self.fdist_classes is not None:
            # fdclasses = torch.tensor(self.fdist_classes, device=gt.device)
            # scale_factor = gt.shape[-1] // feat[lay].shape[-1]
            # #print(scale_factor)
            # gt_rescaled = downscale_label_ratio(gt, scale_factor,
            #                                     self.fdist_scale_min_ratio,
            #                                     self.num_classes,
            #                                     255).long().detach()
            # fdist_mask = torch.any(gt_rescaled[..., None] == fdclasses, -1)
            # feat_dist = self.masked_feat_dist(feat[lay], feat_mat[lay],
            #                                   fdist_mask)

            # self.debug_fdist_mask = fdist_mask
            # self.debug_gt_rescale = gt_rescaled
            ## layer added
            feat_dists = []
            for i in self.mat_layer:
                fdclasses = torch.tensor(self.fdist_classes, device=gt.device)
                scale_factor = gt.shape[-1] // feat[i].shape[-1]
                #print(scale_factor)
                gt_rescaled = downscale_label_ratio(gt, scale_factor,
                                                    self.fdist_scale_min_ratio,
                                                    self.num_classes,
                                                    255).long().detach()
                fdist_mask = torch.any(gt_rescaled[..., None] == fdclasses, -1)
                ##print(fdist_mask.shape)
                feat_dist = self.masked_feat_dist(feat[i], feat_mat[i],
                                                fdist_mask)

                self.debug_fdist_mask = fdist_mask
                self.debug_gt_rescale = gt_rescaled  
                      
                if i == 0:
                    feat_dist = 10 * feat_dist
                if i == 1:
                    feat_dist = 5 * feat_dist
                if i == 2:
                    feat_dist = 2 * feat_dist
                feat_dists.append(feat_dist)
            
        else:
            feat_dist = self.masked_feat_dist(feat[lay], feat_mat[lay])
        #############
        #feat_dist = feat_dist + feat_dist_2
        feat_dist = sum(feat_dists)
        ############3
        feat_dist = self.fdist_lambda * feat_dist / len(self.mat_layer)
        feat_loss, feat_log = self._parse_losses(
            {'loss_imnet_feat_dist': feat_dist})
        feat_log.pop('loss', None)
        return feat_loss, feat_log

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    def _init_auxiliary_head(self, auxiliary_head):
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(builder.build_head(head_cfg))
            else:
                self.auxiliary_head = builder.build_head(auxiliary_head)

    def extract_feat(self, img):
        """Extract features from images."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def extract_wild_feat(self, img):
        """Extract features from images."""
        x = self.wild_model(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def generate_pseudo_label(self, img, img_metas):
        return self.encode_decode(img, img_metas)

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(img)
        out = self._decode_head_forward_test(x, img_metas)
        # print(img_metas)
        # print(out.shape)
        # print(out)#####################################################################################
        # fefe
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out

    def forward_with_aux(self, img, img_metas):
        ret = {}

        x = self.extract_feat(img)
        out = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        ret['main'] = out

        if self.with_auxiliary_head:
            assert not isinstance(self.auxiliary_head, nn.ModuleList)
            out_aux = self.auxiliary_head.forward_test(x, img_metas,
                                                       self.test_cfg)
            out_aux = resize(
                input=out_aux,
                size=img.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            ret['aux'] = out_aux

        return ret

    def _decode_head_forward_train(self,
                                   x,
                                   img_metas,
                                   gt_semantic_seg,
                                   seg_weight=None):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode, seg_logits = self.decode_head.forward_train(x, img_metas,
                                                     gt_semantic_seg,
                                                     self.train_cfg,
                                                     seg_weight)
        #print(loss_decode)
        
        losses.update(add_prefix(loss_decode, 'decode'))
        #print(losses)
        return losses, seg_logits

    def _decode_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        return seg_logits

    def _auxiliary_head_forward_train(self,
                                      x,
                                      img_metas,
                                      gt_semantic_seg,
                                      seg_weight=None):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            #print(self.auxiliary_head)
            for idx, aux_head in enumerate(self.auxiliary_head):
                #print('-------------------\n',idx)
                loss_aux, _ = aux_head.forward_train(x, img_metas,
                                                  gt_semantic_seg,
                                                  self.train_cfg, seg_weight)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux, _ = self.auxiliary_head.forward_train(
                x, img_metas, gt_semantic_seg, self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def forward_dummy(self, img):
        """Dummy forward function."""
        seg_logit = self.encode_decode(img, None)

        return seg_logit

    def forward_train(self,
                      img,
                      img_metas,
                      gt_semantic_seg,
                      target_img,
                      seg_weight=None,
                      return_feat=False):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        ##############WILD IMG TRANSFORM ##############3
        
        
        
        x = self.extract_feat(img)
        losses = dict()
        if return_feat:
            losses['features'] = x
        if self.enable_wild:
            with torch.no_grad():
                get_module(self.backbone).eval()
                #target_img = torch.FloatTensor([target_img])
                target_img = target_img.type(torch.cuda.FloatTensor)
                
                target_img = resize(
                    input=target_img,
                    size = img.shape[2:],
                )

                feat_wilds = self.backbone(target_img)
                feat_wild_ft = [f.detach() for f in feat_wilds]
                feat_wild = feat_wilds[:2]    # B 32 180 180
                del target_img, feat_wild_ft, feat_wilds
                torch.cuda.empty_cache
            get_module(self.backbone).train()
            zws = self.backbone(img, feat_wild)
            
            
        #### Wild styled Source Model ###################################
            styled_losses = dict()
            styled_loss, styled_logits = self.decode_head.forward_train(zws, img_metas,
                                                    gt_semantic_seg,
                                                    self.train_cfg,
                                                    seg_weight)

            #styled_loss['loss_seg'] = styled_loss['loss_seg'] * 0.3
            styled_losses.update(add_prefix(styled_loss, 'styled_decode'))
            losses.update(styled_losses)                                
        #################################################################
        ########### Source Domain Model #################################
        loss_decode, source_logits = self._decode_head_forward_train(x, img_metas,
                                                      gt_semantic_seg,
                                                      seg_weight)
        losses.update(loss_decode)

        loss = dict()
        if self.kl_loss:
            kl_loss = torch.clamp((self.criterion_kl(F.log_softmax(styled_logits, dim=1), 
                                                     F.softmax(source_logits, dim=1)))/
                                  torch.prod(torch.tensor(source_logits.shape[1:])), min=0)
            
            loss['kl_loss'] = kl_loss
            losses.update(loss)

        if self.enable_fdist:
            feat_loss, feat_log = self.calc_feat_dist(img, gt_semantic_seg, x)
            #print(feat_loss) # tensor(xxx, device='cuda:0', grad_fn=<AddBackward0>)
            loss['fd_loss'] = feat_loss

            losses.update(loss)
            
        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x, img_metas, gt_semantic_seg, seg_weight)
            losses.update(loss_aux)

        
        return losses

    # TODO refactor
    def slide_inference(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batched_slide = self.test_cfg.get('batched_slide', False)
        batch_size, _, h_img, w_img = img.size()
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        if batched_slide:
            crop_imgs, crops = [], []
            for h_idx in range(h_grids):
                for w_idx in range(w_grids):
                    y1 = h_idx * h_stride
                    x1 = w_idx * w_stride
                    y2 = min(y1 + h_crop, h_img)
                    x2 = min(x1 + w_crop, w_img)
                    y1 = max(y2 - h_crop, 0)
                    x1 = max(x2 - w_crop, 0)
                    crop_img = img[:, :, y1:y2, x1:x2]
                    crop_imgs.append(crop_img)
                    crops.append((y1, y2, x1, x2))
            crop_imgs = torch.cat(crop_imgs, dim=0)
            crop_seg_logits = self.encode_decode(crop_imgs, img_meta)
            for i in range(len(crops)):
                y1, y2, x1, x2 = crops[i]
                crop_seg_logit = \
                    crop_seg_logits[i * batch_size:(i + 1) * batch_size]
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        else:
            for h_idx in range(h_grids):
                for w_idx in range(w_grids):
                    y1 = h_idx * h_stride
                    x1 = w_idx * w_stride
                    y2 = min(y1 + h_crop, h_img)
                    x2 = min(x1 + w_crop, w_img)
                    y1 = max(y2 - h_crop, 0)
                    x1 = max(x2 - w_crop, 0)
                    crop_img = img[:, :, y1:y2, x1:x2]
                    crop_seg_logit = self.encode_decode(crop_img, img_meta)
                    preds += F.pad(crop_seg_logit,
                                   (int(x1), int(preds.shape[3] - x2), int(y1),
                                    int(preds.shape[2] - y2)))

                    count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=img.device)
        preds = preds / count_mat
        if rescale:
            preds = resize(
                preds,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        return preds

    def whole_inference(self, img, img_meta, rescale):
        """Inference with full image."""

        seg_logit = self.encode_decode(img, img_meta)

        if rescale:
            # support dynamic shape for onnx
            if torch.onnx.is_in_onnx_export():
                size = img.shape[2:]
            else:
                size = img_meta[0]['ori_shape'][:2]##########################################################
            seg_logit = resize(
                seg_logit,
                size=size,
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)

        return seg_logit

    def inference(self, img, img_meta, rescale):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """

        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(img, img_meta, rescale)
        else:
            seg_logit = self.whole_inference(img, img_meta, rescale)
        if hasattr(self.decode_head, 'debug_output_attention') and \
                self.decode_head.debug_output_attention:
            output = seg_logit
        else:
            output = F.softmax(seg_logit, dim=1) ############################################################
            # print("img_name: ",img_meta[0])
            # print("output:",output)
            #####################################################
            # output = output.cpu().numpy() 
            # file_name = img_meta[0]['ori_filename']
            # npy_save = np.asarray(output)
            # file = file_name.split("/")[-1]
            # np.save(os.path.join('./confidence', file), npy_save)
            ######################################################
            
        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3, ))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2, ))

        return output

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        seg_logit = self.inference(img, img_meta, rescale)
        if hasattr(self.decode_head, 'debug_output_attention') and \
                self.decode_head.debug_output_attention:
            seg_pred = seg_logit[:, 0]
        else:
            seg_pred = seg_logit.argmax(dim=1)
            #seg_pred = torch.ones([1,720,1280]) ###############
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

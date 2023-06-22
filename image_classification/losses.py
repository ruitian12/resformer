# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Implements the knowledge distillation loss
"""
import torch
import torch.nn as nn
from torch.nn import functional as F


class DistillationLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """
    def __init__(self, base_criterion: torch.nn.Module, distillation_type: str, alpha: float, tau: float):
        super().__init__()
        self.base_criterion = base_criterion
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau


    def forward(self, outputs, labels, outputs_kd=None, teacher_outputs = None):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """

        base_loss = self.base_criterion(outputs, labels)
        if self.distillation_type == 'none' or teacher_outputs is None:
            return base_loss
        
        if self.distillation_type == 'soft':
            T = self.tau
            # taken from https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
            # with slight modifications
            distillation_loss = F.kl_div(
                F.log_softmax(outputs_kd / T, dim=1),
                #We provide the teacher's targets in log probability because we use log_target=True 
                #(as recommended in pytorch https://github.com/pytorch/pytorch/blob/9324181d0ac7b4f7949a574dbc3e8be30abe7041/torch/nn/functional.py#L2719)
                #but it is possible to give just the probabilities and set log_target=False. In our experiments we tried both.
                F.log_softmax(teacher_outputs / T, dim=1),
                reduction='sum',
                log_target=False #?
            ) * (T * T) / outputs_kd.numel()
            #We divide by outputs_kd.numel() to have the legacy PyTorch behavior. 
            #But we also experiments output_kd.size(0) 
            #see issue 61(https://github.com/facebookresearch/deit/issues/61) for more details
            loss = base_loss  + distillation_loss * self.alpha
        
        elif self.distillation_type == 'cosine':
            distillation_loss = 1 - F.cosine_similarity(outputs_kd, teacher_outputs, eps=1e-6, dim = -1).mean()
            loss = base_loss  + distillation_loss * self.alpha

        elif self.distillation_type == 'smooth-l1':
            teacher_outputs = F.layer_norm(teacher_outputs, tuple((teacher_outputs.shape[-1],)), eps = 1e-6) # feature whitening 
            distillation_loss = F.smooth_l1_loss(outputs_kd, teacher_outputs)
            loss = base_loss  + distillation_loss * self.alpha
                
        elif self.distillation_type == 'l2':
            distillation_loss = F.mse_loss(outputs_kd, teacher_outputs)
            loss = base_loss  + distillation_loss * self.alpha

        return loss

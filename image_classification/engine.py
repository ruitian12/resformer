# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
from gc import enable
import math
import sys
from typing import Iterable, Optional

import torch
import torch.nn.functional as F

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils
from torchvision import transforms
from timm.data.mixup import one_hot


def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, 
                    args = None, set_training_mode = True, writer = None):

    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    
    num_iteration = epoch * len(data_loader)
    multi_res_step, multi_res_mode = args.multi_res_step, args.multi_res_mode
    res_list = args.input_size
    use_checkpoint = args.use_checkpoint #FIXME
    
            
    
    for data_iter_step, (samples, targets) in  enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        teacher_outputs = None
        loss = 0
        
        if (data_iter_step) % args.accum_iter == 0:
            utils.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        if multi_res_mode == 'iter':
            idx_start = num_iteration % (len(res_list) - multi_res_step + 1)
            idx_end = num_iteration % (len(res_list) - multi_res_step + 1) + multi_res_step
        else:
            idx_start = 0
            idx_end = len(res_list)
        
                
        for i in range(len(samples)):
            samples[i] = samples[i].to(device, non_blocking=True)
       
        targets = targets.to(device, non_blocking=True)
            
            
        if not args.sep_mix and mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
    
        
        for idx in range(idx_start, idx_end):
            inputs = samples[idx]
            labels = targets
            n = idx_end - idx_start
                
            if args.sep_mix and mixup_fn is not None:
                inputs, labels = mixup_fn(inputs, labels)
            
            with torch.cuda.amp.autocast(enabled = args.amp):
                if args.distillation_target != None:
                    outputs, outputs_kd  = model(inputs, args.distillation_target)
                    loss_curr = criterion(outputs, labels, outputs_kd, teacher_outputs)
                    loss += loss_curr
                    teacher_outputs = outputs_kd.detach()
                    
                else:
                    outputs, _ = model(inputs)
                    loss_curr = criterion(outputs, labels) 
                    loss += loss_curr
                
            if use_checkpoint:
                loss_scaler(loss_curr / (n * args.accum_iter), optimizer, clip_grad=max_norm,
                parameters=model.parameters(), create_graph=False,
                update_grad= (idx == idx_end - 1) and ((data_iter_step + 1) % args.accum_iter == 0))
                
            loss_value = loss_curr.item()
            metric_logger.updatev2(f'loss-{res_list[idx]}', loss_value)

        loss /= n
        loss_value = loss.item()
        loss /= args.accum_iter
        
        if not use_checkpoint:
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % args.accum_iter == 0)
        
        if (data_iter_step + 1) % args.accum_iter == 0:
            optimizer.zero_grad()
        
        torch.cuda.synchronize()
        
        
        if writer is not None and utils.get_rank() == 0 and (data_iter_step + 1) % args.accum_iter == 0:
            writer.add_scalar('Train / Loss', loss_value, num_iteration)
        num_iteration += 1

        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, res_list, amp = True):
    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    # switch to evaluation mode
    model.eval()
    
    for idx, res in enumerate(res_list):
        for images, target in metric_logger.log_every(data_loader, 10, header):
            inputs = images[idx]
            inputs = inputs.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            
            # compute output
            with torch.cuda.amp.autocast(enabled = amp):
                output = model(inputs, idx)
                loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            batch_size = inputs.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
            metric_logger.meters[f'res{res}-acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters[f'res{res}-acc5'].update(acc5.item(), n=batch_size)
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
        .format(top1=metric_logger.meters[f'res{res}-acc1'], top5=metric_logger.meters[f'res{res}-acc5'], losses=metric_logger.loss))

    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
            .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

                                                        # --------------------------------------------------------
# Modified by $@#Anonymous#@$
# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import time
import json
import random
import argparse
import datetime
import tqdm
import numpy as np
import wandb
from models.vmamba import Mlp
from models.vmamba import VSSBlock
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as nn
import torch.distributed as dist
from models.custom_gates import *
from models.custom_ssm import *
from models.moe_cuda import *
from models.custom_gates import *
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter

from config import get_config
from models import build_model
from data import build_loader
from utils.lr_scheduler import build_scheduler
from utils.optimizer import build_optimizer
from utils.logger import create_logger
from utils.utils import  NativeScalerWithGradNormCount, auto_resume_helper, reduce_tensor
from utils.utils import load_checkpoint_ema, load_pretrained_ema, save_checkpoint_ema
from models.vmamba import MoE_vmamba

from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count

from timm.utils import ModelEma as ModelEma

if torch.multiprocessing.get_start_method() != "spawn":
    print(f"||{torch.multiprocessing.get_start_method()}||", end="")
    torch.multiprocessing.set_start_method("spawn", force=True)

def str2bool(v):
    """
    Converts string to bool type; enables command line 
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, metavar="FILE", default="", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, default="/dataset/ImageNet_ILSVRC2012", help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', default=time.strftime("%Y%m%d%H%M%S", time.localtime()), help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    parser.add_argument('--fused_layernorm', action='store_true', help='Use fused layernorm.')
    parser.add_argument('--optim', type=str, help='overwrite optimizer if provided, can be adamw/sgd.')

    # EMA related parameters
    parser.add_argument('--model_ema', type=str2bool, default=True)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', type=str2bool, default=False, help='')

    parser.add_argument('--memory_limit_rate', type=float, default=-1, help='limitation of gpu memory use')
    parser.add_argument('--wandb', type=str2bool, default=True, help='Use wandb for logging')  # Add wandb argument



    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def main(config, args):
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)
    if args.wandb and dist.get_rank() == 0:
        wandb.init(project="moe_vmamba", config=config, name='start_200_100epoches_loss_balancing0_1_lr_gate1e-3_lrparams_0.0002819_16experts_noise_std0.1_top_2_upcycling_perturbed_router_epsw1e-2_epsx5e-3')
        #wandb.init(project="moe_vmamba", config=config, name='8_experts_hidden_dims_1_perturbed_router_epsw1e-2_epsx5e-3')
    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    checkpoint_path = '/home/ubuntu/trang/repo/VMamba_baseline/VMamba/classification/vssm1_tiny_0230s/20240917082057/ckpt_epoch_200.pth'
    checkpoint = torch.load(checkpoint_path)
    checkpoint_state_dict = checkpoint['model']

    model, model_clone = build_model(config)
    model_state_dict = model.state_dict()
    common_keys = {k: v for k, v in checkpoint_state_dict.items() if k in model_state_dict}

    # Debugging to check the common keys and their shapes
    # print("Common keys with matching shapes:")
    # for k in not_common_keys:
    #     print(f"Layer: {k}, Checkpoint shape: {checkpoint_state_dict[k].shape}, Model shape: {model_state_dict[k].shape}")
    # exit()
    # Update model state dict with common keys that match in shape
    model_state_dict.update(common_keys)

    # Load the updated state dict into the model
    model.load_state_dict(model_state_dict)
    
    mean = 0.0
    std = 0.1
    
    for layer, layer_clone in zip(model.layers, model_clone.layers):
        for block_idx in range(len(layer[0])):
            if len(layer[0]) % 2 == 0:
                if block_idx == len(layer[0]) - 1 or block_idx == len(layer[0]) - 3:
                    for name, module in layer[0][block_idx].named_children():
                        # print("Name")
                        # print(name)
                        # print("Module")
                        # print(module)
                        if isinstance(module, MoE_vmamba):
                            experts_module = module.fc1.experts
                            module_clone = layer_clone[0][block_idx].get_submodule(name)
                            fc1_layer_clone = module_clone.fc1
                            fc2_layer_clone = module_clone.fc2
                            
                            for i in range(experts_module.htoh4.num_expert):
                                weight_noise = torch.randn_like(fc1_layer_clone.weight.data, device='cuda') * std + mean
                                bias_noise = torch.randn_like(fc1_layer_clone.bias.data, device='cuda') * std + mean

                                # Add noise to the weight and bias before copying
                                noisy_weight = fc1_layer_clone.weight.data.to('cuda') + weight_noise
                                noisy_bias = fc1_layer_clone.bias.data.to('cuda') + bias_noise

                                experts_module.htoh4.weight[i].data.copy_(noisy_weight)
                                experts_module.htoh4.bias[i].data.copy_(noisy_bias)
                                
                            for i in range(experts_module.h4toh.num_expert):
                                weight_noise = torch.randn_like(fc2_layer_clone.weight.data, device='cuda') * std + mean
                                bias_noise = torch.randn_like(fc2_layer_clone.bias.data, device='cuda') * std + mean

                                # Add noise to the weight and bias before copying
                                noisy_weight = fc2_layer_clone.weight.data.to('cuda') + weight_noise
                                noisy_bias = fc2_layer_clone.bias.data.to('cuda') + bias_noise

                                experts_module.h4toh.weight[i].data.copy_(noisy_weight)
                                experts_module.h4toh.bias[i].data.copy_(noisy_bias)
                                
            else:
                if block_idx == len(layer[0]) - 2 or block_idx == len(layer[0]) - 4:
                    for name, module in layer[0][block_idx].named_children():
                        if isinstance(module, MoE_vmamba):
                            experts_module = module.fc1.experts
                            module_clone = layer_clone[0][block_idx].get_submodule(name)
                            fc1_layer_clone = module_clone.fc1
                            fc2_layer_clone = module_clone.fc2
                            
                            for i in range(experts_module.htoh4.num_expert):
                                weight_noise = torch.randn_like(fc1_layer_clone.weight.data, device='cuda') * std + mean
                                bias_noise = torch.randn_like(fc1_layer_clone.bias.data, device='cuda') * std + mean

                                # Add noise to the weight and bias before copying
                                noisy_weight = fc1_layer_clone.weight.data.to('cuda') + weight_noise
                                noisy_bias = fc1_layer_clone.bias.data.to('cuda') + bias_noise

                                experts_module.htoh4.weight[i].data.copy_(noisy_weight)
                                experts_module.htoh4.bias[i].data.copy_(noisy_bias)
                                
                            for i in range(experts_module.h4toh.num_expert):
                                weight_noise = torch.randn_like(fc2_layer_clone.weight.data, device='cuda') * std + mean
                                bias_noise = torch.randn_like(fc2_layer_clone.bias.data, device='cuda') * std + mean

                                # Add noise to the weight and bias before copying
                                noisy_weight = fc2_layer_clone.weight.data.to('cuda') + weight_noise
                                noisy_bias = fc2_layer_clone.bias.data.to('cuda') + bias_noise

                                experts_module.h4toh.weight[i].data.copy_(noisy_weight)
                                experts_module.h4toh.bias[i].data.copy_(noisy_bias)
                                               

    if dist.get_rank() == 0:
        if hasattr(model, 'flops'):
            logger.info(str(model))
            n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"number of params: {n_parameters}")
            #flops = model.flops()
            #logger.info(f"number of GFLOPs: {flops / 1e9}")
        else:
            logger.info(flop_count_str(FlopCountAnalysis(model, (dataset_val[0][0][None],))))
    torch.cuda.empty_cache()
    dist.barrier()
    model.cuda()
    model_without_ddp = model

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
        print("Using EMA with decay = %.8f" % args.model_ema_decay)


    optimizer = build_optimizer(config, model, logger)
    # checkpoint_lr_scheduler = checkpoint['lr_scheduler']
    # print("Keys in checkpoint_lr_scheduler:", checkpoint_lr_scheduler.keys())
    # exit()
    # checkpoint_optimizer_state = checkpoint['optimizer']
    # new_optimizer_state = optimizer.state_dict()
    # # Ensure the checkpoint has fewer parameter groups than the new optimizer
    # num_groups_checkpoint = len(checkpoint_optimizer_state['param_groups'])
    # num_groups_new = len(new_optimizer_state['param_groups'])

    # # Copy the matching part from the checkpoint optimizer to the new optimizer
    # for i in range(min(num_groups_checkpoint, num_groups_new)):
    #     # Copy param group state from checkpoint to new optimizer
    #     new_optimizer_state['param_groups'][i] = checkpoint_optimizer_state['param_groups'][i]
        
    #     # Update the state of each parameter in the matching param group
    #     for param_id in checkpoint_optimizer_state['param_groups'][i]['params']:
    #         new_optimizer_state['state'][param_id] = checkpoint_optimizer_state['state'][param_id]

    # # Load the updated state into the optimizer
    # optimizer.load_state_dict(new_optimizer_state)
    # exit()
    model = torch.nn.parallel.DistributedDataParallel(model, broadcast_buffers=False, find_unused_parameters=True)
    model._set_static_graph()
    loss_scaler = NativeScalerWithGradNormCount()

    if config.TRAIN.ACCUMULATION_STEPS > 1:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train) // config.TRAIN.ACCUMULATION_STEPS)
    else:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    if config.AUG.MIXUP > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    max_accuracy = 0.0
    max_accuracy_ema = 0.0

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        max_accuracy, max_accuracy_ema = load_checkpoint_ema(config, model_without_ddp, optimizer, lr_scheduler, loss_scaler, logger, model_ema)
        acc1, acc5, loss = validate(config, data_loader_val, model)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        if model_ema is not None:
            acc1_ema, acc5_ema, loss_ema = validate(config, data_loader_val, model_ema.ema)
            logger.info(f"Accuracy of the network ema on the {len(dataset_val)} test images: {acc1_ema:.1f}%")

        if config.EVAL_MODE:
            return

    if config.MODEL.PRETRAINED and (not config.MODEL.RESUME):
        load_pretrained_ema(config, model_without_ddp, logger, model_ema)
        acc1, acc5, loss = validate(config, data_loader_val, model)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        if model_ema is not None:
            acc1_ema, acc5_ema, loss_ema = validate(config, data_loader_val, model_ema.ema)
            logger.info(f"Accuracy of the network ema on the {len(dataset_val)} test images: {acc1_ema:.1f}%")
        
        if config.EVAL_MODE:
            return

    if config.THROUGHPUT_MODE and (dist.get_rank() == 0):
        logger.info(f"throughput mode ==============================")
        throughput(data_loader_val, model, logger)
        if model_ema is not None:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            throughput(data_loader_val, model_ema.ema, logger)
        return


    logger.info("Start training")
    start_time = time.time()
    best_acc1 = 0.0
    best_checkpoint_path = None
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler, model_ema)
        if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            save_checkpoint_ema(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, loss_scaler, logger, model_ema, max_accuracy_ema,is_best = False)

        acc1, acc5, loss = validate(config, data_loader_val, model)
        if acc1 > best_acc1:
            best_acc1 = acc1
            logger.info(f'New best accuracy: {best_acc1:.2f}%, saving best checkpoint.')
            if best_checkpoint_path and os.path.exists(best_checkpoint_path):
                os.remove(best_checkpoint_path)
            if dist.get_rank() == 0:
                best_checkpoint_path = save_checkpoint_ema(config, epoch, model_without_ddp, best_acc1, optimizer, lr_scheduler, loss_scaler, logger, model_ema, max_accuracy_ema,is_best = True)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        max_accuracy = max(max_accuracy, acc1)
        logger.info(f'Max accuracy: {max_accuracy:.2f}%')
        if model_ema is not None:
            acc1_ema, acc5_ema, loss_ema = validate(config, data_loader_val, model_ema.ema)
            logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1_ema:.1f}%")
            max_accuracy_ema = max(max_accuracy_ema, acc1_ema)
            logger.info(f'Max accuracy ema: {max_accuracy_ema:.2f}%')
        if args.wandb and dist.get_rank() == 0:
            wandb.log({'Epoch': epoch, 'Accuracy 1': acc1, 'Accuracy 5': acc5, 'Val Loss': loss})


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler, model_ema=None, model_time_warmup=50, load_balance = 0.01):
    torch.autograd.set_detect_anomaly(True)
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    model_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, (samples, targets) in enumerate(data_loader):
        torch.cuda.reset_peak_memory_stats()
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        data_time.update(time.time() - end)

        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            outputs = model(samples)
        loss = criterion(outputs, targets)
        
        if load_balance > 0:
            balance_loss = 0
            for name, m in model.named_modules():
                if isinstance(m, CustomNaiveGate_Balance_SMoE) or isinstance(
                    m, CustomNaiveGate_Balance_XMoE
                ):
                    # print("balance loss: ", m.loss)
                    # exit()
                    if m.loss is not None:
                        balance_loss += m.loss
            loss += load_balance * balance_loss
            
        loss = loss / config.TRAIN.ACCUMULATION_STEPS

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=config.TRAIN.CLIP_GRAD,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0)
        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            optimizer.zero_grad()
            lr_scheduler.step_update((epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS)
            if model_ema is not None:
                model_ema.update(model)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        if grad_norm is not None:  # loss_scaler return None if not update
            norm_meter.update(grad_norm)
        scaler_meter.update(loss_scale_value)
        batch_time.update(time.time() - end)
        end = time.time()
        if args.wandb and dist.get_rank() == 0:
            wandb.log({
                'Train Loss': loss_meter.avg,
                'Learning Rate Normal params': optimizer.param_groups[0]["lr"],
                'Learning Rate Gate': optimizer.param_groups[2]["lr"],
                'Batch Time': batch_time.avg,
                'Epoch': epoch
            })
        if idx > model_time_warmup:
            model_time.update(batch_time.val - data_time.val)

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[2]['lr']
            wd = optimizer.param_groups[2]['weight_decay']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t wd {wd:.4f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'data time {data_time.val:.4f} ({data_time.avg:.4f})\t'
                f'model time {model_time.val:.4f} ({model_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def validate(config, data_loader, model):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            output = model(images)

        # measure accuracy and record loss
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        acc1 = reduce_tensor(acc1)
        acc5 = reduce_tensor(acc5)
        loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return


if __name__ == '__main__':
    args, config = parse_option()

    if config.AMP_OPT_LEVEL:
        print("[warning] Apex amp has been deprecated, please use pytorch amp instead!")

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(rank)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    dist.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    if True: 
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    # to make sure all the config.OUTPUT are the same
    config.defrost()
    if dist.get_rank() == 0:
        obj = [config.OUTPUT]
        # obj = [str(random.randint(0, 100))] # for test
    else:
        obj = [None]
    dist.broadcast_object_list(obj)
    dist.barrier()
    config.OUTPUT = obj[0]
    print(config.OUTPUT, flush=True)
    config.freeze()
    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())
    logger.info(json.dumps(vars(args)))

    if args.memory_limit_rate > 0 and args.memory_limit_rate < 1:
        torch.cuda.set_per_process_memory_fraction(args.memory_limit_rate)
        usable_memory = torch.cuda.get_device_properties(0).total_memory * args.memory_limit_rate / 1e6
        print(f"===========> GPU memory is limited to {usable_memory}MB", flush=True)

    main(config, args)

import os
from torch.utils.tensorboard import SummaryWriter
import argparse
import albumentations as A
import time
import random
import numpy as np
from tabulate import tabulate

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from networks.Unet import UNet2D
from networks.Masked_Modeling import Masked_Modeling
from dataloaders.landmark_dataset import CL_Landmark_Mix
from train_util import set_logger, get_eta_time, generate_heatmap
import matplotlib.pyplot as plt

def get_arguments():
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument("--base_path", type=str, default='./data/', help="Path to the dataset.")
    parser.add_argument('--image_size', type=tuple,  default=(512, 512), help='patch size')

    # optimization options
    parser.add_argument('--exp', type=str,  default="unet", help='model')
    parser.add_argument('--batch_size', type=int,  default=2, help='batch size')
    parser.add_argument('--base_lr', type=float,  default=0.001, help='learning rate')
    parser.add_argument('--max_epochs', type=int,  default=150, help='maximum epoch number')
    parser.add_argument('--decay', type=str, default="[50, 100]" , help='decay epoch number')
    parser.add_argument('--eval_epoch', type=int,  default=1, help='eval epoch number')
    parser.add_argument('--save_epoch', type=int,  default=20, help='save epoch number')
    parser.add_argument('--kernel', type=int,  default=37, help='kernel size')
    parser.add_argument('--sigma', type=int,  default=6, help='sigma for kernel')
    parser.add_argument('--peak', type=int,  default=100, help='peak for the kernel')
    parser.add_argument('--lamda_c', type=float,  default=1.0, help='lamda for consistency loss')
    parser.add_argument('--lamda_m', type=float,  default=3.0, help='lamda for modeling loss')
    parser.add_argument('--mask_num', type=int,  default=7, help='masked number')
    parser.add_argument('--theta', type=float,  default=1, help='theta for transformer lr')

    # others
    parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
    parser.add_argument('--seed', type=int, default=1337, help='set the seed of random initialization')
    parser.add_argument("--save_path", type=str, default='./results', help="Path to save.")
    parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
    
    return parser.parse_args()

def set_seed(args):
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

def get_augmentations(args):
    return {
        "train": A.Compose([
            A.Resize(args.image_size[0], args.image_size[1]),
            A.Equalize(p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, p=0.3),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3)
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)),
        "test": A.Compose([
            A.Resize(args.image_size[0], args.image_size[1]),
        ], keypoint_params=A.KeypointParams(format='xy'))
    }  

def get_model(args):
    if args.exp == 'unet':
        net = UNet2D(in_channels=3)
        print('load unet')
    else:
        raise NotImplementedError
    return net.cuda()

if __name__ == "__main__":
    args = get_arguments()
    set_seed(args)
    resultdir = f'{args.save_path}/{args.exp}_{args.batch_size}_{args.max_epochs}_{args.base_lr}_{args.lamda_c}_{args.lamda_m}_{args.image_size}_{args.kernel}_{args.sigma}_{args.seed}'
    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    logger, writer = set_logger(args, resultdir, mode='simple')
    logger.info(f'Using GPU {args.gpu}')
    
    net = get_model(args)
    net.train()
    modeling_net = Masked_Modeling()
    modeling_net.cuda()
    modeling_net.train()
    augmentation = get_augmentations(args)
    train_data = CL_Landmark_Mix(base_dir=args.base_path, splits="train", augmentation=augmentation['train'])
    test_data = CL_Landmark_Mix(base_dir=args.base_path, splits="val", augmentation=augmentation['test'])
    logger.info(f'train data: {len(train_data)} samples, test data: {len(test_data)} samples')
    trainloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,  pin_memory=True, worker_init_fn=worker_init_fn)#, num_workers=8)
    testloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False,  pin_memory=True, worker_init_fn=worker_init_fn)#, num_workers=8)
    params_net = list(net.parameters())
    param_group_net = {'params': params_net, 'lr': args.base_lr}
    params_modeling_net = list(modeling_net.parameters())
    param_group_modeling_net = {'params': params_modeling_net, 'lr': args.base_lr * args.theta}
    combined_params = [param_group_net, param_group_modeling_net]

    optimizer = optim.SGD(combined_params, momentum=0.9, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=eval(args.decay), gamma=0.1)
    logger.info(f'scheduler milestone is {args.decay}')

    max_epoch = args.max_epochs
    start_time = time.time()
    for epoch_num in range(max_epoch):
        loss_all, loss_regression_all, loss_consistency_all, loss_modeling_all = 0, 0, 0, 0
        for i_batch, sampled_batch in enumerate(trainloader):
            image, keypoints = sampled_batch['image'].float().cuda(), sampled_batch['keypoints']
            heatmap = generate_heatmap(Gaussian_size=args.kernel, sigma=args.sigma, peak=args.peak, keypoints=keypoints, image=image)
            heatmap = heatmap.cuda()
            feature_outputs = net(image)
            feature_outputs = [F.interpolate(feature_output, size=heatmap.shape[-2:], mode='bilinear', align_corners=True) for feature_output in feature_outputs]
            features_cat = torch.cat(feature_outputs, dim=1)
            
            prototypes_all = []
            for batch_index in range(image.size(0)):
                features = features_cat[batch_index]
                heatmaps = heatmap[batch_index]
                prototypes = []
                for i in range(len(heatmaps)):
                    valid = features * heatmaps[i]
                    prototypes.append((valid.sum(dim=[1, 2]) / heatmaps[i].sum()).unsqueeze(0))
                prototypes = torch.cat(prototypes, dim=0)
                prototypes_all.append(prototypes.unsqueeze(0))
            prototypes_all = torch.cat(prototypes_all, dim=0) 
            mean_prototype = prototypes_all.mean(dim=0)
        
            # Compute MSE loss between each prototype and the mean prototype
            loss_consistency = F.mse_loss(prototypes_all, mean_prototype.repeat(prototypes_all.size(0), 1, 1))
        
            # masked modeling on the keypoint
            keypoints_all = (sampled_batch['keypoints'].float().cuda() / args.image_size[0])  # [batch_size, 10, 2]
            keypoints_features = prototypes_all.transpose(0, 1)  # [10, batch_size, feature_dim]
            
            keypoints_feature_pred, keypoints_feature_gt, prtotypes, mask = modeling_net(keypoints_features, return_mask=True, xy_embed=keypoints_all, mask_num=args.mask_num)
            keypoints_feature_pred = keypoints_feature_pred.transpose(0, 1)
            keypoints_feature_gt = keypoints_feature_gt.transpose(0, 1)
            prtotypes = prtotypes.transpose(0, 1)
            # Calculate modeling loss for the entire batch
            modeling_loss = F.mse_loss(keypoints_feature_pred, keypoints_feature_gt)*2
            mean_prototypes = torch.mean(prototypes_all, dim=0)
            
            if epoch_num == 0:
                prototypes_pool = net.set_prtotype(mean_prototypes)
            else:
                prototypes_pool = net.ema_update_prototype(mean_prototypes, epoch_num*len(trainloader)+i_batch+1)
            
            similarity = torch.einsum('kd,bdwh->bkwh', prototypes_pool.float(), features_cat.float())
            loss_regression = F.mse_loss(similarity, heatmap)*2
            
            loss = loss_regression + args.lamda_c * loss_consistency + args.lamda_m*modeling_loss 
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_all+=loss.item()
            loss_consistency_all+=loss_consistency.item()
            loss_regression_all+=loss_regression.item()
            loss_modeling_all+=modeling_loss.item()
        logger.info(f'[{epoch_num}/{max_epoch}] train loss={(loss_all/(i_batch+1)):.4f}, loss_regression={(loss_regression_all/(i_batch+1)):.4f}, loss_consistency={(loss_consistency_all/(i_batch+1)):.4f}, loss_modeling={(loss_modeling_all/(i_batch+1)):.4f} ,lr: {optimizer.param_groups[0]["lr"]:.5f}, eta: {get_eta_time(start_time, epoch_num+1, max_epoch)}')
        writer.add_scalar('train/loss', loss_all/(i_batch+1), epoch_num)
        writer.add_scalar('train/loss_regression', loss_regression_all/(i_batch+1), epoch_num)
        writer.add_scalar('train/loss_consistency', loss_consistency_all/(i_batch+1), epoch_num)
        scheduler.step()
        if epoch_num % args.eval_epoch == 0:
            net.eval()
            with torch.no_grad():
                loss_all, loss_regression_all, loss_consistency_all = 0, 0, 0
                for i_batch, sampled_batch in enumerate(testloader):
                    image, keypoints = sampled_batch['image'].cuda(), sampled_batch['keypoints']
                    heatmap = generate_heatmap(Gaussian_size=args.kernel, sigma=args.sigma, peak=args.peak, keypoints=keypoints, image=image)
                    heatmap = heatmap.cuda()
                    feature_outputs = net(image)
                    feature_outputs = [F.interpolate(feature_output, size=heatmap.shape[-2:], mode='bilinear', align_corners=True) for feature_output in feature_outputs]
                    features_cat = torch.cat(feature_outputs, dim=1)
                    similarity = torch.einsum('kd,bdwh->bkwh', net.prototype.float(), features_cat.float())
                    loss_regression = F.mse_loss(similarity, heatmap)*2

                    prototypes_all = []
                    for batch_index in range(image.size(0)):
                        features = features_cat[batch_index]
                        heatmaps = heatmap[batch_index]
                        prototypes = []
                        for i in range(len(heatmaps)):
                            valid = features * heatmaps[i]
                            prototypes.append((valid.sum(dim=[1, 2]) / heatmaps[i].sum()).unsqueeze(0))
                        prototypes = torch.cat(prototypes, dim=0)
                        prototypes_all.append(prototypes.unsqueeze(0))
                    prototypes_all = torch.cat(prototypes_all, dim=0) 
                    mean_prototype = prototypes_all.mean(dim=0)
                    loss_consistency = F.mse_loss(prototypes_all, mean_prototype.repeat(prototypes_all.size(0), 1, 1))
                    
                    loss = loss_regression + args.lamda_c * loss_consistency
                    loss_all+=loss.item()
                    loss_consistency_all+=loss_consistency.item()
                    loss_regression_all+=loss_regression.item()
                if epoch_num % args.save_epoch == 0:
                    save_mode_path = f'{resultdir}/checkpoints/model_{epoch_num}.pth'
                    torch.save(net.state_dict(), save_mode_path)
                    logger.info("save model to {}".format(save_mode_path))
                    save_prototype_path = f'{resultdir}/checkpoints/prototype_{epoch_num}.pth'
                    torch.save(net.prototype, save_prototype_path)
                    save_modeling_path = f'{resultdir}/checkpoints/modeling_{epoch_num}.pth'
                    torch.save(modeling_net.state_dict(), save_modeling_path)
                net.train()
                logger.info(f'[{epoch_num}/{max_epoch}] test loss={(loss_all/(i_batch+1)):.4f}, loss_regression={(loss_regression_all/(i_batch+1)):.4f}, loss_consistency={(loss_consistency_all/(i_batch+1)):.4f}')
                writer.add_scalar('test/loss', loss_all/(i_batch+1), epoch_num)
                writer.add_scalar('test/loss_regression', loss_regression_all/(i_batch+1), epoch_num)
                writer.add_scalar('test/loss_consistency', loss_consistency_all/(i_batch+1), epoch_num)
    save_mode_path = f'{resultdir}/checkpoints/model_final.pth'
    torch.save(net.state_dict(), save_mode_path)
    logger.info("save model to {}".format(save_mode_path))
    # save prtotypes
    save_prototype_path = f'{resultdir}/checkpoints/prototype_final.pth'
    torch.save(net.prototype, save_prototype_path)
    # save modeling
    save_modeling_path = f'{resultdir}/checkpoints/modeling_final.pth'
    torch.save(modeling_net.state_dict(), save_modeling_path)
    writer.close()
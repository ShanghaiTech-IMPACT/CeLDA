import os
import numpy as np

import math
from torch.utils.tensorboard import SummaryWriter

import torch
import time
import logging


def set_logger(args, resultdir, mode='full'):
    logdir = os.path.join(resultdir, 'logs')
    savedir = os.path.join(resultdir, 'checkpoints')
    shotdir = os.path.join(resultdir, 'snapshot')
    print('Result path: {}\nLogs path: {}\nCheckpoints path: {}\nSnapshot path: {}'.format(resultdir, logdir, savedir, shotdir))

    os.makedirs(logdir, exist_ok=True)
    os.makedirs(savedir, exist_ok=True)
    os.makedirs(shotdir, exist_ok=True)

    writer = SummaryWriter(logdir)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if mode == 'full':
        formatter = logging.Formatter('%(asctime)s %(filename)s %(funcName)s [line:%(lineno)d] %(levelname)s %(message)s')
    elif mode == 'simple':
        formatter = logging.Formatter('%(asctime)s %(filename)s [line:%(lineno)d] %(message)s')
    else:
        formatter = logging.Formatter('%(message)s')

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    fh = logging.FileHandler(shotdir+'/'+'snapshot.log', encoding='utf8')
    fh.setFormatter(formatter) 
    logger.addHandler(fh)
    logging.info(str(args))
    return logger, writer

def get_eta_time(start_time, iter_num, total_iter):
    elapsed_time = time.time() - start_time
    estimated_time = (elapsed_time / iter_num) * (total_iter - iter_num)
    hours, rem = divmod(estimated_time, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:02d}:{:02d}:{:02d}".format(int(hours), int(minutes), int(seconds))



def create_Gaussian(Gaussian_size=7, sigma=3, peak=1):
    kernel = np.zeros((Gaussian_size, Gaussian_size))
    center = Gaussian_size // 2
    for x in range(Gaussian_size):
        for y in range(Gaussian_size):
            dist_sq = (x - center) ** 2 + (y - center) ** 2
            kernel[x, y] = np.exp(-dist_sq / (2 * sigma ** 2))
    kernel /= (2 * np.pi * sigma ** 2)
    kernel *= peak / kernel.max()
    return kernel

def generate_heatmap(Gaussian_size=11, sigma=6, peak=1, keypoints=None, image=None):
    B, C, H, W = image.shape
    heatmap = np.zeros((B, keypoints.shape[1], H, W))
    kernel = create_Gaussian(Gaussian_size, sigma, peak)

    for i in range(keypoints.shape[1]): 
        for j in range(B):   
            x, y = keypoints[j, i]   
            x, y = int(x), int(y)

            x1, x2 = max(0, x - Gaussian_size // 2), min(W, x + Gaussian_size // 2 + 1)
            y1, y2 = max(0, y - Gaussian_size // 2), min(H, y + Gaussian_size // 2 + 1)

            kx1, ky1 = max(0, Gaussian_size // 2 - x), max(0, Gaussian_size // 2 - y)
            kx2, ky2 = Gaussian_size - max(0, x + Gaussian_size // 2 + 1 - W), Gaussian_size - max(0, y + Gaussian_size // 2 + 1 - H)
            adjusted_kernel = kernel[ky1:ky2, kx1:kx2]
            heatmap[j, i, y1:y2, x1:x2] += adjusted_kernel
    return torch.from_numpy(heatmap).float()
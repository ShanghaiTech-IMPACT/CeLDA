import os
import cv2
import torch
import numpy as np
import albumentations as A

import matplotlib.pyplot as plt
from networks.Unet import UNet2D

def get_model(exp, base_param_path, param_name='final'):
    if exp == 'unet':
        net = UNet2D(in_channels=3, out_channels=1)
    else:
        raise ValueError('Invalid model name')
    state_dict = torch.load(f'{base_param_path}/checkpoints/model_{param_name}.pth')
    net.load_state_dict(state_dict)
    prototype = torch.load(f'{base_param_path}/checkpoints/prototype_{param_name}.pth')
    net.set_prtotype(prototype)
    net.eval()
    return net

def read_image(image_file, aug):
    file = open(f"./data/{image_file[1]}/txt/" + image_file[0] + ".txt", "r")
    keypoints_tmp = eval(file.readlines()[0])
    keypoints = []
    for i in range(10):
        for ii in keypoints_tmp:
            if eval(ii['type']) == i+1:
                keypoints.append((int(ii['data'][0]['x']), int(ii['data'][0]['y'])))
                break
    image = cv2.imread(f"./data/{image_file[1]}/dataset/" + image_file[0] + ".jpg")
    raw_shape = image.shape[:-1]
    augmentation = aug(image=image, keypoints=keypoints)
    image = augmentation['image']
    keypoints = augmentation['keypoints']
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
    keypoints = torch.from_numpy(np.array(keypoints)).unsqueeze(0)
    keypoints = keypoints.numpy()
    keypoints = np.array(keypoints, dtype=int)
    return image, keypoints, raw_shape

def mappping_back(pred, gt, raw_shape, image_size=512):
    # resize with max size 2048
    index = np.argmax(raw_shape)
    scale = 2048 / raw_shape[index]
    aug = A.Compose([
                        A.Resize(int(raw_shape[0]*scale), int(raw_shape[1]*scale)),
                    ], keypoint_params=A.KeypointParams(format='yx'))
    pred = aug(image=np.zeros((image_size,image_size,3)), keypoints=pred)['keypoints']
    gt = aug(image=np.zeros((image_size,image_size,3)), keypoints=gt)['keypoints']

    return np.array(pred), np.array(gt)


def save_visual(save_path, save_name, image, pred_all, gt_all, split=None, error_num=None):
    save_name = f'{error_num:.2f}_{save_name}'.replace('.', '_') if error_num is not None else save_name
    image_to_show = np.clip(np.transpose(image[0], (1, 2, 0))*0.224 + 0.456, a_min=0, a_max=1)
    plt.figure(figsize=(10, 10))
    plt.imshow(image_to_show)

    for index, pred in enumerate(pred_all):
        gt = gt_all[index]
        plt.scatter(pred[0], pred[1], c='green', s=75, label='Predicted' if index == 0 else "")
        plt.scatter(gt[0], gt[1], c='red', s=75, label='Ground Truth' if index == 0 else "")
        plt.plot([pred[0], gt[0]], [pred[1], gt[1]], c='yellow')
    plt.axis('off')
    if split is None:
        os.makedirs(save_path, exist_ok=True)
        file_path = os.path.join(save_path, save_name)
    else:
        os.makedirs(os.path.join(save_path, split), exist_ok=True)
        file_path = os.path.join(save_path, split, save_name)

    plt.savefig(file_path, dpi=300, bbox_inches='tight', pad_inches=0) 
    plt.close()

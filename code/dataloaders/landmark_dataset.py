import os
import json
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms as T
        
class CL_Landmark_Mix(Dataset):
    def __init__(self, base_dir=None, splits="train", augmentation=None):
        self._base_dir = base_dir
        self.splits = splits
        self.augmentation = augmentation
        self.image_list = self.read_list(base_dir, splits)
        print(f'reading data: {len(self.image_list)} samples with {splits} split')
        self.normalize = T.Compose([
            T.ToTensor(), 
            T.Normalize(
                [0.485, 0.456, 0.406], 
                [0.229, 0.224, 0.225])
        ])

    def read_list(self, base_dir, splits):
        with open(os.path.join(base_dir,'final_splits.json'), 'r') as f:
            data = json.load(f)
        return data[splits]
    
    def __len__(self):
        return len(self.image_list)
    

    def __getitem__(self, idx):
        file_name = self.image_list[idx][0]
        prefex = self.image_list[idx][1]
        image = cv2.imread(os.path.join(self._base_dir, prefex, "dataset", file_name+".jpg"), 0)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        raw_shape = image.shape
        if self.splits in ['train', "test", "val"]:
            anno_file = open(os.path.join(self._base_dir, prefex, "txt", file_name+".txt"))
            anno = eval(anno_file.readlines()[0])
            keypoints = []
            for i in range(10):
                for ii in anno:
                    if eval(ii['type']) == i+1:
                        keypoints.append((int(ii['data'][0]['x']), int(ii['data'][0]['y'])))
                        break
                    
                
            if self.augmentation!=None:
                augmentation = self.augmentation(image=image, keypoints=keypoints)
                image = augmentation['image']
                keypoints = augmentation['keypoints']
            if len(keypoints) != 10:
                print(f'file {file_name} with {len(keypoints)} keypoints!')
            keypoints = np.array(keypoints)
            image = self.normalize(image)
            return {
                "image": torch.FloatTensor(image), 
                "keypoints": keypoints,
                "raw_shape": raw_shape,
                "file_name": file_name}
            
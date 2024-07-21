import json
import torch.nn.functional as F
import tabulate
import json

from infer_util import *
from tqdm import tqdm

import argparse

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default='./', help="base path")
    parser.add_argument("--param_name", type=str, default='final', help="param_name")
    parser.add_argument('--exp', type=str,  default="unet", help='model')
    parser.add_argument("--image_size", type=int, default=512, help="iamge size")
    return parser.parse_args()

def main():
    args = get_arguments()

    base_dir = args.base_dir if base_dir is None else base_dir
    param_name = args.param_name if param_name is None else param_name
    net = get_model(exp=args.exp, base_param_path=base_dir, param_name=param_name)
    net.cuda()
    save_path = f'{base_dir}/infer/{param_name}/'

    with open("./data/final_splits.json", 'r') as f:
        data_all = json.load(f)
    val_list = data_all['val']

    name = {0:"A", 1:"ANS", 2:"UI", 3:"UIA", 4:"Or", 5:"P", 6:"LI", 7:"LIA", 8:"Sn", 9:"Pog"}
    headers = ['index', 'Mean', 'SDR@2', 'SDR@2.5', 'SDR@3', 'SDR@4']
    table = {'adult':[], 'under_age':[], 'all': []}
    error = {'adult':[], 'under_age':[], 'all': []}
    results = {'adult': {i:[] for i in range(10)}, 'under_age': {i:[] for i in range(10)}, 'all': {i:[] for i in range(10)}}
    aug = A.Compose([A.Resize(args.image_size, args.image_size), A.Normalize()], keypoint_params=A.KeypointParams(format='xy'))

    save_res = {}
    for i in tqdm(range(len(val_list))):
        save_name, split = val_list[i]
        val_file_name = val_list[i]
        image, keypoints, raw_shape = read_image(val_file_name, aug)
        with torch.no_grad():
            features = net(image.cuda())
        features = [F.interpolate(feature_output, size=image.shape[-2:], mode='bilinear', align_corners=True) for feature_output in features]
        features = torch.cat(features, dim=1).squeeze().cpu()
        similarity = torch.einsum('bi,ijk->bjk', net.prototype.cpu().float(), features.float()).numpy()
        pred_raw, gt_raw = [], []
        for ii in range(len(similarity)):
            threshold = np.percentile(similarity[ii], 99.95)
            similarity[ii][similarity[ii] < threshold] = 0
            tmp = np.mean(np.argwhere(similarity[ii] > 0), axis=0)
            pred_raw.append([tmp[1], tmp[0]])
            gt_raw.append(keypoints[0][ii])
        pred, gt = mappping_back(pred_raw, gt_raw, raw_shape, args.image_size)
        error_tmp = np.linalg.norm(pred - gt, axis=1)*0.1
        save_visual(save_path=save_path, split=split, save_name=save_name, image=image, pred_all=pred_raw, gt_all=gt_raw, error_num=np.mean(error_tmp))
        for ii in range(len(error_tmp)):
            results[split][ii].append(error_tmp[ii])
            results['all'][ii].append(error_tmp[ii])
            error[split].append(error_tmp[ii])
            error['all'].append(error_tmp[ii])
        save_res[save_name] = {'pred': pred_raw, 'gt': gt_raw, 'error': error_tmp.tolist()}

    for target_split in table:
        for i in range(len(results[target_split])):
            SDR = {2:[], 2.5:[], 3:[], 4:[]}
            for dis in SDR.keys():
                SDR[dis] = np.sum(np.array(results[target_split][i]) < dis) / len(results[target_split][i])
            mean_error = np.mean(results[target_split][i])
            table[target_split].append([name[i], mean_error, SDR[2], SDR[2.5], SDR[3], SDR[4]])

    for target_split in table:
        total_mean = f'{np.mean([row[1] for row in table[target_split]]):.3f}({np.std([row[1] for row in table[target_split]]):.3f})' 
        total_SDR2 = np.mean([row[2] for row in table[target_split]])
        total_SDR25 = np.mean([row[3] for row in table[target_split]])
        total_SDR3 = np.mean([row[4] for row in table[target_split]])  
        total_SDR4 = np.mean([row[5] for row in table[target_split]]) 
        table[target_split].append(["Mean", total_mean, total_SDR2, total_SDR25, total_SDR3, total_SDR4])

    txt_file = open(save_path+f"results_{param_name}.txt", 'w')
    for target_split in table:
        print(f"{target_split} results:")
        txt_file.writelines(f"{target_split} results: \n")
        print(tabulate.tabulate(table[target_split], headers=headers, tablefmt="simple"))
        txt_file.writelines(tabulate.tabulate(table[target_split], headers=headers, tablefmt="simple") + "\n")
    txt_file.close()

    with open(save_path+f"results_{param_name}.json", 'w') as f:
        json.dump(save_res, f, cls=NpEncoder)

if __name__ == "__main__":
    main()
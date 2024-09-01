# Cephalometric Landmark Detection across Ages with Prototypical Network

by [Han Wu](https://hanwu.website/)\*, [Chong Wang](https://cwangrun.github.io/)\*, Lanzhuju Mei, Tong Yang, Min Zhu, [Dingggang Shen](https://idea.bme.shanghaitech.edu.cn/), and [Zhiming Cui](https://shanghaitech-impact.github.io/)<sup>+</sup>.

[[Paper](https://arxiv.org/abs/2406.12577)]   [[Project Page](https://shanghaitech-impact.github.io/CeLDA/)]


This repository contains the code and dataset for our paper "Cephalometric Landmark Detection across Ages with Prototypical Network" in MICCAI 2024.

## Updates
- [09/2024] Dataset released!
- [07/2024] Source code released!
- [06/2024] Our paper is accepted by MICCAI 2024!

## Getting Started
Run the following command to install the required packages:

```
git clone https://github.com/ShanghaiTech-IMPACT/CeLDA.git

cd CeLDA

pip install -r requirements.txt
```

## Training and Testing

To train the model, run the following command:


```
python code/train_CeLDA.py --batch_size 2 --base_lr 0.001 --max_epochs 150 --decay [50,100] --lamda_c 1 --lamda_m 0.1 --theta 1 --mask_num 7
```
To test the model, run the following command:

```
python code/test.py --base_dir /path/to/your/saved/model
```

## Dataset

★ Our dataset is available for reserach purpose only. To apply for CephAdoAdu dataset, please fill out the [form](./data_registration.pdf) and send the **signed e-copy** to Han Wu (email: wuhan2022@shanghaitech.edu.cn) as well as **CC your advisor** as mentioned in Sec.5 of the form. We will send you the dataset link and password when recieving the data registration form.

## Citation

If you find this code or dataset useful, please cite our paper:

    @article{wu2024cephalometric,
        title={Cephalometric Landmark Detection across Ages with Prototypical Network}, 
        author={Han Wu, Chong Wang, Lanzhuju Mei, Tong Yang, Min Zhu, Dingggang Shen, Zhiming Cui},
        journal={arXiv preprint arXiv:2406.12577},
        year={2024}
      }

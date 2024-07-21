
python code/train_CeLDA.py --batch_size 2 --base_lr 0.001 --max_epochs 150 --decay [50,100] --lamda_c 1 --lamda_m 0.1 --theta 1 --mask_num 7

python code/test.py --base_dir ./results/


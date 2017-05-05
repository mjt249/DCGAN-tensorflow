echo "job name: " $1
echo task index: $2
echo cuda device: $3
echo datasetdir: 00$3

CUDA_VISIBLE_DEVICES=$3 python main.py --task_index=$2 --checkpoint_dir=/datadrive/checkpoints  --logdir=/datadrive/checkpoints/logs --job_name=$1 --input_height=108 --is_train --is_crop True --dataset_dir /datadrive/data_share_max_madeleine --dataset 00$2



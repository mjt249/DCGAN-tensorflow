echo $1
echo $2
echo $3
CUDA_VISIBLE_DEVICES=$3 python main.py --task_index=$2 --checkpoint_dir=/datadrive/checkpoints --job_name=$1 --input_height=108 --is_train --is_crop True --dataset_dir /datadrive/data_share_max_madeleine --dataset merged

#python main.py --dataset_dir /datadrive/data_share_max_madeleine --dataset merged --input_height=108 --is_train --is_crop True --checkpoint_dir $WORK/dcgan


#!/usr/bin/env bash

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}
echo ${@:3}s
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS --master_port=$PORT $(dirname "$0")/train.py --config $CONFIG --launcher pytorch ${@:3}
# Any arguments from the third one are captured by ${@:3}
# tools/dist_train.sh ./configs/Aicity/swin_base_patch244_window877_kinetics400_1k.py 4 --work-dir ./work_dir/train/total --validate


# python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" \
#     --nproc_per_node=2 --master_port=29500 $(dirname "$0")/train.py ./configs/Aicity/swin_base_patch244_window877_kinetics400_1k.py --launcher pytorch ${@:3}
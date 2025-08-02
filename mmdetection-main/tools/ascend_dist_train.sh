#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
NNODES=$WORLD_SIZE
NODE_RANK=$RANK
PORT=$MASTER_PORT
ADDR=$MASTER_ADDR

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    $CONFIG \
    --launcher pytorch ${@:3}

#!/bin/bash

CONFIG_FILE="./configs/uscod/Look_Twice_.py"
MASTER_ADDR='localhost'
MASTER_PORT=11145
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=1
MODEL_PATH=""

while getopts "c:p:g:m:" opt; do
  case "$opt" in
    c) CONFIG_FILE="$OPTARG" ;;  
    p) MASTER_PORT="$OPTARG" ;;  
    g) GPUS_PER_NODE="$OPTARG" ;; 
    m) MODEL_PATH="$OPTARG" ;;  
    ?) echo "Usage: $0 [-c config_file] [-p master_port] [-g gpus] [-m model_path]"; exit 1 ;;
  esac
done

if [ -z "$MODEL_PATH" ]; then
  echo "Error: Model path (-m) is required."
  exit 1
fi

DISTRIBUTED_ARGS="--mixed_precision fp16 \
                  --machine_rank $NODE_RANK\
                  --num_machines $NNODES\
                  --main_process_port $MASTER_PORT \
                  --num_processes $GPUS_PER_NODE"

if [ $GPUS_PER_NODE -gt 1 ]; then
  DISTRIBUTED_ARGS="$DISTRIBUTED_ARGS --multi_gpu"
fi

OPTS=""
OPTS+="--config $CONFIG_FILE"
OPTS+=" --load_from $MODEL_PATH"

export NCCL_DEBUG=""
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=./
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export HF_ENDPOINT='https://hf-mirror.com'
CMD="accelerate launch $DISTRIBUTED_ARGS scripts/eval.py $OPTS"

echo $CMD
echo "PYTHONPATH=${PYTHONPATH}"
${CMD}
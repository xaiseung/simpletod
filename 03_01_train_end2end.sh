


export MODEL=$2
export MODEL_NAME=$3
export BATCH=$4
export OUTPUT=output_new/${MODEL_NAME}

export TRAIN_FILE=./resources/llama3/train.history_belief_action_sys_delex
export TEST_FILE=./resources/llama3/val.history_belief_action_sys_delex


CUDA_VISIBLE_DEVICES=$1 python 03_xai_main.py \
    --output_dir=$OUTPUT \
    --model_type=$MODEL \
    --model_name_or_path=$MODEL_NAME \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE \
    --evaluate_during_training \
    --save_steps 5000 \
    --logging_steps 2500 \
    --per_gpu_train_batch_size $BATCH \
    --num_train_epochs 100 \
    --block_size 1024 \
    --gradient_accumulation_steps 16 \
    --learning_rate 0.00005 \
    --weight_decay 0.05 \
    --warmup_steps 2000 \
    --use_lora
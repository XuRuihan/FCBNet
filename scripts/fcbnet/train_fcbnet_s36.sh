DATA_PATH=/disk2/xrh/datasets/imagenet1k
CODE_PATH=./ # modify code path here


ALL_BATCH_SIZE=4096
NUM_GPU=8
GRAD_ACCUM_STEPS=8 # Adjust according to your GPU numbers and memory size.
let BATCH_SIZE=ALL_BATCH_SIZE/NUM_GPU/GRAD_ACCUM_STEPS
MASTER_PORT=29502

cd $CODE_PATH && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.launch \
--nproc_per_node=$NUM_GPU \
--master_port=$MASTER_PORT \
train.py $DATA_PATH \
--model fcbnet_s36 --opt adamw --lr 4e-3 --warmup-epochs 20 \
-b $BATCH_SIZE --grad-accum-steps $GRAD_ACCUM_STEPS \
--drop-path 0.3 --head-dropout 0.0 \
> log/fcbnet_s36_dynamic.log 2>&1

DATA_PATH=datasets/imagenet1k
CODE_PATH=./ # modify code path here


ALL_BATCH_SIZE=2048
NUM_GPU=4
GRAD_ACCUM_STEPS=8 # Adjust according to your GPU numbers and memory size.
let BATCH_SIZE=ALL_BATCH_SIZE/NUM_GPU/GRAD_ACCUM_STEPS
MASTER_PORT=29502

cd $CODE_PATH && CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch \
--nproc_per_node=$NUM_GPU \
--master_port=$MASTER_PORT \
train.py $DATA_PATH \
--model convformer_xs12 --opt adamw --lr 4e-3 --warmup-epochs 20 \
-b $BATCH_SIZE --grad-accum-steps $GRAD_ACCUM_STEPS \
--drop-path 0.2 --head-dropout 0.0 \
> log/convformer_xs12.log 2>&1
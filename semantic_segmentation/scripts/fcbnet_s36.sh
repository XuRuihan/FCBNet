CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./tools/dist_train.sh \
    configs/upernet/fcbnet-s36_upernet_8xb2-amp-160k_ade20k-512x512.py \
    8 --resume \

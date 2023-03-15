CUDA_VISIBLE_DEVICES=$1 python train.py \
--config configs/llff.txt \
--ckpt log/trex/trex.th \ 
--render_only 1 
--render_test 1 
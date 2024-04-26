expname=train
CUDA_VISIBLE_DEVICES=$1 python train_style.py \
--config configs/tnt_style.txt \
--datadir /mnt/sda/Datasets/tanks_temples_nerf++/tanks_and_temples/tat_intermediate_Train \
--expname $expname \
--ckpt log_style/$expname/$expname.th \
--style_img /mnt/sda/Projects/StyleGaussian/images/old/0.jpg \
--render_only 1 \
--render_train 1 \
--render_test 0 \
--render_path 0 \
--chunk_size 1024 \
--rm_weight_mask_thre 0.0001 \
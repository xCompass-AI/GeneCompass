export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7



python -m torch.distributed.launch --nproc_per_node=8 \
--nnodes=3  \
--node_rank=0 \
--master_port=23455 \
--master_addr="192.168.48.4" \
pretrain_genecompass_w_human_mouse_100M.py \



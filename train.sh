HF_ENDPOINT=https://hf-mirror.com \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.run \
--nproc_per_node=4 \
--master_port=23333 \
train.py \
--cs_ratio=0.1 > r0.1g0-3.txt 2>&1
export CUDA_VISIBLE_DEVICES="0, 1, 2, 3, 4, 5, 6, 7"
export PYTHONPATH="."

python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=8 --master_addr="127.0.0.9" --master_port=10002 main.py --cfg /home/ubuntu/trang/repo/VMamba_global_local_lr/classification/configs/vssm/vmambav2v_tiny_224.yaml  --batch-size 128 --data-path /home/ubuntu/data/imagenet --output /home/ubuntu/trang/repo/VMamba_global_local_lr/classification/

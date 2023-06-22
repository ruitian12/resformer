python -m torch.distributed.launch --nproc_per_node 4 main.py --data-path /DATACENTER/raid5/tian/imagenet_2012 \
    --model resformer_small_patch16  --output_dir /DATACENTER/raid5/tian/exp/debug --batch-size 16 \
    --pin-mem --num_workers 8  --auto-resume  --input-size 224 --multi-res-mode iter
    
# --accum-iter 2
# --use-checkpoint
#  --distillation-type 'smooth-l1' 
#  --distillation-type 'l2' 
#  --input-size 224 160 128
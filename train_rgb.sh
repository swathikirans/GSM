python main_swathikiran.py something-v1 RGB --split 1 --arch InceptionV3 --num_segments 16 --consensus_type avg \
                           --batch-size 16 --iter_size 2 --dropout 0.5 --lr 0.01 --warmup 10 --epochs 60 \
                           --eval-freq 5 --gd 20 --run_iter 1 -j 8 --npb --gsm
python test_models.py diving48 RGB models/diving48_RGB_InceptionV3_avg_segment16_checkpoint.pth.tar \
                      --arch InceptionV3 --crop_fusion_type avg --test_segments 16  --test_crops 1 --num_clips 1 \
                      --gsm --save_scores
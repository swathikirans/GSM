python test_models.py something-v1 RGB models/something-v1_RGB_InceptionV3_avg_segment16_checkpoint.pth.tar \
                      --arch InceptionV3 --crop_fusion_type avg --test_segments 16  --test_crops 1 --num_clips 2 --gsm
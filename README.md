



# Gate-Shift Networks for Video Action Recognition
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/gate-shift-networks-for-video-action/action-recognition-in-videos-on-something-1)](https://paperswithcode.com/sota/action-recognition-in-videos-on-something-1?p=gate-shift-networks-for-video-action)

We release the code and trained models of our paper [Gate-Shift Networks for Video Action Recognition](https://arxiv.org/pdf/1912.00381.pdf). If you find our work useful for your research, please cite
```
@article{sudhakaran2019gate,
  title={{Gate-Shift Networks for Video Action Recognition}},
  author={Sudhakaran, Swathikiran and Escalera, Sergio and Lanz, Oswald},
  journal={arXiv preprint arXiv:1912.00381},
  year={2019}
}
```

### Prerequisites
- Python 3.5
- PyTorch 1.2
- [TensorboardX](https://github.com/lanpa/tensorboardX)

### Data preparation

- **Something Something-v1**: Download the frames from the [official website](https://20bn.com/datasets/something-something/v1#download). Copy the directory containing frames and the train-val files to `dataset-->something-v1`.
Run `python data_scripts/process_dataset_something.py` to create the train/val list files.

- **Diving48**: Download the videos and the annotations from the [official website](http://www.svcl.ucsd.edu/projects/resound/dataset.html). Copy the directory containing videos and the annotations to the directory `dataset-->Diving48`.
Run `python data_scripts/extract_frames_diving48.py` for extracting the frames from the videos.
Run `python data_scripts/process_dataset_diving.py` for creating the train/test list files.

### Training
```
python main.py something-v1 RGB --arch BNInception \
               --num_segments 8 --consensus_type avg \
               --batch-size 16 --iter_size 2 --dropout 0.5 \
               --lr 0.01 --warmup 10 --epochs 60 --eval-freq 5 \
               --gd 20 --run_iter 1 -j 16 --npb --gsm
```

### Testing
```
python test_models.py something-v1 RGB models/something-v1_RGB_InceptionV3_avg_segment16_checkpoint.pth.tar \
		      --arch InceptionV3 --crop_fusion_type avg \
                      --test_segments 16 --test_crops 1 --num_clips 1 --gsm
```
To evaluate using 2 clips sampled from each model, change ``--num_clips 1`` to ``--num_clips 2``. 
For prediction using ensemble of models, perform evaluation with the option ``--save_scores`` to save the prediction scores and run ``python average_scores.py``.


### Models
The models can be downloaded by running ``python download_models.py``.
The table shows the results reported in the paper. To reproduce the results, run the script obtained when clicked on the accuracy scores.
<table style="width:100%" align="center">  
<col width="150">
<tr>  
	<th>No. of frames</th>  
	<th>Top-1 Accuracy (%)</th>
    <th>Something Something-v1</th>
</tr>  
<tr>  
	<td align="center">8</td>  
	<td align="center"><a href='https://drive.google.com/open?id=15RdG7EwPw29rk_HRI85zbo5uT13HOaHj'>49.01</a></td>  
    <td rowspan='7'>
    <a href="http://www.youtube.com/watch?feature=player_embedded&v=j7tM4vPEMfs
" target="_blank"><img src="http://img.youtube.com/vi/j7tM4vPEMfs/0.jpg" 
alt="Visualization" width="480" height="320" border="10" /></a>
</td>
</tr>  
<tr>  
	<td align="center">12</td>  
	<td align="center"><a href='https://drive.google.com/open?id=1L0BotmQYZ7bUukjq_kHh1T1yF-sDeH5W'>51.58</a></center></td>  
</tr>  
<tr>  
	<td align="center">16</td>  
	<td align="center"><a href='https://drive.google.com/open?id=1_mREukiDspnDZsrbnJggNhQHTj9lKQ7X'>50.63</a></td>
</tr>  
<tr>  
	<td align="center">24</td>  
	<td align="center"><a href='https://drive.google.com/file/d/1JEnLXhiBAvnafnZ5zNrSq4-iDWM-GFgz/view?usp=sharing'>49.63</a></td>  
</tr>  
<tr>  
	<td align="center">8x2</td>  
	<td align="center"><a href='https://drive.google.com/open?id=1R0U160iVde-wp9LXgTXWbETKXbMH-wXl'>50.43</a></td>  
</tr>  
<tr>  
	<td align="center">12x2</td>  
	<td align="center"><a href='https://drive.google.com/open?id=18M8z941pSP4WjsPQupQkzCfqOpV3c-88'>51.98</a></td>  
</tr>  
<tr>  
	<td align="center">8x2 + 12x2 + 16 + 24</td>  
	<td align="center"><a href='https://drive.google.com/file/d/1ACf0vFI_5scHrLNs5mgdzs70AGHdTA5y/view?usp=sharing'>55.16</a></td>  
</tr>  
</table>



To reproduce the results on Diving48 dataset, click on [39.03%](https://drive.google.com/file/d/1JvviRj1x6p-eo1YJwvkOF50-OuvgjL0d/view?usp=sharing) (16 frames) and  [40.27%](https://drive.google.com/file/d/1N5HPeGO5bNtXh4NEL5vpMiP7vwq7XCoS/view?usp=sharing) (16x2 frames).


### Acknowledgements

This implementation is built upon the [TRN-pytorch](https://github.com/metalbubble/TRN-pytorch) codebase which is based on [TSN-pytorch](https://github.com/yjxiong/tsn-pytorch). We thank Yuanjun Xiong and Bolei Zhou for releasing TSN-pytorch and TRN-pytorch repos.

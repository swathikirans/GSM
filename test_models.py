import argparse
import time

import numpy as np
import torch.nn.parallel
import torch.optim
from sklearn.metrics import confusion_matrix
from dataset import VideoDataset
from models import VideoModel
from transforms import *
from ops import ConsensusModule
import datasets_video
import pdb
from torch.nn import functional as F
import sys


# options
parser = argparse.ArgumentParser(
    description="TRN testing on the full validation set")
parser.add_argument('dataset', type=str, choices=['something-v1','diving48'])
parser.add_argument('modality', type=str, choices=['RGB', 'Flow', 'RGBDiff'])
parser.add_argument('weights', type=str)
parser.add_argument('--arch', type=str, default="BNInception")
parser.add_argument('--save_scores', default=False, action="store_true")
parser.add_argument('--test_segments', type=int, default=8)
parser.add_argument('--max_num', type=int, default=-1)
parser.add_argument('--test_crops', type=int, default=1)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--crop_fusion_type', type=str, default='avg')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--num_clips',type=int, default=1,help='Number of clips sampled from a video')
parser.add_argument('--softmax', type=int, default=0)
parser.add_argument('--gsm', default=False, action="store_true")

args = parser.parse_args()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
         correct_k = correct[:k].view(-1).float().sum(0)
         res.append(correct_k.mul_(100.0 / batch_size))
    return res

args.train_list, args.val_list, args.root_path, prefix = datasets_video.return_dataset(args.dataset)

if args.dataset == 'something-v1':
    num_class = 174
    args.rgb_prefix = ''
    rgb_read_format = "{:05d}.jpg"
elif args.dataset == 'diving48':
    num_class = 48
    args.rgb_prefix = 'frames'
    rgb_read_format = "{:05d}.jpg"
else:
    raise ValueError('Unknown dataset '+args.dataset)


net = VideoModel(num_class=num_class, num_segments=args.test_segments, modality=args.modality,
          base_model=args.arch, consensus_type=args.crop_fusion_type, gsm=args.gsm)

checkpoint = torch.load(args.weights)
print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))

base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(checkpoint['state_dict'].items())}
net.load_state_dict(base_dict, strict=False)

if args.test_crops == 1:
    cropping = torchvision.transforms.Compose([
        GroupScale(net.scale_size),
        GroupCenterCrop(net.input_size),
    ])
elif args.test_crops == 10:
    cropping = torchvision.transforms.Compose([
        GroupOverSample(net.input_size, net.scale_size)
    ])
elif args.test_crops == 5:
    cropping = torchvision.transforms.Compose([
        GroupFiveCrops(net.input_size, net.scale_size)
    ])
else:
    raise ValueError("Only 1 and 10 crops are supported while we got {}".format(args.test_crops))

data_loader = torch.utils.data.DataLoader(
        VideoDataset(args.root_path, args.val_list, num_segments=args.test_segments,
                   new_length=1 if args.modality == "RGB" else 5,
                   modality=args.modality,
                   image_tmpl=args.rgb_prefix+rgb_read_format,
                   test_mode=True,
                   transform=torchvision.transforms.Compose([
                       cropping,
                       Stack(roll=(args.arch in ['BNInception','InceptionV3'])),
                       ToTorchFormatTensor(div=(args.arch not in ['BNInception','InceptionV3'])),
                       GroupNormalize(net.input_mean, net.input_std),
                   ]), num_clips=args.num_clips),
        batch_size=1, shuffle=False,
        num_workers=args.workers * 2, pin_memory=True)

if args.gpus is not None:
    devices = [args.gpus[i] for i in range(args.workers)]
else:
    devices = list(range(args.workers))


net = torch.nn.DataParallel(net.cuda())
net.eval()

data_gen = enumerate(data_loader)

total_num = len(data_loader.dataset)
output = []
output_scores = []


def eval_video(video_data):
    i, data, label = video_data
    num_crop = args.test_crops

    if args.modality == 'RGB':
        length = 3
    elif args.modality == 'Flow':
        length = 10
    elif args.modality == 'RGBDiff':
        length = 18
    else:
        raise ValueError("Unknown modality "+args.modality)
    if args.num_clips > 1:
        input_var = data.view(-1, length, data.size(3), data.size(4))
    else:
        input_var = data.view(-1, length, data.size(2), data.size(3))

    rst = net(input_var)
    if args.softmax==1:
        # take the softmax to normalize the output to probability
        rst = F.softmax(rst)

    rst = rst.data.cpu().numpy().copy()

    rst = rst.reshape(-1, 1, num_class)

    return i, rst, label[0]


proc_start_time = time.time()
max_num = args.max_num if args.max_num > 0 else len(data_loader.dataset)

top1 = AverageMeter()
top5 = AverageMeter()

with torch.no_grad():
    for i, (data, label) in data_gen:
        if i >= max_num:
            break
        rst = eval_video((i, data, label))
        output.append(rst[1:])
        cnt_time = time.time() - proc_start_time
        output_scores.append(np.mean(rst[1], axis=0))
        prec1, prec5 = accuracy(torch.from_numpy(np.mean(rst[1], axis=0)), label, topk=(1, 5))
        top1.update(prec1, 1)
        top5.update(prec5, 1)
        print('video {} done, total {}/{}, average {:.3f} sec/video, moving Prec@1 {:.3f} Prec@5 {:.3f}'.format(i, i+1,
                                                                        total_num,
                                                                        float(cnt_time) / (i+1), top1.avg, top5.avg))

video_pred = [np.argmax(np.mean(x[0], axis=0)) for x in output]

video_labels = [x[1] for x in output]


cf = confusion_matrix(video_labels, video_pred).astype(float)

cls_cnt = cf.sum(axis=1)
cls_hit = np.diag(cf)

cls_acc = cls_hit / cls_cnt

print('-----Evaluation of {} is finished------'.format(args.weights))
print('Class Accuracy {:.02f}%'.format(np.mean(cls_acc) * 100))
print('Overall Prec@1 {:.02f}% Prec@5 {:.02f}%'.format(top1.avg, top5.avg))

if args.save_scores:

    # order_dict = {e:i for i, e in enumerate(sorted(name_list))}
    reorder_output = [None] * len(output)
    reorder_label = [None] * len(output)
    reorder_pred = [None] * len(output)
    output_csv = []
    for i in range(len(output)):
        reorder_output[i] = output_scores[i]
        reorder_label[i] = video_labels[i]
        reorder_pred[i] = video_pred[i]
    save_name = args.weights[:-8] + '_clips_' + str(args.num_clips) + '.npz'
    np.savez(save_name, scores=reorder_output, labels=reorder_label, predictions=reorder_pred, cf=cf)
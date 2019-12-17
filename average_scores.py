import numpy as np
import torch


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = output.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target)
    res = []
    for k in topk:
         correct_k = correct[:k].view(-1).float().sum(0)
         res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
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


files_scores = ['models/something-v1_RGB_InceptionV3_avg_segment8_checkpoint_clips_2.npz',
                'models/something-v1_RGB_InceptionV3_avg_segment12_checkpoint_clips_2.npz',
                'models/something-v1_RGB_InceptionV3_avg_segment16_checkpoint_clips_1.npz',
                'models/something-v1_RGB_InceptionV3_avg_segment24_checkpoint_clips_1.npz']

top1 = AverageMeter()
top5 = AverageMeter()


def compute_acc(labels, scores):
    preds_max = np.argmax(scores, 2)[:, 0]
    num_correct = np.sum(preds_max == labels)
    acc = num_correct * 1.0 / preds.shape[0]
    return acc

scores_agg = None
for filename in files_scores:
    data = np.load(filename)
    preds = data['predictions']
    labels = data['labels']
    scores = data['scores']
    if scores_agg is None:
        scores_agg = scores
    else:
        scores_agg += scores
    acc_scores = compute_acc(labels, scores)
    num_correct = np.sum(preds == labels)
    acc = num_correct * 1.0 / preds.shape[0]
for k in range(labels.shape[0]):
    label = torch.from_numpy(np.array([labels[k]]))
    score = torch.from_numpy(scores_agg[k,:])
    prec1, prec5 = accuracy(score, label, topk=(1, 5))
    top1.update(prec1.item(), 1)
    top5.update(prec5.item(), 1)
print('Accuracy of ensemble: top1:{} top5:{}'.format(top1.avg, top5.avg))

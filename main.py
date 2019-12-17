import argparse
import os
import time
import shutil
import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm_
from tensorboardX import SummaryWriter
from models import VideoModel
from transforms import *
from opts import parser
import CosineAnnealingLR
import datasets_video
from dataset import VideoDataset
import os
import sys


best_prec1 = 0

def main():
    finetuning = False

    global args, best_prec1
    args = parser.parse_args()
    check_rootfolders()

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

    model_dir = os.path.join('experiments', args.dataset, args.arch, args.consensus_type + '-' + args.modality,
                             str(args.run_iter))
    if not args.resume:
        if os.path.exists(model_dir):
            print('Dir {} exists!!!'.format(model_dir))
            sys.exit()
        else:
            os.makedirs(model_dir)
            os.makedirs(os.path.join(model_dir, args.root_log))

    writer = SummaryWriter(model_dir)

    args.train_list, args.val_list, args.root_path, prefix = datasets_video.return_dataset(args.dataset)


    if 'something' in args.dataset:
        # label transformation for left/right categories
        target_transforms = {86:87,87:86,93:94,94:93,166:167,167:166}
        print('Target transformation is enabled....')
    else:
        target_transforms = None

    args.store_name = '_'.join(
        [args.dataset, args.arch, args.consensus_type, 'segment%d' % args.num_segments])
    print('storing name: ' + args.store_name)

    model = VideoModel(num_class=num_class, modality=args.modality, num_segments=args.num_segments,
                       base_model=args.arch, consensus_type=args.consensus_type, dropout=args.dropout,
                       partial_bn=not args.no_partialbn, gsm=args.gsm, target_transform=target_transforms)

    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    policies = model.get_optim_policies()
    train_augmentation = model.get_augmentation()

    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print(("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

    cudnn.benchmark = True

    # Data loading code
    if args.modality != 'RGBDiff':
        normalize = GroupNormalize(input_mean, input_std)
    else:
        normalize = IdentityTransform()

    if args.modality == 'RGB':
        data_length = 1
    elif args.modality in ['Flow', 'RGBDiff']:
        data_length = 5

    train_loader = torch.utils.data.DataLoader(
        VideoDataset(args.root_path, args.train_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl=args.rgb_prefix+rgb_read_format,
                   transform=torchvision.transforms.Compose([
                       train_augmentation,
                       Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                       ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                       normalize,
                   ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        VideoDataset(args.root_path, args.val_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl=args.rgb_prefix+rgb_read_format,
                   random_shift=False,
                   transform=torchvision.transforms.Compose([
                       GroupScale(int(scale_size)),
                       GroupCenterCrop(crop_size),
                       Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                       ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                       normalize,
                   ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    if args.loss_type == 'nll':
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        raise ValueError("Unknown loss type")

    for group in policies:
        print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
            group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))

    optimizer = torch.optim.SGD(policies,
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    lr_scheduler_clr = CosineAnnealingLR.WarmupCosineLR(optimizer=optimizer, milestones=[args.warmup, args.epochs],
                                                        warmup_iters=args.warmup, min_ratio=1e-7)
    if args.resume:
        for epoch in range(0, args.start_epoch):
            lr_scheduler_clr.step()


    if args.evaluate:
        validate(val_loader, model, criterion, 0)
        return

    log_training = open(os.path.join(model_dir, args.root_log, '%s.csv' % args.store_name), 'a')
    for epoch in range(args.start_epoch, args.epochs):

        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch + 1)

        train_prec1 = train(train_loader, model, criterion, optimizer, epoch, log_training, writer=writer)

        lr_scheduler_clr.step()

        # evaluate on validation set
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            prec1 = validate(val_loader, model, criterion, (epoch + 1) * len(train_loader), log_training,
                             writer=writer, epoch=epoch)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'current_prec1': prec1,
                'lr': optimizer.param_groups[-1]['lr'],
            }, is_best, model_dir)
        else:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'current_prec1': train_prec1,
                'lr': optimizer.param_groups[-1]['lr'],
            }, False, model_dir)


def train(train_loader, model, criterion, optimizer, epoch, log, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    if args.no_partialbn:
        model.module.partialBN(False)
    else:
        model.module.partialBN(True)

    # switch to train mode
    model.train()
    loss_summ = 0
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var) / args.iter_size
        loss_summ += loss
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1,5))
        losses.update(loss_summ.data, input.size(0))
        top1.update(prec1, input.size(0))
        top5.update(prec5, input.size(0))

        # compute gradient and do SGD step
        loss.backward()

        if args.clip_gradient is not None:
            total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)
            if total_norm > args.clip_gradient:
                print("clipping gradient: {} with coef {}".format(total_norm, args.clip_gradient / total_norm))
        if (i+1) % args.iter_size == 0:
            # scale down gradients when iter size is functioning
            optimizer.step()
            optimizer.zero_grad()
            loss_summ = 0

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        epoch, i, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr']))
            print(output)
            writer.add_scalar('train/batch_loss', losses.avg, epoch * len(train_loader) + i)
            writer.add_scalar('train/batch_top1Accuracy', top1.avg, epoch * len(train_loader) + i)
            log.write(output + '\n')
            log.flush()
    writer.add_scalar('train/loss', losses.avg, epoch + 1)
    writer.add_scalar('train/top1Accuracy', top1.avg, epoch + 1)
    writer.add_scalar('train/top5Accuracy', top5.avg, epoch + 1)
    return top1.avg


def validate(val_loader, model, criterion, iter, log, epoch, writer):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(async=True)
            input_var = torch.autograd.Variable(input, volatile=True)
            target_var = torch.autograd.Variable(target, volatile=True)

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1,5))

            losses.update(loss.data, input.size(0))
            top1.update(prec1, input.size(0))
            top5.update(prec5, input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                output = ('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))
                print(output)
                log.write(output + '\n')
                log.flush()

    output = ('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
          .format(top1=top1, top5=top5, loss=losses))
    print(output)
    output_best = '\nBest Prec@1: %.3f'%(best_prec1)
    print(output_best)
    writer.add_scalar('test/loss', losses.avg, epoch + 1)
    writer.add_scalar('test/top1Accuracy', top1.avg, epoch + 1)
    writer.add_scalar('test/top5Accuracy', top5.avg, epoch + 1)
    log.write(output + ' ' + output_best + '\n')
    log.flush()

    return top1.avg

def save_checkpoint(state, is_best, model_dir):
    torch.save(state, '%s/%s_checkpoint.pth.tar' % (model_dir, args.store_name))
    if is_best:
        shutil.copyfile('%s/%s_checkpoint.pth.tar' % (model_dir, args.store_name),
                        '%s/%s_best.pth.tar' % (model_dir, args.store_name))
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


def adjust_learning_rate(optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    lr = args.lr * decay
    decay = args.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']

def adjust_learning_rate_step(optimizer, gamma):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    decay = args.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = gamma * param_group['lr'] * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']


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

def check_rootfolders():
    """Create log and model folder"""
    folders_util = [args.root_log, args.root_model, args.root_output]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.mkdir(folder)


if __name__ == '__main__':
    main()

import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
from tqdm import tqdm
import numpy as np
import faiss

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import pcl.loader
import pcl.builder
import wandb

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 12)')
parser.add_argument('--epochs', default=400, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.005, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[60, 100], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 100)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training.')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')

parser.add_argument('--low-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--pcl-r', default=16384, type=int,
                    help='queue size; number of negative pairs (default: 16384)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--temperature', default=0.2, type=float,
                    help='softmax temperature')

parser.add_argument('--mlp', action='store_true',
                    help='use mlp head')
parser.add_argument('--aug-plus', action='store_true',
                    help='use moco-v2/SimCLR data augmentation')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')

parser.add_argument('--num-cluster', default='500,1000,2000', type=str,  
                    help='number of clusters')
parser.add_argument('--warmup-epoch', default=1, type=int,
                    help='number of warm-up epochs to only train with InfoNCE loss')
parser.add_argument('--exp-dir', default='pth_pcl', type=str,
                    help='experiment directory')

# dataset setting 
parser.add_argument("--dataset", default="TinyImageNet", help="dataset")
parser.add_argument("--size" ,default=64, help="Image size")
parser.add_argument('--id', type=str, default='')


def main():
    args = parser.parse_args()
    wandb.init(project="Baseline", config=args, name=f"Pretrained_{args.id}_{args.arch}_{args.dataset}_{args.epochs}")
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dataset == "TinyImageNet":
        args.num_cluster = "200,500,1000"
        args.pcl_r = 16384
    elif args.dataset == "CIFAR10":
        args.num_cluster = "10,50,100"
        args.pcl_r = 1024
    elif args.dataset == "CIFAR100":
        args.num_cluster = "100,250,500"
        args.pcl_r = 4096
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    wandb.config.update({"num_cluster": args.num_cluster,"pcl-r": args.pcl_r}, allow_val_change=True)
    args.num_cluster = args.num_cluster.split(',')
    if not os.path.exists(args.exp_dir):
        os.mkdir(args.exp_dir)

    ngpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    # 保持 args.gpu 的原始值，並根據情況設置 GPU
    if gpu is not None:
        # 如果用戶明確指定了 GPU，則使用該 GPU
        torch.cuda.set_device(gpu)
    else:
        # 如果未指定 GPU，則默認使用 GPU 0
        gpu = 0
        torch.cuda.set_device(gpu)

    print("=> creating model '{}'".format(args.arch))
    model = pcl.builder.MoCo(
        models.__dict__[args.arch],
        args.low_dim, args.pcl_r, args.moco_m, args.temperature, args.mlp)
    model = model.cuda(gpu)
    print(model)

    criterion = nn.CrossEntropyLoss().cuda(gpu)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            loc = 'cuda:{}'.format(gpu)
            checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True


    # ---- 加入 dataset 設定邏輯 ---- #
    if args.dataset == "TinyImageNet":
        args.size = 64
        traindir = os.path.join(args.data, "train")
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    elif args.dataset == "CIFAR10":
        args.size = 32
        traindir = os.path.join(args.data)
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    elif args.dataset == "CIFAR100":
        args.size = 32
        traindir = os.path.join(args.data)
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    wandb.config.update({"size": args.size}, allow_val_change=True)
    # traindir = os.path.join(args.data, 'train')
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])

    if args.aug_plus:
        augmentation = [
            transforms.RandomResizedCrop(args.size, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([pcl.loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
    else:
        augmentation = [
            transforms.RandomResizedCrop(args.size, scale=(0.2, 1.)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]

    eval_augmentation = transforms.Compose([
        transforms.Resize(args.size),
        transforms.CenterCrop(args.size),
        transforms.ToTensor(),
        normalize
    ])

    # train_dataset = pcl.loader.ImageFolderInstance(
    #     traindir,
    #     pcl.loader.TwoCropsTransform(transforms.Compose(augmentation)))
    # eval_dataset = pcl.loader.ImageFolderInstance(
    #     traindir,
    #     eval_augmentation)
    if args.dataset == "TinyImageNet":
        train_dataset = pcl.loader.ImageFolderInstance(
            traindir,
            pcl.loader.TwoCropsTransform(transforms.Compose(augmentation)))
        eval_dataset = pcl.loader.ImageFolderInstance(
            traindir,
            eval_augmentation)

    elif args.dataset == "CIFAR10":
        train_dataset = datasets.CIFAR10(
            root=traindir,
            train=True,
            download=True,
            transform=pcl.loader.TwoCropsTransform(transforms.Compose(augmentation))
        )
        eval_dataset = datasets.CIFAR10(
            root=traindir,
            train=False,
            download=True,
            transform=eval_augmentation
        )
        
    elif args.dataset == "CIFAR100":
        train_dataset = datasets.CIFAR100(
            root=traindir,
            train=True,
            download=True,
            transform=pcl.loader.TwoCropsTransform(transforms.Compose(augmentation))
        )
        eval_dataset = datasets.CIFAR100(
            root=traindir,
            train=False,
            download=True,
            transform=eval_augmentation
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # ---- 檢查資料是否正確載入 ---- #
    print("Dataset:",args.dataset)
    print("image size:",args.size)
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")
    

    for epoch in range(args.start_epoch, args.epochs):
        cluster_result = None
        if epoch >= args.warmup_epoch:
            features = compute_features(eval_loader, model, args)
            cluster_result = run_kmeans(features, args)

        adjust_learning_rate(optimizer, epoch, args)
        train(train_loader, model, criterion, optimizer, epoch, args, cluster_result=cluster_result)

        if (epoch + 1) % 5 == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, is_best=False, filename='{}/checkpoint_{:04d}.pth.tar'.format(args.exp_dir, epoch))

def train(train_loader, model, criterion, optimizer, epoch, args, cluster_result):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    total_losses = AverageMeter('TotalLoss', ':.4e')
    info_losses = AverageMeter('InfoNCE_Loss', ':.4e')
    proto_losses = AverageMeter('ProtoLoss', ':.4e')
    acc_inst = AverageMeter('Acc@Inst', ':6.2f')
    acc_proto = AverageMeter('Acc@Proto', ':6.2f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, total_losses, info_losses, proto_losses, acc_inst, acc_proto],
        prefix="Epoch: [{}]".format(epoch))

    model.train()
    end = time.time()

    epoch_iterator = tqdm(train_loader, desc=f"Epoch {epoch}", unit="batch")

    for i, (images, index) in enumerate(epoch_iterator):
        data_time.update(time.time() - end)

        images[0] = images[0].cuda(args.gpu, non_blocking=True)
        images[1] = images[1].cuda(args.gpu, non_blocking=True)

        output, target, output_proto, target_proto = model(im_q=images[0], im_k=images[1], cluster_result=cluster_result, index=index)

        loss_info_value = criterion(output, target)
        info_losses.update(loss_info_value.item(), images[0].size(0))

        total_loss_value = loss_info_value

        if output_proto is not None:
            loss_proto_value = 0
            for proto_out, proto_target in zip(output_proto, target_proto):
                proto_target = proto_target.cuda(args.gpu, non_blocking=True)
                proto_loss = criterion(proto_out, proto_target)
                loss_proto_value += proto_loss
                accp = accuracy(proto_out, proto_target)[0]
                acc_proto.update(accp.item(), images[0].size(0))

            if len(args.num_cluster) > 0:
                loss_proto_value /= len(args.num_cluster)
                total_loss_value += loss_proto_value
                proto_losses.update(loss_proto_value.item(), images[0].size(0))

        total_losses.update(total_loss_value.item(), images[0].size(0))
        acc = accuracy(output, target)[0]
        acc_inst.update(acc.item(), images[0].size(0))

        optimizer.zero_grad()
        total_loss_value.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    wandb.log({
        "epoch": epoch,
        "total_loss": total_losses.avg,
        "info_nce_loss": info_losses.avg,
        "proto_nce_loss": proto_losses.avg,
        "accuracy_inst": acc_inst.avg,
        "accuracy_proto": acc_proto.avg,
    })

def compute_features(eval_loader, model, args):
    print('Computing features...')
    model.eval()
    features = torch.zeros(len(eval_loader.dataset), args.low_dim).cuda()
    for i, (images, index) in enumerate(tqdm(eval_loader)):
        with torch.no_grad():
            images = images.cuda(non_blocking=True)
            feat = model(images, is_eval=True)
            features[index] = feat
    return features.cpu()

def run_kmeans(x, args):
    print('performing kmeans clustering')
    results = {'im2cluster': [], 'centroids': [], 'density': []}

    for seed, num_cluster in enumerate(args.num_cluster):
        d = x.shape[1]
        k = int(num_cluster)
        clus = faiss.Clustering(d, k)
        clus.verbose = True
        clus.niter = 20
        clus.nredo = 5
        clus.seed = seed
        clus.max_points_per_centroid = 100
        clus.min_points_per_centroid = 10

        res = faiss.StandardGpuResources()
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = args.gpu
        index = faiss.GpuIndexFlatL2(res, d, cfg)

        clus.train(x, index)

        D, I = index.search(x, 1)
        im2cluster = [int(n[0]) for n in I]

        centroids = faiss.vector_to_array(clus.centroids).reshape(k, d)

        Dcluster = [[] for c in range(k)]
        for im, i in enumerate(im2cluster):
            Dcluster[i].append(D[im][0])

        density = np.zeros(k)
        for i, dist in enumerate(Dcluster):
            if len(dist) > 1:
                d = (np.asarray(dist) ** 0.5).mean() / np.log(len(dist) + 10)
                density[i] = d

        dmax = density.max()
        for i, dist in enumerate(Dcluster):
            if len(dist) <= 1:
                density[i] = dmax

        density = density.clip(np.percentile(density, 10), np.percentile(density, 90))
        density = args.temperature * density / density.mean()

        centroids = torch.Tensor(centroids).cuda()
        centroids = nn.functional.normalize(centroids, p=2, dim=1)

        im2cluster = torch.LongTensor(im2cluster).cuda()
        density = torch.Tensor(density).cuda()

        results['centroids'].append(centroids)
        results['density'].append(density)
        results['im2cluster'].append(im2cluster)

    return results

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr
    if args.cos:
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

if __name__ == '__main__':
    main()

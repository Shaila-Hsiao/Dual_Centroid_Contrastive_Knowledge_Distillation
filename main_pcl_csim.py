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

import torch.nn.functional as F
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
# parser.add_argument('--lr', '--learning-rate', default=0.005, type=float,
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
# parser.add_argument('--pcl-r', default=16384, type=int,
parser.add_argument('--pcl-r', default=256, type=int,
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

# parser.add_argument('--num-cluster', default='1000,2000,5000', type=str, 
parser.add_argument('--num-cluster', default='500,1000,2000', type=str, 
                    help='number of clusters')
parser.add_argument('--warmup-epoch', default=1, type=int,
                    help='number of warm-up epochs to only train with InfoNCE loss')
parser.add_argument('--exp-dir', default='experiment_pcl', type=str,
                    help='experiment directory')

parser.add_argument("--alpha", default=1, type=float, help="質心損失的權重")


def main():
    args = parser.parse_args()
    wandb.init(project="PCL", config=args)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

    # if args.gpu is not None:
    #     warnings.warn('You have chosen a specific GPU. This will completely '
    #                   'disable data parallelism.')

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

     # 初始化老師模型
    print("=> loading teacher model")
    teacher_model = pcl.builder.MoCo(
        models.__dict__[args.arch],
        args.low_dim, args.pcl_r, args.moco_m, args.temperature, args.mlp)
    print(f"Teacher model device: {next(teacher_model.parameters()).device}")

    print("=> creating student model '{}'".format(args.arch))
    model = pcl.builder.MoCo(
        models.__dict__[args.arch],
        args.low_dim, args.pcl_r, args.moco_m, args.temperature, args.mlp)
    model = model.cuda(gpu)
    # print(model)
    # 在此插入檢查代碼
    print(f"Teacher model device: {next(teacher_model.parameters()).device}")
    print(f"Student model device: {next(model.parameters()).device}")
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

    traindir = os.path.join(args.data, 'train')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if args.aug_plus:
        augmentation = [
            transforms.RandomResizedCrop(64, scale=(0.2, 1.)),
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
            transforms.RandomResizedCrop(64, scale=(0.2, 1.)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]

    eval_augmentation = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        normalize
    ])

    train_dataset = pcl.loader.ImageFolderInstance(
        traindir,
        pcl.loader.TwoCropsTransform(transforms.Compose(augmentation)))
    eval_dataset = pcl.loader.ImageFolderInstance(
        traindir,
        eval_augmentation)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)


    # 加載預訓練的老師模型權重
    checkpoint_path = r"D:\Document\Project\contrastivelearnig\TEST\moco\checkpoint_0099.pth.tar"
    checkpoint = torch.load(checkpoint_path, map_location="cpu",weights_only=True)
    state_dict = checkpoint["state_dict"]

    # 處理權重名稱
    new_state_dict = {}
    for k in list(state_dict.keys()):
        if k.startswith("encoder_q"):
            new_state_dict[f"encoder_q.{k[len('encoder_q.'):]}"] = state_dict[k]
        elif k.startswith("encoder_k"):
            new_state_dict[f"encoder_k.{k[len('encoder_k.'):]}"] = state_dict[k]

    # 加載權重到老師模型
    msg = teacher_model.load_state_dict(new_state_dict, strict=False)
    print(f"Missing keys: {msg.missing_keys}")
    print(f"Unexpected keys: {msg.unexpected_keys}")
    teacher_model = teacher_model.cuda(gpu)
    teacher_model.eval()

    # # 計算類別質心
    class_centroids = get_class_centroids(teacher_model, train_loader,gpu)
    class_centroids = class_centroids.cuda(gpu)
    # print(f"class_centroids device: {class_centroids.device}")

    for epoch in range(args.start_epoch, args.epochs):
        cluster_result = None
        if epoch >= args.warmup_epoch:
            # cluster_result
            features = compute_features(eval_loader, model, args)
            cluster_result = run_kmeans(features, args)

        adjust_learning_rate(optimizer, epoch, args)
        train(train_loader, model,teacher_model, criterion, optimizer, epoch, args, cluster_result,class_centroids)
        # train(train_loader, model,teacher_model, criterion, optimizer, epoch, args, cluster_result)

        if (epoch + 1) % 5 == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, is_best=False, filename='{}/checkpoint_{:04d}.pth.tar'.format(args.exp_dir, epoch))

def train(train_loader, model, teacher_model,criterion, optimizer, epoch, args, cluster_result,class_centroids):
# def train(train_loader, model, teacher_model,criterion, optimizer, epoch, args, cluster_result):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    total_losses = AverageMeter('TotalLoss', ':.4e')
    info_losses = AverageMeter('InfoNCE_Loss', ':.4e')
    proto_losses = AverageMeter('ProtoLoss', ':.4e')
    centroid_losses = AverageMeter('Centroid Loss', ':.4e')
    acc_inst = AverageMeter('Acc@Inst', ':6.2f')
    acc_proto = AverageMeter('Acc@Proto', ':6.2f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, total_losses, info_losses, proto_losses, acc_inst, acc_proto],
        prefix="Epoch: [{}]".format(epoch))

    model.train()
    teacher_model.eval()
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
        
        # # 切換為評估模式，計算質心對齊損失
        model.eval()
        with torch.no_grad():
            z_q = model.encoder_q(images[0]).cuda(args.gpu)  # 明確使用 query 編碼器
            z_k = model.encoder_k(images[1]).cuda(args.gpu)  # 明確使用 key 編碼器
            # # 在此插入檢查代碼
            # print(f"Class centroids device: {class_centroids.device}")
            # print(f"z_q device: {z_q.device}")
            # print(f"z_k device: {z_k.device}")

        # # # 確保所有計算都在 GPU
        # sim_q = cosine_similarity(z_q.unsqueeze(1), class_centroids.unsqueeze(0), dim=2).cuda(args.gpu)
        # sim_k = cosine_similarity(z_k.unsqueeze(1), class_centroids.unsqueeze(0), dim=2).cuda(args.gpu)
        sim_q = cosine_similarity_matrix(z_q, class_centroids)
        sim_k = cosine_similarity_matrix(z_k, class_centroids)


        # # 再次檢查 sim_q 和 sim_k
        # print(f"sim_q device: {sim_q.device}")
        # print(f"sim_k device: {sim_k.device}")

        # # Normalize centroids on GPU
        # print(f"Before normalizing centroids: {class_centroids.device}")
        class_centroids = F.normalize(class_centroids, p=2, dim=1).cuda(args.gpu)
        # print(f"After normalizing centroids: {class_centroids.device}")


        # print(f"z_q device: {z_q.device}, class_centroids device: {class_centroids.device}")
        loss_centroid_value = 1 - F.cosine_similarity(sim_q, sim_k).mean()
        # print(f"Loss centroid value calculated on device: {loss_centroid_value.device}")
        model.train()
        
        # # Update Centroid Loss
        total_loss_value += args.alpha * loss_centroid_value
        centroid_losses.update(loss_centroid_value.item(), images[0].size(0))

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
        "centroid_loss": centroid_losses.avg,
        "accuracy_inst": acc_inst.avg,
        "accuracy_proto": acc_proto.avg,
    })

def get_class_centroids(model, loader, device):
    print("計算類別質心...")
    centroids = {}
    count = {}
    
    model.eval()
    with torch.no_grad():
        for images, target in tqdm(loader):
            
            # 明確將資料搬移到 GPU，並轉換為 CUDA Tensor
            # images = images.cuda(device, non_blocking=True)
            images = [img.cuda(device, non_blocking=True) for img in images]
            # 使用 key encoder 提取特徵
            features = model.encoder_k(images[0]).to(device)
            
            for feature, label in zip(features, target):
                label = label.item()
                if label not in centroids:
                    centroids[label] = feature.cpu()
                    count[label] = 1
                else:
                    centroids[label] += feature.cpu()
                    count[label] += 1

    # 計算每個類別的平均值
    for label in centroids:
        centroids[label] /= count[label]
    class_centroids = torch.stack([centroids[key] for key in sorted(centroids.keys())]).to(device)
    # 在此插入檢查代碼
    # print(f"[DEBUG] Teacher model output feature dimension: {features.shape[-1]}")
    # print(f"Features device: {features.device}")
    # print(f"Class centroids device: {class_centroids.device}")
    return class_centroids
def cosine_similarity_matrix(z_q, class_centroids, eps=1e-8):
    # 正規化
    z_q_norm = F.normalize(z_q, p=2, dim=1)
    class_centroids_norm = F.normalize(class_centroids, p=2, dim=1)

    # 矩陣乘法計算餘弦相似度
    return torch.matmul(z_q_norm, class_centroids_norm.t())


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
        clus.max_points_per_centroid = 200
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
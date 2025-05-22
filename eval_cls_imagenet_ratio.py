import argparse
import builtins
import os
import random
import shutil
import time
import warnings
from datetime import datetime  
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
# import tensorboard_logger as tb_logger
import wandb
import wandb.sklearn

from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from resnet import StudentResNet50,student_resnet50

from early_stopping import EarlyStopping

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR', 
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=5., type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[60, 80], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by a ratio)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=0., type=float,
                    metavar='W', help='weight decay (default: 0.)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--pretrained', default='', type=str,
                    help='path to pretrained checkpoint')
parser.add_argument('--id', type=str, default='')

# dataset setting 
parser.add_argument("--dataset", default="TinyImageNet", help="dataset")
parser.add_argument("--size" ,default=64, help="Image size")
parser.add_argument("--num-classes" ,default=200, type=int)
parser.add_argument('--exp-dir', default='experiment_pcl', type=str,
                    help='experiment directory')
parser.add_argument('--student-ratio', default='60%', type=str,
                    choices=['80%', '60%', '40%', '20%'],
                    help='Student network block ratio')

# early stopping
parser.add_argument('--patience', default=20, type=int,
                    help='Number of epochs to wait for improvement before early stopping (default: 20)')

def main():
    args = parser.parse_args()
    wandb.init(project="Baseline",config=args,name=f"Cls_{args.id}_{args.arch}_{args.dataset}_{args.epochs}",tags=[f"{args.id}",f"{args.dataset}","Classification","mask_proportation"])
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
        
    timestamp = datetime.now().strftime("%Y%m%d")+"_bs"+str(args.batch_size)
    args.exp_dir = os.path.join(args.exp_dir, timestamp) 

    if not os.path.exists(args.exp_dir):
        # os.mkdir(args.exp_dir)
        os.makedirs(args.exp_dir, exist_ok=True)

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    
    args.tb_folder = 'Linear_eval/{}_tensorboard'.format(args.id)
    if not os.path.isdir(args.tb_folder):
        os.makedirs(args.tb_folder)
        
    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
        
    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass
        
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    
    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.dataset == "TinyImageNet":
        args.num_classes = 200
    elif args.dataset == "CIFAR10":
        args.num_classes = 10
        
    elif args.dataset == "CIFAR100":
        args.num_classes = 100
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    wandb.config.update({"num_classes": args.num_classes}, allow_val_change=True)

    print("=> creating student model (custom StudentResNet50)")
    student_block_configs = {
        '100%':[3,6,4,3],
        '80%': [3, 4, 4, 2],
        '60%': [3, 3, 3, 1],
        '40%': [2, 2, 2, 1],
        '20%': [1, 1, 1, 1],
    }

    block_counts = student_block_configs[args.student_ratio]
    backbone_fn = student_resnet50(block_counts)
    model = backbone_fn(num_classes=args.num_classes)


    # model = models.__dict__[args.arch](num_classes=args.num_classes)
    # freeze all layers but the last fc
    for name, param in model.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False
    # init the fc layer
    model.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.fc.bias.data.zero_()
    
    # if args.gpu==0:
        # logger = tb_logger.Logger(logdir=args.tb_folder, flush_secs=2)
    # else:
    logger = None
        
    # load from pre-trained, before DistributedDataParallel constructor
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")

            # rename pre-trained keys
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                # retain only encoder_q up to before the embedding layer
                if k.startswith('encoder_q') and not k.startswith('encoder_q.fc'):
                    # remove prefix
                    state_dict[k[len("encoder_q."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

            args.start_epoch = 0
            msg = model.load_state_dict(state_dict, strict=False)
            assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

            print("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # optimize only the linear classifier
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    assert len(parameters) == 2  # fc.weight, fc.bias
    optimizer = torch.optim.SGD(parameters, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    best_f1 = 0.0
    best_acc = 0.0
    best_recall = 0.0
    best_precision = 0.0
    best_conf_matrix = None
    best_class_report = ""

    # Data loading code
    # traindir = os.path.join(args.data, 'train')
    # valdir = os.path.join(args.data, 'val')
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])

    # train_dataset = datasets.ImageFolder(
    #     traindir,
    #     transforms.Compose([
    #         transforms.RandomResizedCrop(64),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         normalize,
    #     ]))
    # Data loading code
    
    # ---- 加入 dataset 設定邏輯 ---- #
    if args.dataset == "TinyImageNet":
        args.size = 64
        traindir = os.path.join(args.data, "train")
        valdir = os.path.join(args.data, 'val')
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(args.size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        val_dataset = datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(args.size),
                transforms.CenterCrop(args.size),
                transforms.ToTensor(),
                normalize,
            ]))
    elif args.dataset == "CIFAR10":
        args.size = 32
        traindir = os.path.join(args.data)
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        train_dataset = datasets.CIFAR10(
            root=traindir,
            train=True,
            download=True,
            transform= transforms.Compose([
                transforms.RandomResizedCrop(args.size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        )
        val_dataset = datasets.CIFAR10(
            root=traindir,
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.Resize(args.size),
                transforms.CenterCrop(args.size),
                transforms.ToTensor(),
                normalize,
            ])
        )

    elif args.dataset == "CIFAR100":
        args.size = 32
        traindir = os.path.join(args.data)
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        train_dataset = datasets.CIFAR100(
            root=traindir,
            train=True,
            download=True,
            transform= transforms.Compose([
                transforms.RandomResizedCrop(args.size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        )
        val_dataset = datasets.CIFAR100(
            root=traindir,
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.Resize(args.size),
                transforms.CenterCrop(args.size),
                transforms.ToTensor(),
                normalize,
            ])
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    wandb.config.update({"size": args.size}, allow_val_change=True)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
       val_dataset,batch_size=args.batch_size, shuffle=False,
       num_workers=args.workers, pin_memory=True)
    
    class_names = val_dataset.classes
    wandb.config.update({"class_names": class_names})

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    early_stopping = EarlyStopping(
        patience=args.patience,
        verbose=True,
        path=os.path.join(args.exp_dir, 'earlystop_best_loss.pth')
    )

    for epoch in range(args.start_epoch, args.epochs):

        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train_loss,train_acc1,train_acc5 = train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        eval_loss, eval_acc1, eval_acc5, precision, recall, f1, class_report, conf_matrix, all_targets, all_preds = \
        validate(val_loader, model, criterion, args, epoch)
        
        # check if f1 the best?
        is_best = f1 > best_f1
        if is_best:
            best_f1 = f1
            best_acc = eval_acc1
            best_recall = recall
            best_precision = precision
            save_best_results(args, epoch, f1, class_report, conf_matrix, val_dataset)

            # wandb log best performance
            wandb.log({
                "best/epoch": epoch,
                "best/f1": best_f1,
                "best/accuracy": best_acc,
                "best/precision": best_precision,
                "best/recall": best_recall,
            })
            # # wandb confusion matrix
            # if hasattr(val_loader.dataset, 'classes'):
            #     class_names = val_loader.dataset.classes
            # else:
            #     class_names = [str(i) for i in range(args.num_classes)]
            # wandb.sklearn.plot_confusion_matrix(all_targets, all_preds, class_names)
        
        # Early Stopping check
        early_stopping(eval_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

        wandb.log({
            "epoch":epoch,
            "training_loss":train_loss,
            "training_acc1":train_acc1,
            "training_acc5":train_acc5,
            "validate_loss":eval_loss,
            "validate_acc1":eval_acc1,
            "validate_acc5":eval_acc5,
            "validate_precision": precision,
            "validate_recall": recall,
            "validate_f1": f1,
            
        })
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            # if (epoch + 1) % 5 == 0:
            save_checkpoint(args,{
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            },is_best=is_best, filename='{}/checkpoint_{:04d}.pth.tar'.format(args.exp_dir, epoch))
            if epoch == args.start_epoch:
                sanity_check(model.state_dict(), args.pretrained)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))
    epoch_iterator = tqdm(train_loader, desc=f"Epoch {epoch}", unit="batch")
    """
    Switch to eval mode:
    Under the protocol of linear classification on frozen features/models,
    it is not legitimate to change any part of the pre-trained model.
    BatchNorm in train mode may revise running mean/std (even if it receives
    no gradient), which are part of the model parameters too.
    """
    model.eval()

    end = time.time()
    for i, (images, target) in enumerate(epoch_iterator):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    
    # return losses.avg,top1.avg,top5.avg
    return losses.avg, top1.avg, top5.avg


def validate(val_loader, model, criterion, args,  epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')
    epoch_iterator = tqdm(val_loader, desc=f"Epoch {epoch}", unit="batch")
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        all_preds = []
        all_targets = []
        end = time.time()
        for i, (images, target) in enumerate(epoch_iterator):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            _, preds = torch.max(output, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
    # if args.gpu==0:    
    #     logger.log_value('test_acc', top1.avg, epoch)
    #     logger.log_value('test_acc5', top5.avg, epoch)
    precision = precision_score(all_targets, all_preds, average='macro')  # or 'weighted'
    recall = recall_score(all_targets, all_preds, average='macro')
    f1 = f1_score(all_targets, all_preds, average='macro')
    print(f' * Precision: {precision:.4f} | Recall: {recall:.4f} | F1-score: {f1:.4f}')
    
    class_report = classification_report(all_targets, all_preds, digits=4)
    conf_matrix = confusion_matrix(all_targets, all_preds)

    return losses.avg, top1.avg, top5.avg, precision, recall, f1, class_report, conf_matrix, all_targets, all_preds

def save_best_results(args, epoch, f1, class_report, conf_matrix,val_dataset):
    """
    Save classification report and confusion matrix plot for best F1-score.
    """
    # Save classification report
    report_path = os.path.join(args.exp_dir, 'best_classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(class_report)

    # set class name
    if args.dataset == "CIFAR10":
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']
    elif args.dataset == "CIFAR100":
        class_names = val_dataset.classes
    elif args.dataset == "TinyImageNet":
        class_names = val_dataset.classes
    else:
        class_names = [str(i) for i in range(args.num_classes)]  # fallback

    # Plot and save confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix (Epoch {epoch}, F1={f1:.4f})')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    cm_path = os.path.join(args.exp_dir, 'best_confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()
    return report_path, cm_path


# def save_checkpoint(state, filename='checkpoint.pth.tar'):
#     torch.save(state, filename)
def save_checkpoint(args,state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        best_model_path = os.path.join(args.exp_dir, 'linear_best.pth.tar')
        shutil.copyfile(filename, best_model_path)
def sanity_check(state_dict, pretrained_weights):
    """
    Linear classifier should not change any weights other than the linear layer.
    This sanity check asserts nothing wrong happens (e.g., BN stats updated).
    """
    # print("=> loading '{}' for sanity check".format(pretrained_weights))
    # checkpoint = torch.load(pretrained_weights, map_location="cpu")
    # state_dict_pre = checkpoint['state_dict']

    # for k in list(state_dict.keys()):
    #     # only ignore fc layer
    #     if 'fc.weight' in k or 'fc.bias' in k:
    #         continue
    #     # name in pretrained model
    #     k_pre = 'module.encoder_q.' + k[len('module.'):] \
    #         if k.startswith('module.') else 'module.encoder_q.' + k

    #     assert ((state_dict[k].cpu() == state_dict_pre[k_pre]).all()), \
    #         '{} is changed in linear classifier training.'.format(k)

    # print("=> sanity check passed.")

    print("=> loading '{}' for sanity check".format(pretrained_weights))
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    state_dict_pre = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        # only ignore fc layer
        if 'fc.weight' in k or 'fc.bias' in k:
            continue

        # Modify the key to match the naming in state_dict_pre
        if k.startswith('module.'):
            k_pre = 'encoder_q.' + k[len('module.'):]
        else:
            k_pre = 'encoder_q.' + k

        if k_pre not in state_dict_pre:
            raise KeyError(f"Key {k_pre} not found in pretrained state_dict.")

        # Perform sanity check to ensure the weights match
        assert ((state_dict[k].cpu() == state_dict_pre[k_pre]).all()), \
            '{} is changed in linear classifier training.'.format(k)

    print("=> sanity check passed.")



class AverageMeter(object):
    """Computes and stores the average and current value"""
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
    """Decay the learning rate based on schedule"""
    lr = args.lr
    for milestone in args.schedule:
        lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()

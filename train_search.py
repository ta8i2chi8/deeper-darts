import os
import sys
import time
import glob
import numpy as np
import torch
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

import utils
from model_search import Network
from architect import Architect


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')                     # もともとは64
parser.add_argument('--learning_rate', type=float, default=0.05, help='init learning rate')       # もともとは0.025
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
# parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')  # not use for search
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--arch_reg_rate', type=float, default=1, help='regularization rate in arch search')
parser.add_argument('--arch_sn_width', type=float, default=0.3, help='the width of start_node in arch search')
args = parser.parse_args()


def main():
    start_time = time.strftime("%Y%m%d-%H%M%S")
    args.save = 'logs/search-{}-{}'.format(args.save, start_time)
    utils.create_exp_dir(args.save)

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %H:%M:%S')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    writer = SummaryWriter(log_dir="./runs/{}".format(start_time))

    CIFAR_CLASSES = 10

    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion)
    model = model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    train_transform, valid_transform = utils._data_transforms_cifar10(args)
    train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=2)

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True, num_workers=2)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=args.learning_rate_min)

    architect = Architect(model, args)

    for epoch in range(1, args.epochs + 1):
        lr = scheduler.get_last_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)

        genotype = model.genotype()
        logging.info('genotype = %s', genotype)

        print('アーキテクチャ選択のパラメータ（1. normal, 2. reduce）')
        print(F.softmax(model.alphas_normal, dim=-1))
        print(F.softmax(model.alphas_reduce, dim=-1))

        # ペナルティ項の大きさを更新
        architect.r_rate = args.arch_reg_rate * (args.epochs - epoch) / args.epochs

        # train
        train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr)
        logging.info('train_acc %f', train_acc)
        writer.add_scalar('search/accuracy/train', train_acc, epoch)
        writer.add_scalar('search/loss/train', train_obj, epoch)
        writer.add_scalar('search/real_penalty', architect.reg, epoch)
        writer.add_scalar('search/real_penalty * p_rate', architect.reg * architect.r_rate, epoch)

        # inference
        valid_acc, valid_obj = infer(valid_queue, model, criterion)
        logging.info('valid_acc %f', valid_acc)
        writer.add_scalar('search/accuracy/valid', valid_acc, epoch)
        writer.add_scalar('search/loss/valid', valid_obj, epoch)

        # 学習率スケジューラの更新
        scheduler.step()

        utils.save(model, os.path.join(args.save, 'weights.pt'))


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr):
    """
        引数： {
            train_queue: trainデータのデータローダー,
            valid_queue: validデータのデータローダー,
            model: Networkクラス（ネットワークアーキテクチャの実体）,
            architect: Architectクラス（アーキテクチャ探索用のクラス）,
            criterion: cross entropy loss,
            optimizer: optimizer,
            lr: 学習率（スケジューラによって変化するため）,
        }
    """

    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    valid_queue_iter = iter(valid_queue)

    model.train()

    for step, (input, target) in enumerate(train_queue):
        n = input.size(0)

        input = input.cuda()
        target = target.cuda(non_blocking=True)

        # 1バッチ分のデータ取得（アーキテクチャ探索に用いる用）
        input_search, target_search = next(valid_queue_iter)
        input_search = input_search.cuda()
        target_search = target_search.cuda(non_blocking=True)

        # アーキテクチャ(α)探索　（∂Lval(ω - lr * [∂Ltrain(ω,α) / ∂ω],α) / ∂α）
        architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)

        # 重み(ω)学習
        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        # loss, accuracy出力
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = input.cuda()
            target = target.cuda(non_blocking=True)

            logits = model(input)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if step % args.report_freq == 0:
                logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


if __name__ == '__main__':
    main()

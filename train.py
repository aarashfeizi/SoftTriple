"""
    PyTorch Package for SoftTriple Loss

    Reference
    ICCV'19: "SoftTriple Loss: Deep Metric Learning Without Triplet Sampling"

    Copyright@Alibaba Group

"""

import argparse
import os

import numpy as np
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
from PIL import Image
import loss
import evaluation as eva
import net
import timm
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('data', help='path to dataset')
parser.add_argument('-j', '--workers', default=2, type=int,
                    help='number of data loading workers')
parser.add_argument('--epochs', default=50, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number')
parser.add_argument('-b', '--batch_size', default=32, type=int,
                    help='mini-batch size')
parser.add_argument('--modellr', default=0.0001, type=float,
                    help='initial model learning rate')
parser.add_argument('--centerlr', default=0.01, type=float,
                    help='initial center learning rate')
parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float,
                    help='weight decay', dest='weight_decay')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--eps', default=0.01, type=float,
                    help='epsilon for Adam')
parser.add_argument('--rate', default=0.1, type=float,
                    help='decay rate')
parser.add_argument('--dim', default=64, type=int,
                    help='dimensionality of embeddings')
parser.add_argument('--freeze_BN', action='store_true',
                    help='freeze bn')
parser.add_argument('--la', default=20, type=float,
                    help='lambda')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='gamma')
parser.add_argument('--tau', default=0.2, type=float,
                    help='tau')
parser.add_argument('--margin', default=0.01, type=float,
                    help='margin')
parser.add_argument('-C', default=98, type=int,
                    help='C')
parser.add_argument('-K', default=10, type=int,
                    help='K')
parser.add_argument('--mode', default='train', type=str,
                    choices=['train', 'test'])
parser.add_argument('--train_name', default='train', type=str,
                    help='train dir name')
parser.add_argument('--test_name', default='test', type=str,
                    help='test dir name')
parser.add_argument('--x_name', default=None, type=str,
                    help='extra name to be added')


def save_best_checkpoint(filename, model):
    torch.save(model.state_dict(), 'results/' + filename + '.pth')


def load_best_checkpoint(filename, model):
    model.load_state_dict(torch.load('results/' + filename + '.pth'))
    model = model.cuda()
    return model

def RGB2BGR(im):
    assert im.mode == 'RGB'
    r, g, b = im.split()
    return Image.merge('RGB', (b, g, r))


def main():
    args = parser.parse_args()

    # create model
    # model = net.bninception(args.dim)
    model = timm.create_model('resnet50', pretrained=True, num_classes=args.dim)
    torch.cuda.set_device(args.gpu)
    model = model.cuda()

    # define loss function (criterion) and optimizer
    criterion = loss.SoftTriple(args.la, args.gamma, args.tau, args.margin, args.dim, args.C, args.K).cuda()
    optimizer = torch.optim.Adam([{"params": model.parameters(), "lr": args.modellr},
                                  {"params": criterion.parameters(), "lr": args.centerlr}],
                                 eps=args.eps, weight_decay=args.weight_decay)
    cudnn.benchmark = True

    # load data
    traindir = os.path.join(args.data, args.train_name)
    testdir = os.path.join(args.data, args.test_name)

    normalize = transforms.Normalize(mean=[0.4620, 0.3980, 0.3292],
                                     std=[0.2619, 0.2529, 0.2460])

    # normalize = transforms.Normalize(mean=[104., 117., 128.],
    #                                  std=[1., 1., 1.])


    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.Lambda(RGB2BGR),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Lambda(lambda x: x.mul(255)),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(testdir, transforms.Compose([
            transforms.Lambda(RGB2BGR),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # transforms.Lambda(lambda x: x.mul(255)),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    extra_name = args.x_name
    if extra_name is not None:
        extra_name += '_'
    pretrained_filename = f'{extra_name}pretrained_resnet50_tr-{args.train_name}_ep{args.epochs}-mg{args.margin}-dim{args.dim}-K{args.K}-lambda{args.la}-bs{args.batch_size}_lrm{args.modellr}_lrc{args.centerlr}'
    myloss = np.Inf
    if args.mode == 'train':
        for epoch in range(args.start_epoch, args.epochs):
            print('Training in Epoch[{}]'.format(epoch), f'Loss: {myloss}')
            adjust_learning_rate(optimizer, epoch, args)

            # train for one epoch
            myloss = train(train_loader, model, criterion, optimizer, args)
        save_best_checkpoint(filename=pretrained_filename, model=model)
    else:
        print(f'Not training, loading: {pretrained_filename}.pt')
        model = load_best_checkpoint(filename=pretrained_filename, model=model)

    # evaluate on validation set
    auc, nmi, recall = validate(test_loader, model, args)
    print('Recall@1, 2, 4, 8: {recall[0]:.3f}, {recall[1]:.3f}, {recall[2]:.3f}, {recall[3]:.3f}; NMI: {nmi:.3f}; AUROC: {auc:.3f} \n'
          .format(recall=recall, nmi=nmi, auc=auc))
    results_text = 'Recall@1, 2, 4, 8: {recall[0]:.3f}, {recall[1]:.3f}, {recall[2]:.3f}, {recall[3]:.3f}; NMI: {nmi:.3f}; AUROC: {auc:.3f} \n'\
        .format(recall=recall, nmi=nmi, auc=auc)

    with open(f'./results/res_{pretrained_filename}.txt', 'w') as f:
        f.write(results_text)

def train(train_loader, model, criterion, optimizer, args):
    # switch to train mode
    model.train()
    if args.freeze_BN:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    for i, (input, target) in enumerate(train_loader):
        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(input)
        output = F.normalize(output, p=2, dim=1)
        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.data


def validate(test_loader, model, args):
    # switch to evaluation mode
    model.eval()
    testdata = torch.Tensor()
    testlabel = torch.LongTensor()
    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            output = F.normalize(output, p=2, dim=1)
            testdata = torch.cat((testdata, output.cpu()), 0)
            testlabel = torch.cat((testlabel, target))
    auc, nmi, recall = eva.evaluation(testdata.numpy(), testlabel.numpy(), [1, 2, 4, 8])
    return auc, nmi, recall


def adjust_learning_rate(optimizer, epoch, args):
    # decayed lr by 10 every 20 epochs
    if (epoch + 1) % 20 == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= args.rate


if __name__ == '__main__':
    main()

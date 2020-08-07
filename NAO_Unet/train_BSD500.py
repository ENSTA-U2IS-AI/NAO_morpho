import sys
import glob
import numpy as np
import torch
from utils import utils,metric_F1,dataset
import logging
import argparse
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from model.model import NASUNetBSD
import torchvision.transforms as transforms
import os

parser = argparse.ArgumentParser()

# Basic model parameters.
parser.add_argument('--mode', type=str, default='train',
                    choices=['train', 'test'])
parser.add_argument('--data', type=str, default='data')
parser.add_argument('--dataset', type=str, default='BSD500', choices='BSD500')
parser.add_argument('--autoaugment', action='store_true', default=False)
parser.add_argument('--output_dir', type=str, default='models')
parser.add_argument('--search_space', type=str, default='small', choices=['small', 'middle', 'large'])
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--eval_batch_size', type=int, default=2)
parser.add_argument('--epochs', type=int, default=700)
parser.add_argument('--layers', type=int, default=5)
parser.add_argument('--nodes', type=int, default=5)
parser.add_argument('--channels', type=int, default=16)
parser.add_argument('--cutout_size', type=int, default=None)
parser.add_argument('--grad_bound', type=float, default=5.0)
parser.add_argument('--lr_max', type=float, default=0.025)
parser.add_argument('--lr_min', type=float, default=0)
parser.add_argument('--keep_prob', type=float, default=0.6)
parser.add_argument('--drop_path_keep_prob', type=float, default=0.8)
parser.add_argument('--l2_reg', type=float, default=5e-4)
parser.add_argument('--arch', type=str, default=None)
parser.add_argument('--use_aux_head', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--classes', type=int, default=2)
args = parser.parse_args()

utils.create_exp_dir(args.output_dir, scripts_to_save=glob.glob('*.py'))
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')


def train(train_queue, model, optimizer, global_step, criterion):
    objs = utils.AvgrageMeter()
    F1_score = utils.AvgrageMeter()

    model.train()
    for step, (input, target) in enumerate(train_queue):
        input = input.cuda().requires_grad_()
        target = target.cuda()
    
        optimizer.zero_grad()
        img_predict = model(input)
        global_step += 1
        loss = criterion(img_predict, target.long())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_bound)
        optimizer.step()

        F1_measure = metric_F1.evaluation_F1_measure(img_predict, target)
        n = input.size(0)
        objs.update(loss.data, n)
        F1_score.update(F1_measure, n)

        if (step+1) % 100 == 0:
            logging.info('train %03d loss %e F1_score %f ', step+1, objs.avg, F1_score.avg)

    return F1_score.avg, objs.avg, global_step


def valid(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    F1_score = utils.AvgrageMeter()

    with torch.no_grad():
        model.eval()
        for step, (input, target) in enumerate(valid_queue):
            input = input.cuda()
            target = target.cuda()

            img_predict= model(input)
            loss = criterion(img_predict, target.long())

            F1_measure = metric_F1.evaluation_F1_measure(img_predict, target)
            n = input.size(0)
            objs.update(loss.data, n)
            F1_score.update(F1_measure, n)
        
            if (step+1) % 100 == 0:
                logging.info('valid %03d loss %e f1_score %f ', step+1, objs.avg, F1_score.avg)

    return F1_score.avg, objs.avg


def get_builder(dataset):
    if dataset == 'BSD500':
        return build_BSD_500
    

def build_BSD_500(model_state_dict, optimizer_state_dict, **kwargs):
    epoch = kwargs.pop('epoch')

    data_path = os.getcwd() + "/data/BSR/BSDS500/data/"
    train_data = dataset.BSD_loader(data_path, type='train', transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
    ]))
    valid_data = dataset.BSD_loader(data_path, type='val', transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225])
    ]))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, pin_memory=True, num_workers=16)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.eval_batch_size, pin_memory=True, num_workers=16)


    model = NASUNetBSD(args, args.classes, depth=5, c=args.channels, keep_prob=0.6, nodes=args.nodes,
                       use_aux_head=args.use_aux_head, arch=args.arch)

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
    if model_state_dict is not None:
        model.load_state_dict(model_state_dict)
    
    if torch.cuda.device_count() > 1:
        logging.info("Use %d %s", torch.cuda.device_count(), "GPUs !")
        model = nn.DataParallel(model)
    model = model.cuda()

    train_criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.10, 0.90])).cuda()
    eval_criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.10, 0.90])).cuda()

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr_max,
        momentum=0.9,
        weight_decay=args.l2_reg,
    )
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs), args.lr_min, epoch)
    return train_queue, valid_queue, model, train_criterion, eval_criterion, optimizer, scheduler



def main():
    if not torch.cuda.is_available():
        logging.info('No GPU found!')
        sys.exit(1)
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.enabled = True
    cudnn.benchmark = True
    
    args.steps = int(np.ceil(2000 / args.batch_size)) * args.epochs
    logging.info("Args = %s", args)
    
    _, model_state_dict, epoch, step, optimizer_state_dict, best_f1_score = utils.load(args.output_dir)
    build_fn = get_builder(args.dataset)
    train_queue, valid_queue, model, train_criterion, eval_criterion, optimizer, scheduler = build_fn(model_state_dict, optimizer_state_dict, epoch=epoch-1)

    while epoch < args.epochs:
        scheduler.step()
        logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
        train_acc, train_obj, step = train(train_queue, model, optimizer, step, train_criterion)
        logging.info('train_f1_score %f', train_acc)
        valid_acc, valid_obj = valid(valid_queue, model, eval_criterion)
        logging.info('valid_f1_score %f', valid_acc)
        epoch += 1
        is_best = False
        if valid_acc > best_f1_score:
            best_f1_score = valid_acc
            is_best = True
        if is_best:
          utils.save(args.output_dir, args, model, epoch, step, optimizer, best_f1_score, is_best)
        

if __name__ == '__main__':
    main()

import os
import sys
import glob
import random
import numpy as np
import torch
from utils import utils, dataset, evaluate
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.transforms as transforms
from utils.evaluate import calculate_f_measure
import torch.backends.cudnn as cudnn
from model.model import NASUNetBSD
from search.model_search import NASUNetSegmentationWS
from ops.operations import OPERATIONS_search_with_mor, OPERATIONS_search_without_mor
from controller import NAO

parser = argparse.ArgumentParser(description='NAO Search')

# Basic model parameters.
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
parser.add_argument('--data', type=str, default='data')
parser.add_argument('--dataset', type=str, default='BSD500')
parser.add_argument('--zip_file', action='store_true', default=False)
parser.add_argument('--lazy_load', action='store_true', default=False)
parser.add_argument('--output_dir', type=str, default='models')
parser.add_argument('--search_space', type=str, default='with_mor_ops', choices=['with_mor_ops', 'without_mor_ops'])
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--child_batch_size', type=int, default=4)
parser.add_argument('--child_eval_batch_size', type=int, default=20)
parser.add_argument('--child_epochs', type=int, default=50)  # 60
parser.add_argument('--child_layers', type=int, default=2)
parser.add_argument('--child_nodes', type=int, default=5)
parser.add_argument('--child_channels', type=int, default=8)
parser.add_argument('--child_cutout_size', type=int, default=None)
parser.add_argument('--child_grad_bound', type=float, default=5.0)
parser.add_argument('--child_lr_max', type=float, default=0.025)
parser.add_argument('--child_lr_min', type=float, default=0.001)
parser.add_argument('--child_keep_prob', type=float, default=0.8)
parser.add_argument('--child_drop_path_keep_prob', type=float, default=None)
parser.add_argument('--child_l2_reg', type=float, default=5e-4)
parser.add_argument('--child_use_aux_head', action='store_true', default=True)
parser.add_argument('--child_arch_pool', type=str, default=None)
parser.add_argument('--child_lr', type=float, default=0.1)
parser.add_argument('--child_double_down_channel', type=bool, default=True)

parser.add_argument('--child_label_smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('--child_gamma', type=float, default=0.97, help='learning rate decay')
parser.add_argument('--child_decay_period', type=int, default=1, help='epochs between two learning rate decays')
parser.add_argument('--controller_seed_arch', type=int, default=300)
parser.add_argument('--controller_expand', type=int, default=None)
parser.add_argument('--controller_new_arch', type=int, default=100)
parser.add_argument('--controller_encoder_layers', type=int, default=1)
parser.add_argument('--controller_encoder_hidden_size', type=int, default=64)
parser.add_argument('--controller_encoder_emb_size', type=int, default=32)
parser.add_argument('--controller_mlp_layers', type=int, default=0)
parser.add_argument('--controller_mlp_hidden_size', type=int, default=200)
parser.add_argument('--controller_decoder_layers', type=int, default=1)
parser.add_argument('--controller_decoder_hidden_size', type=int, default=64)
parser.add_argument('--controller_source_length', type=int, default=40)
parser.add_argument('--controller_encoder_length', type=int, default=20)
parser.add_argument('--controller_decoder_length', type=int, default=40)
parser.add_argument('--controller_encoder_dropout', type=float, default=0)
parser.add_argument('--controller_mlp_dropout', type=float, default=0.1)
parser.add_argument('--controller_decoder_dropout', type=float, default=0)
parser.add_argument('--controller_l2_reg', type=float, default=0)
parser.add_argument('--controller_encoder_vocab_size', type=int, default=12)
parser.add_argument('--controller_decoder_vocab_size', type=int, default=12)
parser.add_argument('--controller_trade_off', type=float, default=0.8)
parser.add_argument('--controller_epochs', type=int, default=1000)
parser.add_argument('--controller_batch_size', type=int, default=100)
parser.add_argument('--controller_lr', type=float, default=0.001)
parser.add_argument('--controller_optimizer', type=str, default='adam')
parser.add_argument('--controller_grad_bound', type=float, default=5.0)
args = parser.parse_args()

utils.create_exp_dir(args.output_dir, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')


class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss

def cross_entropy_loss(prediction, label):
    #ref:https://github.com/mayorx/rcf-edge-detection
    label = label.long()
    mask = label.float()
    num_positive = torch.sum((mask == 1.).float()).float()
    num_negative = torch.sum((mask == 0.).float()).float()
    #print(mask)

    mask[mask == 1] = 1.0 * num_negative / (num_positive + num_negative)
    mask[mask == 0] = 1.1 * num_positive / (num_positive + num_negative)
    mask[mask == 2] = 0

    # print('num pos', num_positive)
    # print('num neg', num_negative)
    #print(1.0 * num_negative / (num_positive + num_negative), 1.1 * num_positive / (num_positive + num_negative))
    cost = torch.nn.functional.binary_cross_entropy(
        prediction.float(), label.float(), weight=mask, reduction='none')
    # print(torch.sum(cost) / (num_negative + num_positive))
    return torch.sum(cost) / (num_negative + num_positive)

def get_builder(dataset):
    if dataset == 'BSD500':
        return build_BSD_500

def build_BSD_500(model_state_dict=None, optimizer_state_dict=None, **kwargs):
    epoch = kwargs.pop('epoch')
    ratio = kwargs.pop('ratio')
    data_path = os.getcwd() + "/data/BSR/BSDS500/data/"
    train_data = dataset.BSD_loader(root=data_path, split='train', random_crop=True, random_flip=False,
                                    normalisation=False)
    valid_data = dataset.BSD_loader(root=data_path, split='val', random_crop=False, random_flip=False,
                                    normalisation=False)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.child_batch_size, pin_memory=True, num_workers=16, shuffle=True)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.child_eval_batch_size, pin_memory=True, num_workers=16, shuffle=True)

    model = NASUNetSegmentationWS(args, depth=args.child_layers, classes=args.num_class, nodes=args.child_nodes,
                                  chs=args.child_channels,
                                  keep_prob=args.child_keep_prob,
                                  use_aux_head=args.child_use_aux_head,
                                  double_down_channel=args.child_double_down_channel)
    model = model.cuda()

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.child_lr_max,
        momentum=0.9,
        weight_decay=args.child_l2_reg,
    )
    if model_state_dict is not None:
        model.load_state_dict(model_state_dict)
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.child_epochs, args.child_lr_min, epoch)
    return train_queue, valid_queue, model, optimizer, scheduler


def get_scheduler(optimizer, dataset):
    if 'BSD500' in dataset:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.child_epochs, args.child_lr_min)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.child_decay_period, gamma=args.child_gamma)
    return scheduler


def child_train(train_queue, model, optimizer, global_step, arch_pool, arch_pool_prob, criterion=None):
    objs = utils.AvgrageMeter()
    OIS = utils.AvgrageMeter()
    OIS.reset()

    # set the mode of model to train
    model.train()

    for step, (input, target) in enumerate(train_queue):
        input = input.cuda().requires_grad_()
        target = target.cuda()

        arch = utils.sample_arch(arch_pool, arch_pool_prob)
        img_predict = model(input, arch)
        if criterion==None:
            loss = cross_entropy_loss(img_predict, target)
        else:
            loss = cross_entropy_loss(img_predict, target.long())

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.child_grad_bound)
        optimizer.step()

        ois_ = evaluate.evaluation_OIS(img_predict, target)

        n = input.size(0)
        objs.update(loss.data, n)
        OIS.update(ois_, 1)
        if (step + 1) % 50 == 0:
            logging.info('Train %03d loss %e OIS %f ', step + 1, objs.avg, OIS.avg)
            logging.info('Arch: %s', ' '.join(map(str, arch[0] + arch[1])))
        global_step += 1

    return OIS.avg, objs.avg, global_step


def child_valid(valid_queue, model, arch_pool, criterion=None):
    valid_acc_list = []

    # set the mode of model to eval
    model.eval()

    with torch.no_grad():
        for i, arch in enumerate(arch_pool):
            # for (inputs, targets) in valid_queue:
            inputs, targets = next(iter(valid_queue))
            inputs = inputs.cuda()
            targets = targets.cuda()

            img_val_predict = model(inputs, arch, bn_train=True)
            if criterion == None:
                loss = cross_entropy_loss(img_val_predict, targets)
            else:
                loss = cross_entropy_loss(img_val_predict, targets.long())

            ois_ = evaluate.evaluation_OIS(img_val_predict, targets)

            valid_acc_list.append(ois_)
            if (i + 1) % 5 == 0:
                logging.info('Valid arch %s\n loss %.2f OIS %f', ' '.join(map(str, arch[0] + arch[1])), loss, ois_)

    return valid_acc_list


def train_and_evaluate_top_on_BSD500(archs, train_queue, valid_queue):
    res = []

    objs = utils.AvgrageMeter()
    OIS = utils.AvgrageMeter()
    for i, arch in enumerate(archs):
        objs.reset()
        OIS.reset()
        logging.info('Train and evaluate the {} arch'.format(i + 1))
        model = NASUNetBSD(args, args.num_class, depth=args.child_layers, c=args.child_channels,
                           keep_prob=args.child_keep_prob, nodes=args.child_nodes,
                           use_aux_head=args.child_use_aux_head, arch=arch,
                           double_down_channel=args.child_double_down_channel)
        model = model.cuda()
        model.train()
        optimizer = torch.optim.SGD(
            model.parameters(),
            args.child_lr_max,
            momentum=0.9,
            weight_decay=args.child_l2_reg,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10, args.child_lr_min)
        global_step = 0
        for e in range(10):
            scheduler.step()
            # set the mode of model to train
            model.train()

            for step, (input, target) in enumerate(train_queue):
                input = input.cuda().requires_grad_()
                target = target.cuda()

                # sample an arch to train
                logits = model(input)
                loss = cross_entropy_loss(logits, target.long())

                optimizer.zero_grad()
                global_step += 1
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.child_grad_bound)
                optimizer.step()

                ois_ = evaluate.evaluation_OIS(logits, target)
                n = input.size(0)
                objs.update(loss.data, n)
                OIS.update(ois_, n)

                if (step + 1) % 50 == 0:
                    logging.info('Train epoch %03d %03d loss %e OIS %f', e + 1, step + 1, objs.avg, OIS.avg)
        objs.reset()
        OIS.reset()
        # set the mode of model to eval
        model.eval()

        with torch.no_grad():
            for step, (input, target) in enumerate(valid_queue):
                input = input.cuda()
                target = target.cuda()

                logits = model(input)
                loss = cross_entropy_loss(logits, target.long())

                ois_ = evaluate.evaluation_OIS(logits, target)
                n = input.size(0)
                objs.update(loss.data, n)
                OIS.update(ois_, n)

                if (step + 1) % 5 == 0:
                    logging.info('valid %03d loss %e OIS %f ', step + 1, objs.avg, OIS.avg)
        res.append(OIS.avg)
    return res


def nao_train(train_queue, model, optimizer):
    objs = utils.AvgrageMeter()
    mse = utils.AvgrageMeter()
    nll = utils.AvgrageMeter()
    model.train()
    for step, sample in enumerate(train_queue):
        encoder_input = sample['encoder_input']
        encoder_target = sample['encoder_target']
        decoder_input = sample['decoder_input']
        decoder_target = sample['decoder_target']

        encoder_input = encoder_input.cuda()
        encoder_target = encoder_target.cuda().requires_grad_()
        decoder_input = decoder_input.cuda()
        decoder_target = decoder_target.cuda()

        optimizer.zero_grad()
        predict_value, log_prob, arch = model(encoder_input, decoder_input)
        loss_1 = F.mse_loss(predict_value.squeeze(), encoder_target.squeeze())
        loss_2 = F.nll_loss(log_prob.contiguous().view(-1, log_prob.size(-1)), decoder_target.view(-1))
        loss = args.controller_trade_off * loss_1 + (1 - args.controller_trade_off) * loss_2
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.controller_grad_bound)
        optimizer.step()

        n = encoder_input.size(0)
        objs.update(loss.data, n)
        mse.update(loss_1.data, n)
        nll.update(loss_2.data, n)

    return objs.avg, mse.avg, nll.avg


def nao_valid(queue, model):
    inputs = []
    targets = []
    predictions = []
    archs = []
    with torch.no_grad():
        model.eval()
        for step, sample in enumerate(queue):
            encoder_input = sample['encoder_input']
            encoder_target = sample['encoder_target']
            decoder_target = sample['decoder_target']

            encoder_input = encoder_input.cuda()
            encoder_target = encoder_target.cuda()
            decoder_target = decoder_target.cuda()

            predict_value, logits, arch = model(encoder_input)
            n = encoder_input.size(0)
            inputs += encoder_input.data.squeeze().tolist()
            targets += encoder_target.data.squeeze().tolist()
            predictions += predict_value.data.squeeze().tolist()
            archs += arch.data.squeeze().tolist()
    pa = utils.pairwise_accuracy(targets, predictions)
    hd = utils.hamming_distance(inputs, archs)
    return pa, hd


def nao_infer(queue, model, step, direction='+'):
    new_arch_list = []
    model.eval()
    for i, sample in enumerate(queue):
        encoder_input = sample['encoder_input']
        encoder_input = encoder_input.cuda()
        model.zero_grad()
        new_arch = model.generate_new_arch(encoder_input, step, direction=direction)
        new_arch_list.extend(new_arch.data.squeeze().tolist())
    return new_arch_list


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.enabled = True
    cudnn.benchmark = True
    cudnn.deterministic = True

    if args.dataset == 'BSD500':
        args.num_class = 1
    else:
        args.num_class = None

    if args.search_space == 'with_mor_ops':
        OPERATIONS = OPERATIONS_search_with_mor
    elif args.search_space == 'without_mor_ops':
        OPERATIONS = OPERATIONS_search_without_mor

    args.child_num_ops = len(OPERATIONS)
    args.controller_encoder_vocab_size = 1 + (args.child_nodes + 2 - 1) + args.child_num_ops
    args.controller_decoder_vocab_size = args.controller_encoder_vocab_size
    args.steps = int(np.ceil(4000 / args.child_batch_size)) * args.child_epochs

    logging.info("args = %s", args)

    # load search archs
    if args.child_arch_pool is not None:
        logging.info('Architecture pool is provided, loading')
        with open(args.child_arch_pool) as f:
            archs = f.read().splitlines()
            archs = list(map(utils.build_dag, archs))
            child_arch_pool = archs
    elif os.path.exists(os.path.join(args.output_dir, 'arch_pool')):
        logging.info('Architecture pool is founded, loading')
        with open(os.path.join(args.output_dir, 'arch_pool')) as f:
            archs = f.read().splitlines()
            archs = list(map(utils.build_dag, archs))
            child_arch_pool = archs
            logging.info(child_arch_pool)
    else:
        child_arch_pool = None

    # load network model
    build_fn = get_builder(args.dataset)
    train_queue, valid_queue, model, optimizer, scheduler,= build_fn(ratio=0.9,
                                                                    epoch=-1)

    # initial NAO algorithm model
    nao = NAO(
        args.controller_encoder_layers,
        args.controller_encoder_vocab_size,
        args.controller_encoder_hidden_size,
        args.controller_encoder_dropout,
        args.controller_encoder_length,
        args.controller_source_length,
        args.controller_encoder_emb_size,
        args.controller_mlp_layers,
        args.controller_mlp_hidden_size,
        args.controller_mlp_dropout,
        args.controller_decoder_layers,
        args.controller_decoder_vocab_size,
        args.controller_decoder_hidden_size,
        args.controller_decoder_dropout,
        args.controller_decoder_length,
    )
    nao = nao.cuda()
    logging.info("Encoder-Predictor-Decoder param size = %fMB", utils.count_parameters_in_MB(nao))

    if child_arch_pool is None:
        logging.info('Architecture pool is not provided, randomly generating now')
        child_arch_pool = utils.generate_arch(args.controller_seed_arch, args.child_nodes, args.child_num_ops)  # [[[downc],[upc]]]
    arch_pool = []
    arch_pool_valid_acc = []
    for i in range(4):
        logging.info('Iteration %d', i)

        child_arch_pool_prob = []
        for arch in child_arch_pool:
            tmp_model = NASUNetBSD(args, args.num_class, depth=args.child_layers, c=args.child_channels,
                                   keep_prob=args.child_keep_prob, nodes=args.child_nodes,
                                   use_aux_head=args.child_use_aux_head, arch=arch,
                                   double_down_channel=args.child_double_down_channel)
            child_arch_pool_prob.append(utils.count_parameters_in_MB(tmp_model))
            del tmp_model

        step = 0
        scheduler = get_scheduler(optimizer, args.dataset)
        for epoch in range(1, args.child_epochs + 1):
            lr = scheduler.get_last_lr()[0]
            logging.info('epoch %d lr %e', epoch, lr)
            # Randomly sample an example to train
            train_acc, train_obj, step = child_train(train_queue, model, optimizer, step, child_arch_pool,
                                                     child_arch_pool_prob)
            scheduler.step()
            logging.info('train_OIS %f', train_acc)

        logging.info("Evaluate seed archs")
        arch_pool += child_arch_pool
        arch_pool_valid_acc = child_valid(valid_queue, model, arch_pool)

        arch_pool_valid_acc_sorted_indices = np.argsort(arch_pool_valid_acc)[::-1]
        arch_pool = [arch_pool[i] for i in arch_pool_valid_acc_sorted_indices]
        arch_pool_valid_acc = [arch_pool_valid_acc[i] for i in arch_pool_valid_acc_sorted_indices]
        with open(os.path.join(args.output_dir, 'arch_pool.{}'.format(i)), 'w') as fa:
            with open(os.path.join(args.output_dir, 'arch_pool.perf.{}'.format(i)), 'w') as fp:
                for arch, perf in zip(arch_pool, arch_pool_valid_acc):
                    arch = ' '.join(map(str, arch[0] + arch[1]))
                    fa.write('{}\n'.format(arch))
                    fp.write('{}\n'.format(perf))
        if i == 3:
            break

        # Train Encoder-Predictor-Decoder
        logging.info('Train Encoder-Predictor-Decoder')
        encoder_input = list(map(lambda x: utils.parse_arch_to_seq(x[0]) + utils.parse_arch_to_seq(x[1]), arch_pool))
        # [[downc, upc]]
        print(encoder_input)
        min_val = min(arch_pool_valid_acc)
        max_val = max(arch_pool_valid_acc)
        encoder_target = [(i - min_val) / (max_val - min_val) for i in arch_pool_valid_acc]

        if args.controller_expand:
            dataset = list(zip(encoder_input, encoder_target))
            n = len(dataset)
            ratio = 0.9
            split = int(n * ratio)
            np.random.shuffle(dataset)
            encoder_input, encoder_target = list(zip(*dataset))
            train_encoder_input = list(encoder_input[:split])
            train_encoder_target = list(encoder_target[:split])
            valid_encoder_input = list(encoder_input[split:])
            valid_encoder_target = list(encoder_target[split:])
            for _ in range(args.controller_expand - 1):
                for src, tgt in zip(encoder_input[:split], encoder_target[:split]):
                    a = np.random.randint(0, args.child_nodes)
                    b = np.random.randint(0, args.child_nodes)
                    src = src[:4 * a] + src[4 * a + 2:4 * a + 4] + \
                          src[4 * a:4 * a + 2] + src[4 * (a + 1):20 + 4 * b] + \
                          src[20 + 4 * b + 2:20 + 4 * b + 4] + src[20 + 4 * b:20 + 4 * b + 2] + \
                          src[20 + 4 * (b + 1):]
                    train_encoder_input.append(src)
                    train_encoder_target.append(tgt)
        else:
            train_encoder_input = encoder_input
            train_encoder_target = encoder_target
            valid_encoder_input = encoder_input
            valid_encoder_target = encoder_target
        logging.info('Train data: {}\tValid data: {}'.format(len(train_encoder_input), len(valid_encoder_input)))

        nao_train_dataset = utils.NAODataset(train_encoder_input, train_encoder_target, True,
                                             swap=True if args.controller_expand is None else False)
        nao_valid_dataset = utils.NAODataset(valid_encoder_input, valid_encoder_target, False)
        nao_train_queue = torch.utils.data.DataLoader(
            nao_train_dataset, batch_size=args.controller_batch_size, shuffle=True, pin_memory=True)
        nao_valid_queue = torch.utils.data.DataLoader(
            nao_valid_dataset, batch_size=args.controller_batch_size, shuffle=False, pin_memory=True)
        nao_optimizer = torch.optim.Adam(nao.parameters(), lr=args.controller_lr, weight_decay=args.controller_l2_reg)
        for nao_epoch in range(1, args.controller_epochs + 1):
            nao_loss, nao_mse, nao_ce = nao_train(nao_train_queue, nao, nao_optimizer)
            logging.info("epoch %04d train loss %.6f mse %.6f ce %.6f", nao_epoch, nao_loss, nao_mse, nao_ce)
            if nao_epoch % 100 == 0:
                pa, hs = nao_valid(nao_valid_queue, nao)
                logging.info("Evaluation on valid data")
                logging.info('epoch %04d pairwise accuracy %.6f hamming distance %.6f', nao_epoch, pa, hs)

        # Generate new archs
        new_archs = []
        max_step_size = 50
        predict_step_size = 0
        top30_archs = list(map(lambda x: utils.parse_arch_to_seq(x[0]) + utils.parse_arch_to_seq(x[1]), arch_pool[:30]))
        nao_infer_dataset = utils.NAODataset(top30_archs, None, False)
        nao_infer_queue = torch.utils.data.DataLoader(
            nao_infer_dataset, batch_size=len(nao_infer_dataset), shuffle=False, pin_memory=True)
        while len(new_archs) < args.controller_new_arch:
            predict_step_size += 1
            logging.info('Generate new architectures with step size %d', predict_step_size)
            new_arch = nao_infer(nao_infer_queue, nao, predict_step_size, direction='+')
            for arch in new_arch:
                if arch not in encoder_input and arch not in new_archs:
                    new_archs.append(arch)
                if len(new_archs) >= args.controller_new_arch:
                    break
            logging.info('%d new archs generated now', len(new_archs))
            if predict_step_size > max_step_size:
                break

        child_arch_pool = list(map(lambda x: utils.parse_seq_to_arch(x), new_archs))  # [[[conv],[reduc]]]
        logging.info("Generate %d new archs", len(child_arch_pool))

    logging.info('Finish Searching')
    logging.info('Reranking top 5 architectures')
    # reranking top 5
    top_archs = arch_pool[:5]
    print(top_archs)
    top_archs_perf = train_and_evaluate_top_on_BSD500(top_archs, train_queue, valid_queue)
    top_archs_sorted_indices = np.argsort(top_archs_perf)[::-1]
    top_archs = [top_archs[i] for i in top_archs_sorted_indices]
    top_archs_perf = [top_archs_perf[i] for i in top_archs_sorted_indices]
    with open(os.path.join(args.output_dir, 'arch_pool.final'), 'w') as fa:
        with open(os.path.join(args.output_dir, 'arch_pool.perf.final'), 'w') as fp:
            for arch, perf in zip(top_archs, top_archs_perf):
                arch = ' '.join(map(str, arch[0] + arch[1]))
                fa.write('{}\n'.format(arch))
                fp.write('{}\n'.format(perf))


if __name__ == '__main__':
    main()

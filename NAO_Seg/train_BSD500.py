import sys
import glob
import numpy as np
import torch
from utils import utils, evaluate, dataset
import logging
import argparse
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from model.model import NASUNetBSD
from search.model_search import NASUNetSegmentationWS
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
parser.add_argument('--search_space', type=str, default='with_mor_ops', choices=['with_mor_ops', 'without_mor_ops'])
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--eval_batch_size', type=int, default=4)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--layers', type=int, default=4)
parser.add_argument('--nodes', type=int, default=5)
parser.add_argument('--channels', type=int, default=16)  # 64
parser.add_argument('--cutout_size', type=int, default=None)
parser.add_argument('--grad_bound', type=float, default=5.0)
parser.add_argument('--lr_max', type=float, default=1e-1)
parser.add_argument('--lr_min', type=float, default=1e-5)
parser.add_argument('--keep_prob', type=float, default=0.5)
parser.add_argument('--drop_path_keep_prob', type=float, default=None)
parser.add_argument('--l2_reg', type=float, default=5e-4)
parser.add_argument('--arch', type=str, default=None)
parser.add_argument('--use_aux_head', action='store_true', default=True)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--classes', type=int, default=2)
parser.add_argument('--save', type=bool, default=True)
args = parser.parse_args()

utils.create_exp_dir(args.output_dir, scripts_to_save=glob.glob('*.py'))
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')


def train(train_queue, model, optimizer, global_step, criterion):
    objs = utils.AvgrageMeter()
    OIS = utils.AvgrageMeter()

    # set the mode of model to train
    model.train()

    for step, (input, target) in enumerate(train_queue):
        input = input.cuda().requires_grad_()
        target = target.cuda()

        img_predict = model(input)
        loss = criterion(img_predict, target.squeeze(1).long())

        optimizer.zero_grad()
        global_step += 1
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_bound)
        optimizer.step()

        ois = evaluate.evaluation_OIS(img_predict, target)
        ods = evaluate.evaluation_ODS(img_predict, target)
        n = input.size(0)
        objs.update(loss.data, n)
        OIS.update(ois, n)

        if (step + 1) % 25 == 0:
            logging.info('train %03d loss %e OIS %f ', step + 1, objs.avg, OIS.avg)

    return OIS.avg, objs.avg, global_step


def valid(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()

    # set the mode of model to eval
    model.eval()
    imgs_predict = []
    imgs_gt = []
    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = input.cuda()
            target = target.cuda()

            img_predict = model(input)
            loss = criterion(img_predict, target.squeeze(1).long())

            img_predict = torch.nn.functional.softmax(img_predict, 1)
            ## with channel=1 we get the img[B,H,W]
            img_predict = img_predict[:, 1]
            img_predict = img_predict.cpu().detach().numpy().astype('float32')
            img_GT = target.cpu().detach().numpy().astype(np.bool)
            imgs_predict.append(img_predict)
            imgs_gt.append((img_GT))

            n = input.size(0)
            objs.update(loss.data, n)

            if (step + 1) % 25 == 0:
                logging.info('valid %03d loss %e ', step + 1, objs.avg)

        # --calculate the ODS
        imgs_predict = np.concatenate(imgs_predict, axis=0)
        imgs_gt = np.concatenate(imgs_gt, axis=0)
        thresholds = np.linspace(0, 1, 100)
        f1_score_sum = 0.
        ODS_th = []
        for th in thresholds:
            f1_scores = []
            for i in range(imgs_predict.shape[0]):
                edge = np.where(imgs_predict[i] >= th, 1, 0).astype(np.bool)
                f1_scores.append(evaluate.calculate_f1_score(edge, imgs_gt[i]))
            f1_score_sum = np.sum(np.array(f1_scores))
            ODS_th.append(f1_score_sum)
        ODS = np.argmax(np.array(ODS_th)) / 100

        logging.info('valid ODS %f ', ODS)
    return ODS, objs.avg


def test(test_queue, model, criterion):
    objs = utils.AvgrageMeter()

    # set the mode of model to eval
    model.eval()

    imgs_predict = []
    imgs_gt = []
    with torch.no_grad():
        for step, (input, target) in enumerate(test_queue):
            input = input.cuda()
            target = target.cuda()

            img_predict = model(input)
            loss = criterion(img_predict, target.squeeze(1).long())

            img_predict = torch.nn.functional.softmax(img_predict, 1)
            ## with channel=1 we get the img[B,H,W]
            img_predict = img_predict[:, 1]
            img_predict = img_predict.cpu().detach().numpy().astype('float32')
            img_GT = target.cpu().detach().numpy().astype(np.bool)
            imgs_predict.append(img_predict)
            imgs_gt.append((img_GT))

            n = input.size(0)
            objs.update(loss.data, n)

            if (step + 1) % 20 == 0:
                logging.info('test  loss %e ', objs.avg)

        logging.info("begin to calculate the OIS and ODS")
        imgs_predict = np.concatenate(imgs_predict, axis=0)
        imgs_gt = np.concatenate(imgs_gt, axis=0)

        thresholds = np.linspace(0, 1, 100)
        # ---calculate the OIS
        OIS_th = 0.
        for i in range(imgs_predict.shape[0]):
            f1_scores = []
            for th in thresholds:
                edge = np.where(imgs_predict[i] >= th, 1, 0).astype(np.bool)
                f1_scores.append(evaluate.calculate_f1_score(edge, imgs_gt[i]))
            OIS_th += np.argmax(np.array(f1_scores)) / 100
        OIS = OIS_th / imgs_predict.shape[0]

        # --calculate the ODS
        f1_score_sum = 0.
        ODS_th = []
        for th in thresholds:
            f1_scores = []
            for i in range(imgs_predict.shape[0]):
                edge = np.where(imgs_predict[i] >= th, 1, 0).astype(np.bool)
                f1_scores.append(evaluate.calculate_f1_score(edge, imgs_gt[i]))
            f1_score_sum = np.sum(np.array(f1_scores))
            ODS_th.append(f1_score_sum)
        ODS = np.argmax(np.array(ODS_th)) / 100

        print("OIS: %f ODS: %f", OIS, ODS)


def save_pre_imgs(test_queue, model):
    from PIL import Image
    import scipy.io as io

    folder = './results/'
    predict_folder = os.path.join(folder, 'predict')
    gt_folder = os.path.join(folder, 'groundTruth')
    try:
        os.makedirs(predict_folder)
        os.makedirs(gt_folder)
        os.makedirs(os.path.join(predict_folder, 'png'))
        os.makedirs(os.path.join(predict_folder, 'mat'))
        os.makedirs(os.path.join(gt_folder, 'mat'))
        os.makedirs(os.path.join(gt_folder, 'png'))
    except Exception:
        print('dir already exist....')
        pass
        # set the mode of model to eval
        model.eval()

    imgs_predict = []
    imgs_gt = []
    with torch.no_grad():
        for step, (input, target) in enumerate(test_queue):
            # print("dsaldhal")
            input = input.cuda()
            target = target.cuda()

            img_predict = model(input)

            img_predict = torch.nn.functional.softmax(img_predict, 1)
            ## with channel=1 we get the img[B,H,W]
            img_predict = img_predict[:, 1]
            img_predict = img_predict.cpu().detach().numpy().astype('float32')
            img_GT = target.cpu().detach().numpy().astype(np.uint8)
            img_predict = img_predict.squeeze()
            img_GT = img_GT.squeeze()
            # print(img_predict.shape)
            imgs_predict.append(img_predict)
            imgs_gt.append((img_GT))

            # ---save the image
            mat_predict = img_predict
            img_predict *= 255
            img_predict = Image.fromarray(np.uint8(img_predict))
            img_predict = img_predict.convert('L')  # single channel
            img_predict.save(os.path.join(predict_folder, 'png', '{}.png'.format(step)))
            io.savemat(os.path.join(predict_folder, 'mat', '{}.mat'.format(step)), {'predict': mat_predict})

            mat_gt = img_GT
            img_GT *= 255
            img_GT = Image.fromarray(np.uint8(img_GT))
            img_GT = img_GT.convert('L')
            img_GT.save(os.path.join(gt_folder, 'png', '{}.png'.format(step)))
            io.savemat(os.path.join(gt_folder, 'mat', '{}.mat'.format(step)), {'gt': mat_gt})

    print("save is finished")


def get_builder(dataset):
    if dataset == 'BSD500':
        return build_BSD_500


def build_BSD_500(model_state_dict, optimizer_state_dict, **kwargs):
    epoch = kwargs.pop('epoch')

    data_path = os.getcwd() + "/data/BSR/BSDS500/data/"
    train_data = dataset.BSD_loader(data_path, type='train', random_crop=True, random_flip=True)
    valid_data = dataset.BSD_loader(data_path, type='val', random_crop=False, random_flip=False)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, pin_memory=True, num_workers=16, shuffle=True)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.eval_batch_size, pin_memory=True, num_workers=16, shuffle=False)

    model = NASUNetBSD(args, args.classes, depth=args.layers, c=args.channels, nodes=args.nodes,
                       use_aux_head=args.use_aux_head, arch=args.arch, use_softmax_head=False,
                       double_down_channel=False)

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
    if model_state_dict is not None:
        model.load_state_dict(model_state_dict)

    if torch.cuda.device_count() > 1:
        logging.info("Use %d %s", torch.cuda.device_count(), "GPUs !")
        model = nn.DataParallel(model)
    model = model.cuda()

    train_criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.065, 0.935])).cuda()
    eval_criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.065, 0.935])).cuda()

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr_max,
        momentum=0.9,
        weight_decay=args.l2_reg,
    )
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs), args.lr_min)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs), args.lr_min, epoch)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=3)
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

    args.steps = int(np.ceil(4000 / args.batch_size)) * args.epochs
    logging.info("Args = %s", args)
    output_dir = './exp/NAONet_BSD_500/'
    _, model_state_dict, epoch, step, optimizer_state_dict, best_OIS = utils.load(output_dir)
    build_fn = get_builder(args.dataset)
    train_queue, valid_queue, model, train_criterion, eval_criterion, optimizer, scheduler = build_fn(model_state_dict,
                                                                                                      optimizer_state_dict,
                                                                                                      epoch=epoch - 1)

    filename = "./curve/accuracy_loss_validation.txt"  # --for draw save the loss and ods of valid set
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except:
            logging.info('creat the curve folder failed.')
    valid_pre_ODS = 0
    while epoch < args.epochs:
        # logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
        logging.info('epoch %d lr %e', epoch, optimizer.param_groups[0]['lr'])
        train_OIS, train_obj, step = train(train_queue, model, optimizer, step, train_criterion)
        logging.info('train_OIS %f', train_OIS)
        valid_ODS, valid_obj = valid(valid_queue, model, eval_criterion)
        scheduler.step(valid_ODS)
        is_best = False
        if valid_ODS > valid_pre_ODS:
            valid_pre_ODS = valid_ODS
            best_OIS = train_OIS
            is_best = True
        if is_best:
            logging.info('the current best model is model %d', epoch)
            utils.save(args.output_dir, args, model, epoch, step, optimizer, best_OIS, is_best)
        epoch += 1
        # draw the curve
        with open(filename, 'a+')as f:
            f.write(str(valid_obj.cpu().numpy()))
            f.write(',')
            f.write(str(valid_ODS))
            f.write('\n')

    logging.info('train is finished!')
    try:
        loss = []
        accuracy_ODS = []
        with open(filename, 'r') as f:
            for line in f:
                loss.append(eval(line.split(',')[0]))
                accuracy_ODS.append(eval(line.split(',')[1]))

        evaluate.accuracyandlossCurve(loss, accuracy_ODS, args.epochs)
    except:
        logging.info('the plot of valid set is failed')
        pass

    data_path = os.getcwd() + "/data/BSR/BSDS500/data/"
    test_data = dataset.BSD_loader(data_path, type='test', random_crop=False, random_flip=False)
    test_queue = torch.utils.data.DataLoader(test_data, batch_size=1, pin_memory=True, num_workers=16, shuffle=False)

    logging.info('loading the best model.')
    output_dir = './exp/NAONet_BSD_500/'
    _, model_state_dict, epoch, step, optimizer_state_dict, best_OIS = utils.load(output_dir)
    build_fn = get_builder(args.dataset)
    train_queue, valid_queue, model, train_criterion, eval_criterion, optimizer, scheduler = build_fn(model_state_dict,
                                                                                                      optimizer_state_dict,
                                                                                                      epoch=epoch - 1)
    test(test_queue, model, eval_criterion)
    logging.info('test is finished!')
    if (args.save == True):
        try:
            save_pre_imgs(test_queue, model)
            logging.info('save is finished!')
        except:
            logging.info('save is failed!')


if __name__ == '__main__':
    main()

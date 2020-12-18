import sys
import glob
import numpy as np
import torch
from utils import utils, evaluate, dataloader_BSD_aux,dataset
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
from model.model import NASUNetBSD
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
parser.add_argument('--batch_size', type=int, default=2)#8
parser.add_argument('--eval_batch_size', type=int, default=1)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--layers', type=int, default=5)
parser.add_argument('--nodes', type=int, default=5)
parser.add_argument('--channels', type=int, default=16)  # 64
parser.add_argument('--cutout_size', type=int, default=None)
parser.add_argument('--grad_bound', type=float, default=5.0)
parser.add_argument('--lr_max', type=float, default=1e-1)
parser.add_argument('--lr_min', type=float, default=1e-3)
parser.add_argument('--keep_prob', type=float, default=0.8)
parser.add_argument('--drop_path_keep_prob', type=float, default=None)
parser.add_argument('--l2_reg', type=float, default=5e-4)
parser.add_argument('--arch', type=str, default=None)
parser.add_argument('--use_aux_head', action='store_true', default=True)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--classes', type=int, default=2)
parser.add_argument('--save', type=bool, default=True)
parser.add_argument('--iterations', type=int, default=20000)
parser.add_argument('--val_per_iter', type=int, default=4000)
parser.add_argument('--lr_schedule_power', type=float, default=0.9)
parser.add_argument('--double_down_channel', type=bool, default=True)
args = parser.parse_args()

utils.create_exp_dir(args.output_dir, scripts_to_save=glob.glob('*.py'))
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')


def train(train_queue, model, optimizer,criterion):
    objs = utils.AvgrageMeter()
    OIS = utils.AvgrageMeter()

    # set the mode of model to train
    model.train()

    for step, (input, target) in enumerate(train_queue):
        input = input.cuda().requires_grad_()
        target = target.cuda()

        img_predict = model(input)
        loss = criterion(img_predict, target.long())

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_bound)
        optimizer.step()

        ois = evaluate.evaluation_OIS(img_predict, target)
        n = input.size(0)
        objs.update(loss.data, n)
        OIS.update(ois, n)

        if (step + 1) % 50 == 0:
            logging.info('train %03d loss %e OIS %f ', step + 1, objs.avg, OIS.avg)

    return OIS.avg, objs.avg

def valid(valid_queue, model,criterion):
    objs = utils.AvgrageMeter()
    OIS = utils.AvgrageMeter()

    # set the mode of model to eval
    model.eval()
    # imgs_predict = []
    # imgs_gt = []
    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = input.cuda()
            target = target.cuda()

            img_predict = model(input)
            loss = criterion(img_predict, target.long())

            ois = evaluate.evaluation_OIS(img_predict, target)
            n = input.size(0)
            objs.update(loss.data, n)
            OIS.update(ois, n)

            if (step + 1) % 100 == 0:
                logging.info('valid %03d loss %e OIS %f', step + 1, objs.avg, OIS.avg)

    return OIS.avg, objs.avg


def test(test_queue, model,criterion):
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
            loss = criterion(img_predict, target.long())

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

        thresholds = np.linspace(0, 1, 1000)
        # ---calculate the OIS
        OIS_th = 0.
        for i in range(imgs_predict.shape[0]):
            f1_scores = []
            for th in thresholds:
                edge = np.where(imgs_predict[i] >= th, 1, 0).astype(np.bool)
                f1_scores.append(evaluate.calculate_f1_score(edge, imgs_gt[i]))
            OIS_th += np.max(np.array(f1_scores))
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
        ODS = np.amax(np.array(ODS_th)) / 1000

        print("OIS: %f ODS: %f", OIS, ODS)
    return OIS,ODS


def save_pre_imgs(test_queue, model, ODS=None):
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
            img_GT = target.cpu().detach().numpy().astype('float32')
            img_predict = img_predict.squeeze()
            img_GT = img_GT.squeeze()

            # ---save the image
            if(ODS==None):
              mat_predict = img_predict
              img_predict *= 255.0
            else:
              img_predict[img_predict<ODS]=0
              mat_predict = 1-img_predict
              img_predict = 255.0 * (1-img_predict)
            img_predict = Image.fromarray(np.uint8(img_predict))
            img_predict = img_predict.convert('L')  # single channel
            img_predict.save(os.path.join(predict_folder, 'png', '{}.png'.format(step)))
            io.savemat(os.path.join(predict_folder, 'mat', '{}.mat'.format(step)), {'predict': mat_predict})

            mat_gt = img_GT
            img_GT *= 255.0
            img_GT = Image.fromarray(np.uint8(img_GT))
            img_GT = img_GT.convert('L')
            img_GT.save(os.path.join(gt_folder, 'png', '{}.png'.format(step)))
            io.savemat(os.path.join(gt_folder, 'mat', '{}.mat'.format(step)), {'gt': mat_gt})

    print("save is finished")


def get_builder(dataset):
    if dataset == 'BSD500':
        return build_BSD_500


def cross_entropy_loss(prediction, label):
    label = label.long()
    mask = label.float()

    # prediction = torch.nn.functional.softmax(prediction, 1)
    # ## with channel=1 we get the img[B,H,W]
    # prediction = prediction[:, 1, :, :].unsqueeze(1)

    num_positive = torch.sum((mask == 1).float()).float()
    num_negative = torch.sum((mask == 0).float()).float()

    mask[mask == 1] = 1.0 * num_negative / (num_positive + num_negative)
    mask[mask == 0] = 1.0 * num_positive / (num_positive + num_negative)

    # print(num_negative)
    # print(num_positive)
    cost = torch.nn.functional.binary_cross_entropy_with_logits(
        prediction.float(), label.float(), weight=mask, reduce=False)

    return torch.sum(cost) / (num_negative + num_positive)
    # return torch.sum(cost)


def lr_poly(base_lr, i_iter, max_iter, power):
    return base_lr * ((1 - float(i_iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter, max_iter):
    lr = lr_poly(args.lr_max, i_iter, max_iter, args.lr_schedule_power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def build_BSD_500(model_state_dict, optimizer_state_dict, **kwargs):
    epoch = kwargs.pop('epoch')
    root = "./data/BSR/BSDS500/data/"
    train_data = dataset.BSD_loader(root=root, split='train',random_crop=True,random_flip=False,normalisation=False)
    valid_data = dataset.BSD_loader(root=root, split='val',random_crop=False,random_flip=False,normalisation=False)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, pin_memory=True, num_workers=16, shuffle=True)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.eval_batch_size, pin_memory=True, num_workers=16, shuffle=False)
        # valid_data, batch_size=1, pin_memory=True, num_workers=16, shuffle=False)

    # model = DeepLab(output_stride=16, class_num=1, pretrained=False, freeze_bn=False)
    model = NASUNetBSD(args, args.classes, depth=args.layers, c=args.channels,
                       keep_prob=args.keep_prob,nodes=args.nodes,
                       use_aux_head=args.use_aux_head, arch=args.arch, use_softmax_head=False,
                       double_down_channel=args.double_down_channel)

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
    if model_state_dict is not None:
        model.load_state_dict(model_state_dict)

    if torch.cuda.device_count() > 1:
        logging.info("Use %d %s", torch.cuda.device_count(), "GPUs !")
        model = nn.DataParallel(model)
    model = model.cuda()

    train_criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.065,0.935])).cuda()
    eval_criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.065,0.935])).cuda()
   

    optimizer = torch.optim.SGD(
        # [{'params': model.parameters(), 'initial_lr': args.lr_max}]
        model.parameters(),
        lr=args.lr_max,
        momentum=0.9,
        weight_decay=args.l2_reg,
    )
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs), args.lr_min)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.iterations), args.lr_min, iter)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    # return train_queue, valid_queue, model, optimizer, scheduler
    return train_queue, valid_queue, model, optimizer,scheduler,train_criterion,eval_criterion


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
    _, model_state_dict, epoch, optimizer_state_dict,_ = utils.load_best_model(output_dir)
    build_fn = get_builder(args.dataset)
    train_queue, valid_queue, model, optimizer,scheduler,train_criterion,eval_criterion = build_fn(model_state_dict,
                                                                        optimizer_state_dict,
                                                                        epoch=epoch - 1)

    filename = "./curve/loss.txt"  # --for draw save the loss and ods of valid set
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except:
            logging.info('creat the curve folder failed.')

    train_queue_iter = iter(train_queue)
    epochs_since_start = 0
    loss_valid = 1000
    each_epoch_iter = len(train_queue)
    print(each_epoch_iter)
    total_iter = args.epochs * each_epoch_iter
    print(total_iter)
    i_iter = 0
    logging.info("=====================start training=====================")
    valid_best_OIS=0.
    while epoch < args.epochs:
        # logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
        logging.info('epoch %d lr %e', epoch, optimizer.param_groups[0]['lr'])
        train_OIS, train_obj = train(train_queue, model, optimizer,train_criterion)
        valid_OIS, valid_obj = valid(valid_queue, model,eval_criterion)
        scheduler.step(train_obj)
        is_best = False
        if valid_OIS > valid_best_OIS:
            valid_best_OIS = valid_OIS
            is_best = True
        if is_best:
            logging.info('the current best model is model %d', epoch)
            utils.save_best_model(args.output_dir, args, model, epoch, optimizer, valid_best_OIS, is_best)
            save_pre_imgs(valid_queue,model)
        epoch += 1
        # draw the curve
        with open(filename, 'a+')as f:
            f.write(str(valid_obj.cpu().numpy()))
            f.write(',')
            f.write(str(valid_OIS))
            f.write('\n')
  
    logging.info('train is finished!')
    try:
        loss = []
        accuracy_OIS = []
        with open(filename, 'r') as f:
            for line in f:
                loss.append(eval(line.split(',')[0]))
                accuracy_OIS.append(eval(line.split(',')[1]))
        evaluate.accuracyandlossCurve(loss, accuracy_OIS, args.epochs)
        # evaluate.accuracyandlossCurve(loss, accuracy_ODS, args.iterations // args.val_per_iter)
    except:
        logging.info('the plot of valid set is failed')
        pass

    root = "./data/BSR/BSDS500/data/"
    test_data = dataset.BSD_loader(root=root, split='test',random_crop=False,random_flip=False,normalisation=False)
    test_queue = torch.utils.data.DataLoader(test_data, batch_size=1, pin_memory=True, num_workers=16, shuffle=False)

    logging.info('loading the best model.')
    output_dir = './exp/NAONet_BSD_500/'
    _, model_state_dict, start_epoch, optimizer_state_dict ,_= utils.load_best_model(output_dir)
    build_fn = get_builder(args.dataset)
    train_queue, valid_queue, model, optimizer,scheduler,train_criterion,_ = build_fn(model_state_dict,
                                                                              optimizer_state_dict,
                                                                              epoch=start_epoch - 1)
    _,ODS_test=test(test_queue, model,train_criterion)
    logging.info('test is finished!')
    if (args.save == True):
        try:
            save_pre_imgs(test_queue, model,ODS=ODS_test)
            logging.info('save is finished!')
        except:
            logging.info('save is failed!')


if __name__ == '__main__':
    main()

import sys
import glob
import numpy as np
import torch
from utils import utils, evaluate, dataloader_BSD_Pascal
import logging
import argparse
import random
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
from model.NAO_Unet import NASUNetBSD
from model.decoder import NAOMSCBC
from model.NAO_deeplabv3plus import NAO_deeplabv3plus
from utils import dataset
import utils_deeplabv3plus
import os
from utils_deeplabv3plus import ext_transforms as et
from utils import Cityscapes
from tqdm import tqdm
from metrics import StreamSegMetrics
import time
from torch.utils import data

#parser = argparse.ArgumentParser()

def get_argparser():
    parser = argparse.ArgumentParser()
    # Basic model parameters.
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test'])
    parser.add_argument("--data_root", type=str, default='/home/student/workspace_Yufei/CityScapes/NAO_Cityscapes',
                            help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='cityscapes',
                            choices=['voc', 'cityscapes'], help='Name of datasets')
    parser.add_argument("--num_classes", type=int, default=None,
                            help="num classes (default: None)")
    # Deeplab Options
    parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                        choices=['deeplabv3_resnet50',  'deeplabv3plus_resnet50',
                                 'deeplabv3_resnet101', 'deeplabv3plus_resnet101',
                                 'deeplabv3_mobilenet', 'deeplabv3plus_mobilenet'], help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=8, choices=[8, 16])
    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")
    parser.add_argument("--total_itrs", type=int, default=30e3,
                        help="epoch number (default: 30k)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=16,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=513)

    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=False)

    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=100,
                        help="epoch interval for eval (default: 100)")
    parser.add_argument("--download", action='store_true', default=False,
                        help="download datasets")

    # PASCAL VOC Options
    parser.add_argument("--year", type=str, default='2012',
                        choices=['2012_aug', '2012', '2011', '2009', '2008', '2007'], help='year of VOC')

    # Visdom options
    parser.add_argument("--enable_vis", action='store_true', default=False,
                        help="use visdom for visualization")
    parser.add_argument("--vis_port", type=str, default='13570',
                        help='port for visdom')
    parser.add_argument("--vis_env", type=str, default='main',
                        help='env for visdom')
    parser.add_argument("--vis_num_samples", type=int, default=8,
                        help='number of samples for visualization (default: 8)')

    parser.add_argument('--autoaugment', action='store_true', default=False)
    parser.add_argument('--output_dir', type=str, default='models')
    parser.add_argument('--search_space', type=str, default='with_mor_ops', choices=['with_mor_ops', 'without_mor_ops'])
    parser.add_argument('--layers', type=int, default=5)  # 5
    parser.add_argument('--nodes', type=int, default=5)
    parser.add_argument('--cutout_size', type=int, default=None)
    parser.add_argument('--grad_bound', type=float, default=5.0)
    parser.add_argument('--drop_path_keep_prob', type=float, default=None)
    parser.add_argument('--l2_reg', type=float, default=5e-4)
    parser.add_argument('--arch', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save', type=bool, default=True)
    parser.add_argument('--iterations', type=int, default=20000)
    parser.add_argument('--val_per_iter', type=int, default=10000)
    parser.add_argument('--lr_schedule_power', type=float, default=0.9)
    parser.add_argument('--double_down_channel', type=bool, default=False)
    return parser

args = get_argparser().parse_args()
utils.create_exp_dir(args.output_dir, scripts_to_save=glob.glob('*.py'))
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')

def get_dataset(args):
    """ Dataset And Augmentation
    """
    if args.dataset == 'cityscapes':
        train_transform = et.ExtCompose([
            #et.ExtResize( 512 ),
            et.ExtRandomCrop(size=(args.crop_size, args.crop_size)),
            et.ExtColorJitter( brightness=0.5, contrast=0.5, saturation=0.5 ),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        val_transform = et.ExtCompose([
            #et.ExtResize( 512 ),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        train_dst = Cityscapes(root=args.data_root,
                               split='train', transform=train_transform)
        val_dst = Cityscapes(root=args.data_root,
                             split='val', transform=val_transform)
    return train_dst, val_dst

def valid(args,  model, valid_queue, device, metrics, criterion=None):
    metrics.reset()
    nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

    interval_loss = 0
    # set the mode of model to eval
    model.eval()
    if args.save_val_results:
        if not os.path.exists('results'):
            os.mkdir('results')
        denorm = utils_deeplabv3plus.Denormalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        img_id = 0

    with torch.no_grad():
        for step, (images, labels) in tqdm(enumerate(valid_queue)):
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outs = model(images, labels.size()[2:4])
            preds = outs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)

            loss = criterion(outs, labels)
            np_loss = loss.detach().cpu().numpy()
            interval_loss += np_loss

        interval_loss = interval_loss/step

        score = metrics.get_results()
        logging.info(" MIOU: %f loss: %e", score['Mean IoU'], interval_loss)
    return score

def get_builder(dataset):
    if dataset == 'BSD500':
        return build_BSD_500
    elif dataset == 'cityscapes':
        return  build_NAO_deeplabv3plus_cityscapes


def cross_entropy_loss(prediction, label):
    # ref:https://github.com/mayorx/rcf-edge-detection
    label = label.long()
    mask = label.float()
    num_positive = torch.sum((mask == 1.).float()).float()
    num_negative = torch.sum((mask == 0.).float()).float()
    # print(mask)

    mask[mask == 1] = 1.0 * num_negative / (num_positive + num_negative)
    mask[mask == 0] = 1.1 * num_positive / (num_positive + num_negative)
    mask[mask == 2] = 0

    cost = torch.nn.functional.binary_cross_entropy(
        prediction.float(), label.float(), weight=mask, reduction='none')
    return torch.sum(cost) / (num_negative + num_positive)


def lr_poly(base_lr, i_iter, max_iter, power):
    return base_lr * ((1 - float(i_iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter, max_iter):
    lr = lr_poly(args.lr_max, i_iter, max_iter, args.lr_schedule_power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def build_BSD_500(model_state_dict, optimizer_state_dict, **kwargs):
    # epoch = kwargs.pop('epoch')
    # i_iter = kwargs.pop('i_iter')
    root = "./data/"

    train_data = dataloader_BSD_Pascal.BSD_loader(root=root, split='train', normalisation=False,keep_size=False)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, pin_memory=True, num_workers=16, shuffle=True)

    # model = DeepLab(output_stride=16, class_num=2, pretrained=False, freeze_bn=False)
    # model = NASUNetBSD(args, args.classes, depth=args.layers, c=args.channels,
    #                    keep_prob=args.keep_prob, nodes=args.nodes,
    #                    use_aux_head=args.use_aux_head, arch=args.arch,
    #                    double_down_channel=args.double_down_channel)
    model = NAOMSCBC(args,args.classes,args.arch,channels=42,pretrained=True,res='101')

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
    if model_state_dict is not None:
        model.load_state_dict(model_state_dict)

    if torch.cuda.device_count() > 1:
        logging.info("Use %d %s", torch.cuda.device_count(), "GPUs !")
        model = nn.DataParallel(model)
    model = model.cuda()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr_max,
        momentum=0.9,
        weight_decay=args.l2_reg,
    )
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)

    return train_queue, model, optimizer

def build_NAO_deeplabv3plus_cityscapes(model_state_dict, optimizer_state_dict, **kwargs):
    # epoch = kwargs.pop('epoch')
    # i_iter = kwargs.pop('i_iter')

    train_dst, val_dst = get_dataset(args)
    train_queue = args.DataLoader(
        train_dst, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = args.DataLoader(
        val_dst, batch_size=args.val_batch_size, shuffle=True, num_workers=2)
    print("Dataset: %s, Train set: %d, Val set: %d" %
          (args.dataset, len(train_dst), len(val_dst)))

    model = NAO_deeplabv3plus(args,args.classes,args.arch)
    # if args.separable_conv and 'plus' in args.model:
    #     model.network.convert_to_separable_conv(model.classifier)
    utils_deeplabv3plus.set_bn_momentum(model.backbone, momentum=0.01)

    # Set up metrics
    metrics = StreamSegMetrics(args.num_classes)

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
    if model_state_dict is not None:
        model.load_state_dict(model_state_dict)

    if torch.cuda.device_count() > 1:
        logging.info("Use %d %s", torch.cuda.device_count(), "GPUs !")
        model = nn.DataParallel(model)
    model = model.cuda()

    # Set up optimizer
    optimizer = torch.optim.SGD(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1 * args.lr},
        {'params': model.classifier.parameters(), 'lr': args.lr},
    ], lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    if args.lr_policy == 'poly':
        scheduler = utils_deeplabv3plus.PolyLR(optimizer, args.total_itrs, power=0.9)
    elif args.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    # Set up criterion
    # criterion = utils.get_loss(opts.loss_type)
    if args.loss_type == 'focal_loss':
        criterion = utils_deeplabv3plus.FocalLoss(ignore_index=255, size_average=True)
    elif args.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

    return train_queue, model, optimizer

# def save_ckpt(path):
#         """ save current model
#         """
#         torch.save({
#             "cur_itrs": cur_itrs,
#             "model_state": model.module.state_dict(),
#             "optimizer_state": optimizer.state_dict(),
#             "scheduler_state": scheduler.state_dict(),
#             "best_score": best_score,
#         }, path)
#         print("Model saved as %s" % path)

def main():
    args = get_argparser().parse_args()
    if args.dataset.lower() == 'voc':
        args.num_classes = 21
    elif args.dataset.lower() == 'cityscapes':
        args.num_classes = 19

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.random_seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.enabled = True
    cudnn.benchmark = True

    # Setup dataloader
    if args.dataset == 'voc' and not args.crop_val:
        args.val_batch_size = 1

    logging.info("Args = %s", args)

    train_dst, val_dst = get_dataset(args)
    train_loader = data.DataLoader(
        train_dst, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = data.DataLoader(
        val_dst, batch_size=args.val_batch_size, shuffle=True, num_workers=2)
    print("Dataset: %s, Train set: %d, Val set: %d" %
          (args.dataset, len(train_dst), len(val_dst)))

    model = NAO_deeplabv3plus(args,args.num_classes,args.arch)
    # if args.separable_conv and 'plus' in args.model:
    #     network.convert_to_separable_conv(model.classifier)
    utils_deeplabv3plus.set_bn_momentum(model.NAO_deeplabv3plus.backbone, momentum=0.01)

    # Set up metrics
    metrics = StreamSegMetrics(args.num_classes)

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
    # if model_state_dict is not None:
    #     model.load_state_dict(model_state_dict)

    # if torch.cuda.device_count() > 1:
    #     logging.info("Use %d %s", torch.cuda.device_count(), "GPUs !")
    #     model = nn.DataParallel(model)
    # model = model.cuda()

    # Set up optimizer
    optimizer = torch.optim.SGD(params=[
        {'params': model.NAO_deeplabv3plus.backbone.parameters(), 'lr': 0.1 * args.lr},
        {'params': model.NAO_deeplabv3plus.classifier.parameters(), 'lr': args.lr},
        {'params': model.decoder.parameters(), 'lr': args.lr},
        {'params': model.score.parameters(), 'lr': args.lr},
    ], lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    if args.lr_policy == 'poly':
        scheduler = utils_deeplabv3plus.PolyLR(optimizer, args.total_itrs, power=0.9)
    elif args.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    # Set up criterion
    # criterion = utils.get_loss(opts.loss_type)
    if args.loss_type == 'focal_loss':
        criterion = utils_deeplabv3plus.FocalLoss(ignore_index=255, size_average=True)
    elif args.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')


    def save_ckpt(path):
        """ save current model
        """
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)
    # Restore
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0

    utils_deeplabv3plus.mkdir('checkpoints')
    logging.info("[!] Retrain")
    model = nn.DataParallel(model)
    model.to(device)

    # ==========   Train Loop   ==========#
    denorm = utils_deeplabv3plus.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images

    if args.test_only:
        model.eval()
        val_score, ret_samples = valid(
            args=args, model=model, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
        print(metrics.to_str(val_score))
        return

    scaler = torch.cuda.amp.GradScaler()
    torch.cuda.empty_cache()

    interval_loss = 0
    while True:  # cur_itrs < opts.total_itrs:
        # =====  Train  =====
        model.train()
        cur_epochs += 1
        torch.cuda.synchronize()
        since = int(round(time.time() * 1000))

        for (images, labels) in train_loader:
            cur_itrs += 1

            # images = images.to(device, dtype=torch.float32)
            images = images.to(device, dtype=torch.float16)
            labels = labels.to(device, dtype=torch.long)

            optimizer.zero_grad()
            # Casts operations to mixed precision
            with torch.cuda.amp.autocast():
                outputs = model(images,labels.size()[2:4])
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            np_loss = loss.detach().cpu().numpy()
            interval_loss += np_loss
            # if vis is not None:
            #     vis.vis_scalar('Loss', cur_itrs, np_loss)

            if (cur_itrs) % 10 == 0:
                interval_loss = interval_loss / 10
                print("Epoch %d, Itrs %d/%d, Loss=%f" %
                      (cur_epochs, cur_itrs, args.total_itrs, interval_loss))
                interval_loss = 0.0

            if (cur_itrs) % args.val_interval == 0:
                save_ckpt('checkpoints/latest_%s_%s_os%d.pth' %
                          (args.model, args.dataset, args.output_stride))
                print("validation...")
                model.eval()
                val_score = valid(
                    args=args, model=model, valid_queue=val_loader, device=device, metrics=metrics,
                criterion=criterion)
                print(metrics.to_str(val_score))
                if val_score['Mean IoU'] > best_score:  # save best model
                    best_score = val_score['Mean IoU']
                    save_ckpt('checkpoints/best_%s_%s_os%d.pth' %
                              (args.model, args.dataset, args.output_stride))
                model.train()
            scheduler.step()

            if cur_itrs >= args.total_itrs:
                return

        torch.cuda.synchronize()
        time_elapsed = int(round(time.time() * 1000)) - since
        print('training time elapsed {}ms'.format(time_elapsed))

if __name__ == '__main__':
    main()

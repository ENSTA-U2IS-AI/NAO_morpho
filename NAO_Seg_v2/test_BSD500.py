import sys
import glob
import numpy as np
import torch
from utils import utils, evaluate, dataloader_BSD_Pascal
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
from model.NAO_Unet import NASUNetBSD
from model.decoder import NAOMSCBC
from utils import dataset
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
parser.add_argument('--batch_size', type=int, default=5)  # 8
parser.add_argument('--eval_batch_size', type=int, default=1)
parser.add_argument('--epochs', type=int, default=15)
parser.add_argument('--layers', type=int, default=5)  # 5
parser.add_argument('--nodes', type=int, default=5)
parser.add_argument('--channels', type=int, default=8)  # 16
parser.add_argument('--cutout_size', type=int, default=None)
parser.add_argument('--grad_bound', type=float, default=5.0)
parser.add_argument('--lr_max', type=float, default=1e-2)
parser.add_argument('--lr_min', type=float, default=1e-3)
parser.add_argument('--keep_prob', type=float, default=1)
parser.add_argument('--drop_path_keep_prob', type=float, default=None)
parser.add_argument('--l2_reg', type=float, default=5e-4)
parser.add_argument('--arch', type=str, default=None)
parser.add_argument('--use_aux_head', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--classes', type=int, default=1)
parser.add_argument('--save', type=bool, default=True)
parser.add_argument('--iterations', type=int, default=20000)
parser.add_argument('--val_per_iter', type=int, default=10000)
parser.add_argument('--lr_schedule_power', type=float, default=0.9)
parser.add_argument('--double_down_channel', type=bool, default=False)
args = parser.parse_args()

utils.create_exp_dir(args.output_dir, scripts_to_save=glob.glob('*.py'))
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')


def recursive_glob(rootdir=".", suffix=""):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]

def save_pre_imgs(queue, model, ODS=None):
    from PIL import Image
    import scipy.io as io

    folder = './results/'
    predict_folder = os.path.join(folder, 'predict')
    try:
        os.makedirs(predict_folder)
        os.makedirs(os.path.join(predict_folder, 'png'))
        os.makedirs(os.path.join(predict_folder, 'mat'))
        os.makedirs(os.path.join(predict_folder, 'all'))

    except Exception:
        print('dir already exist....')
        pass

    # set the mode of model to eval
    model.eval()

    with torch.no_grad():
        for step, (input, img_original, file_name) in enumerate(queue):
            logging.info("working on the {}th image... ".format(step))
            h,w = img_original.size()[2:]
            input = input.cuda()

            outs = model(input,(h,w))
            # print(outs[-1].size())

            img_predict = outs[-1].cpu().detach().numpy().astype('float32')
            img_predict = img_predict.squeeze()

            import cv2
            # img_predict = cv2.resize(img_predict, dsize=img_original.size()[2:4][::-1], interpolation=cv2.INTER_LINEAR)dsize=(w,h)!
            io.savemat(os.path.join(predict_folder, 'mat', '{}.mat'.format(file_name[0])), {'result': img_predict})
            # ---save the image
            if (ODS == None):
                img_predict *= 255.0
            else:
                img_predict[img_predict < ODS] = 0
                img_predict = 255.0 * (1 - img_predict)
            img_predict = Image.fromarray(np.uint8(img_predict))
            img_predict.save(os.path.join(predict_folder, 'png', '{}.png'.format(file_name[0])))

            item = 0
            for out in outs:
                item += 1
                out = out.cpu().detach().numpy().squeeze()
                out = (out * 255).astype(np.uint8)
                Image.fromarray(out).save(os.path.join(predict_folder, 'all', '{}-{}.png'.format(file_name[0], item)))

    print("save is finished")

def valid(valid_queue, model,criterion=None):
    objs = utils.AvgrageMeter()
    ODS = utils.AvgrageMeter()
    objs.reset()
    ODS.reset()
    # set the mode of model to eval
    model.eval()

    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = input.cuda()
            target = target.cuda()

            outs = model(input, target.size()[2:4])
            loss = cross_entropy_loss(outs[-1], target.long())
            #all layers
            # loss = 0
            # for out in outs:
            #     loss_ = cross_entropy_loss(out, target.long())
            #     loss += loss_

            ods_ = evaluate.evaluation_ODS(outs[-1], target)
            n = input.size(0)
            objs.update(loss.data, n)
            ODS.update(ods_, n)

            if (step + 1) % 10 == 0:
                logging.info('valid %03d loss %e ODS %f ', step + 1, objs.avg, ODS.avg)

        logging.info(" ODS: %f loss: %e", ODS.avg, objs.avg)
    return ODS.avg

def get_builder(dataset):
    if dataset == 'BSD500':
        return build_BSD_500


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

    return model, optimizer


def main():

    logging.info("Args = %s", args)
    output_dir = './exp/NAONet_BSD_500/'

    _, model_state_dict, start_iteration, optimizer_state_dict = utils.load_model(output_dir)
    build_fn = get_builder(args.dataset)

    root = "./data/BSR/BSDS500/data/images"
    test_list = os.path.join(root,"test.lst")
    if os.path.exists(test_list):
        os.remove(test_list)
    f = open(test_list, 'a')
    for file in recursive_glob(rootdir=os.path.join(root,'test'),suffix="jpg"):
        f.write("test/"+os.path.basename(file))
        f.write('\n')
        print(file)
    f.close()
    test_data = dataloader_BSD_Pascal.BSD_loader(root=root, split='test', keep_size=False)
    test_queue = torch.utils.data.DataLoader(test_data, batch_size=1, pin_memory=True, num_workers=16, shuffle=False)

    logging.info("=====================start testing=====================")
    logging.info('loading the best model.')
    output_dir = './exp/NAONet_BSD_500/'
    _, model_state_dict, start_iteration, optimizer_state_dict = utils.load_model(output_dir)
    print('i_iter', start_iteration)
    build_fn = get_builder(args.dataset)
    model, optimizer = build_fn(model_state_dict,
                                             optimizer_state_dict,
                                             )

    save_pre_imgs(test_queue, model)


if __name__ == '__main__':
    main()

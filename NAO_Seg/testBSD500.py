import sys
import glob
import numpy as np
import torch
from utils import utils,evaluate,dataset
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
from PIL import Image

parser = argparse.ArgumentParser()

# Basic model parameters.
parser.add_argument('--mode', type=str, default='test',
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
parser.add_argument('--channels', type=int, default=16)#64
parser.add_argument('--cutout_size', type=int, default=None)
parser.add_argument('--grad_bound', type=float, default=5.0)
parser.add_argument('--lr_max', type=float, default=1e-1)
parser.add_argument('--lr_min', type=float, default=1e-5)
parser.add_argument('--keep_prob', type=float, default=1)
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

folder = 'results/'
mat_files = os.path.join(folder,'mat')
png_files = os.path.join(folder,'png')
try:
    os.mkdir(png_files)
    os.mkdir(mat_files)
except Exception:
    print('dir already exist....')
    pass

def test(test_queue, model, criterion):
    # set the mode of model to eval
    model.eval()

    with torch.no_grad():
        for step, (input, target) in enumerate(test_queue):
            input = input.cuda()
            target = target.cuda()

            img_predict = model(input)
            loss = criterion(img_predict, target.long())

            ois = evaluate.evaluation_OIS(img_predict, target)
            ods = evaluate.evaluation_ODS(img_predict, target)
            evaluate.save_predict_imgs(img_predict,step)

            n = input.size(0)
            objs.update(loss.data, n)
            OIS.update(ois, n)
            ODS.update(ods, n)

            if (step + 1) % 20 == 0:
                logging.info('test  loss %e OIS %f ODS %f', objs.avg, OIS.avg, ODS.avg)

def get_builder(dataset):
    if dataset == 'BSD500':
        return build_BSD_500
    
def build_BSD_500(model_state_dict, optimizer_state_dict, **kwargs):
    epoch = kwargs.pop('epoch')

    data_path = os.getcwd() + "/data/BSR/BSDS500/data/"

    test_data = dataset.BSD_loader(data_path, type='test', transform=transforms.Compose([
        transforms.ToTensor(),
    ]))

    test_queue = torch.utils.data.DataLoader(
        test_data, batch_size=1, pin_memory=True, num_workers=16,shuffle=False)


    model = NASUNetBSD(args, args.classes, depth=args.layers, c=args.channels, nodes=args.nodes,
                       use_aux_head=args.use_aux_head, arch=args.arch, use_softmax_head=False,
                       double_down_channel=False)

    return test_queue, model



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
    
    loss = []
    accuracy_OIS = []

    args.steps = int(np.ceil(4000 / args.batch_size)) * args.epochs
    logging.info("Args = %s", args)
    output_dir = './exp/NAONet_BSD_500/'
    _, model_state_dict, epoch, step, optimizer_state_dict, best_OIS, best_ODS = utils.load(output_dir)
    build_fn = get_builder(args.dataset)
    test_queue, model = build_fn(model_state_dict, optimizer_state_dict, epoch=epoch-1)

    valid_loss=1000
    train_loss=1000
    while epoch < args.epochs:
        # logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
        logging.info('epoch %d lr %e', epoch, optimizer.param_groups[0]['lr'])
        train_OIS, train_ODS, train_obj, step = train(train_queue, model, optimizer, step, train_criterion)
        logging.info('train_OIS %f train_ODS %f', train_OIS,train_ODS)
        valid_OIS, valid_ODS, valid_obj = valid(valid_queue, model, eval_criterion)
        logging.info('valid_OIS %f valid_ODS %f', valid_OIS, valid_ODS)
        epoch += 1
        scheduler.step(valid_obj)

        is_best = False
        if train_obj>train_loss :
            best_ODS = valid_ODS
            best_OIS = valid_OIS
            train_loss = train_obj
            is_best = True

        elif valid_obj>valid_loss :
            best_ODS = valid_ODS
            best_OIS = valid_OIS
            valid_loss = valid_obj
            is_best = True
        if is_best:
          utils.save(args.output_dir, args, model, epoch, step, optimizer, best_OIS, best_ODS, is_best)
        #draw the curve
        with open('./curve/accuracy_loss_validation.txt','a+')as f:
          f.write(str(valid_obj.cpu().numpy()))
          f.write(',')
          f.write(str(train_obj.cpu().numpy()))
          f.write('\n')

    with open('./curve/accuracy_loss_validation.txt','r') as f:
        for line in f:
          loss.append(eval(line.split(',')[0]))
          accuracy_OIS.append(eval(line.split(',')[1]))

    evaluate.accuracyandlossCurve(loss,accuracy_OIS,args.epochs)
    logging.info('train is finished!')
    
    # data_path = os.getcwd() + "/data/BSR/BSDS500/data/"

    # test(test_queue, model, eval_criterion)
    # logging.info('test is finished!')

if __name__ == '__main__':
    main()

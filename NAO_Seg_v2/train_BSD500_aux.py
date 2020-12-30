import sys
import glob
import numpy as np
import torch
from utils import utils, evaluate,dataloader_BSD_aux_original
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
from model.model import NASUNetBSD
from model.deeplab_v3.decoder import DeepLab
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
parser.add_argument('--batch_size', type=int, default=4)  # 8
parser.add_argument('--eval_batch_size', type=int, default=1)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--layers', type=int, default=4)#5
parser.add_argument('--nodes', type=int, default=5)
parser.add_argument('--channels', type=int, default=16)  # 16
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
parser.add_argument('--double_down_channel', type=bool, default=True)
args = parser.parse_args()

utils.create_exp_dir(args.output_dir, scripts_to_save=glob.glob('*.py'))
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')


def save_pre_imgs(queue, model, ODS=None):
    from PIL import Image
    import scipy.io as io

    folder = './results/'
    predict_folder = os.path.join(folder, 'predict')
    try:
        os.makedirs(predict_folder)
        os.makedirs(os.path.join(predict_folder, 'png'))
        os.makedirs(os.path.join(predict_folder, 'mat'))

    except Exception:
        print('dir already exist....')
        pass
        # set the mode of model to eval
        model.eval()

    with torch.no_grad():
        for step, (input, img_original,file_name) in enumerate(queue):
            # print("dsaldhal")
            input = input.cuda()

            img_predict = model(input)

            img_predict = img_predict.cpu().detach().numpy().astype('float32')
            img_predict = img_predict.squeeze()

            import cv2
            img_predict = cv2.resize(img_predict, dsize=img_original.size()[2:4][::-1], interpolation=cv2.INTER_LINEAR)
            io.savemat(os.path.join(predict_folder, 'mat', '{}.mat'.format(file_name[0])), {'result': img_predict})
            # ---save the image
            if (ODS == None):
                img_predict *= 255.0
            else:
                img_predict[img_predict < ODS] = 0
                img_predict = 255.0 * (1 - img_predict)
            img_predict = Image.fromarray(np.uint8(img_predict))
            img_predict.save(os.path.join(predict_folder, 'png', '{}.png'.format(file_name[0])))


    print("save is finished")


def get_builder(dataset):
    if dataset == 'BSD500':
        return build_BSD_500


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
    
    train_data = dataloader_BSD_aux_original.BSD_loader(root=root, split='train',normalisation=False)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, pin_memory=True, num_workers=16, shuffle=True)

    # model = DeepLab(output_stride=16, class_num=2, pretrained=False, freeze_bn=False)
    model = NASUNetBSD(args, args.classes, depth=args.layers, c=args.channels,
                       keep_prob=args.keep_prob,nodes=args.nodes,
                       use_aux_head=args.use_aux_head, arch=args.arch, 
                       double_down_channel=args.double_down_channel)

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

    logging.info("Args = %s", args)
    output_dir = './exp/NAONet_BSD_500/'
    start_iteration=0
    _, model_state_dict, start_iteration, optimizer_state_dict = utils.load_model(output_dir)
    build_fn = get_builder(args.dataset)
    train_queue, model, optimizer = build_fn(model_state_dict,
                                             optimizer_state_dict,
                                             )

    filename = "./curve/loss.txt"  # --for draw save the loss and ods of valid set
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except:
            logging.info('creat the curve folder failed.')

    each_epoch_iter = len(train_queue)
    print(each_epoch_iter)
    total_iter = args.epochs * each_epoch_iter
    print(total_iter)
    i_iter = start_iteration
    valid_loss=10
    #root = "./data/HED-BSDS"
    #test_data = dataloader_BSD_aux.BSD_loader(root=root, split='test',normalisation=False)
    #test_queue = torch.utils.data.DataLoader(test_data, batch_size=1, pin_memory=True, num_workers=16, shuffle=False)
    logging.info("=====================start training=====================")
    model.train()
    for epoch in range(args.epochs):
       avg_loss = 0.
       for i, (images, labels) in enumerate(train_queue):
           i_iter += 1
           adjust_learning_rate(optimizer, i_iter, total_iter)

           images = images.cuda().requires_grad_()
           labels = labels.cuda()

           outs=model(images)
           loss = cross_entropy_loss(outs[-1], labels)

           optimizer.zero_grad()
           loss.backward()
           nn.utils.clip_grad_norm_(model.parameters(), args.grad_bound)
           optimizer.step()

           avg_loss += float(loss)

           if (i_iter % 100 == 0):
               logging.info('[{}/{}] lr {:e} train_avg_loss {:e} loss {:e}'.format(i_iter,total_iter,optimizer.param_groups[0]['lr'],
                                                                                              avg_loss / 100, float(loss)))
               avg_loss = 0

           if (i_iter % args.val_per_iter == 0):
               logging.info(' save the current model %d', i_iter)
               utils.save_model(args.output_dir, args, model, i_iter, optimizer, is_best=False)

    utils.save_model(args.output_dir, args, model, i_iter, optimizer,is_best=False)


                #save_pre_imgs(test_queue, model)

                # # draw the curve
                # with open(filename, 'a+')as f:
                #     f.write(str(valid_obj.cpu().numpy()))
                #     f.write(',')
                #     f.write(str(valid_OIS))
                #     f.write('\n')
                #
                # model.train()

    root = "./data/HED-BSDS"
    test_data = dataloader_BSD_aux_original.BSD_loader(root=root, split='test')
    test_queue = torch.utils.data.DataLoader(test_data, batch_size=1, pin_memory=True, num_workers=16, shuffle=False)

    logging.info('loading the best model.')
    output_dir = './exp/NAONet_BSD_500/'
    _, model_state_dict, start_iteration, optimizer_state_dict = utils.load_model(output_dir)
    build_fn = get_builder(args.dataset)
    train_queue, model, optimizer = build_fn(model_state_dict,
                                             optimizer_state_dict,
                                             )

    if (args.save == True):
        try:
            save_pre_imgs(test_queue, model)
            logging.info('save is finished!')
        except:
            logging.info('save is failed!')


if __name__ == '__main__':
    main()

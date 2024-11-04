"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import random
import time
import warnings
import argparse
import shutil
import os.path as osp

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR, StepLR, MultiStepLR
from torch.utils.data import DataLoader
import torch.nn.functional as F

import utils
from tllib.modules.domain_discriminator import DomainDiscriminator
from tllib.alignment.cdan import ConditionalDomainAdversarialLoss, _ImageClassifier, CT_Classifier
from tllib.utils.data import ForeverDataIterator
from tllib.utils.metric import accuracy
from tllib.utils.meter import AverageMeter, ProgressMeter
from tllib.utils.logger import CompleteLogger
from tllib.utils.analysis import collect_feature, tsne, a_distance

from conformer_cdan_v1 import Conformer
from vision_transformer import vit_small_patch16_224
import numpy as np

from tllib.self_training.mcc import MinimumClassConfusionLoss

from thop import profile, clever_format

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args: argparse.Namespace):
    logger = CompleteLogger(args.log, args.phase)
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    cudnn.benchmark = True

    # Data loading code
    train_transform = utils.get_train_transform(args.train_resizing, scale=args.scale, ratio=args.ratio,
                                                random_horizontal_flip=not args.no_hflip,
                                                random_color_jitter=False, resize_size=args.resize_size,
                                                norm_mean=args.norm_mean, norm_std=args.norm_std)
    val_transform = utils.get_val_transform(args.val_resizing, resize_size=args.resize_size,
                                            norm_mean=args.norm_mean, norm_std=args.norm_std)
    print("train_transform: ", train_transform)
    print("val_transform: ", val_transform)

    train_source_dataset, train_target_dataset, val_dataset, test_dataset, num_classes, args.class_names = \
        utils.get_dataset(args.data, args.root, args.source, args.target, train_transform, val_transform)

    train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)

    # create model
    print("=> using model '{}'".format(args.arch))
    backbone = utils.get_model(args.arch, pretrain=not args.scratch)
    pool_layer = nn.Identity() if args.no_pool else None
    cnn_classifier = _ImageClassifier(backbone, bottleneck_dim=args.bottleneck_dim,
                                      pool_layer=pool_layer, finetune=not args.scratch).to(device)
    classifier_feature_dim = cnn_classifier.features_dim

    vit_classifier = vit_small_patch16_224(pretrained=False).to(device)

    model_path = args.root + '/vit_small_p16_224-15ec54c9.pth'
    model_dict = vit_classifier.state_dict()
    pretrained_dict = torch.load(model_path, map_location=device)
    temp = {}
    for k, v in pretrained_dict.items():
        try:
            if np.shape(model_dict[k]) == np.shape(v):
                temp[k] = v
        except:
            pass
    model_dict.update(temp)
    vit_classifier.load_state_dict(model_dict)

    classifier = CT_Classifier(cnn_classifier, vit_classifier, num_classes, bottleneck_dim=classifier_feature_dim).to(device)

    # if args.randomized:
    #     domain_discri = DomainDiscriminator(args.randomized_dim, hidden_size=1024).to(device)
    # else:
    #     domain_discri = DomainDiscriminator(classifier_feature_dim * num_classes, hidden_size=1024).to(device)

    # classifier = Conformer(patch_size=16, in_chans=3, num_classes=num_classes, base_channel=64, channel_ratio=4,
    #                        num_med_block=0, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4., qkv_bias=True,
    #                        qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1).to(device)
    # classifier = SwinTransformer(pretrain_img_size=224, patch_size=4, window_size=7, embed_dim=96, depths=(2, 2, 6, 2),
    #                              num_heads=(3, 6, 12, 24)).to(device)
    # model_path = args.root + '/Conformer_tiny_patch16.pth'
    # model_path = args.root + '/Conformer_small_patch16.pth'
    # model_path ='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'
    # classifier.init_weights(model_path)

    # model_dict = classifier.state_dict()
    # pretrained_dict = torch.load(model_path, map_location=device)
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
    # model_dict.update(pretrained_dict)
    # classifier.load_state_dict(model_dict)


    if args.randomized:
        domain_discri_1 = DomainDiscriminator(args.randomized_dim, hidden_size=1024).to(device)
        domain_discri_2 = DomainDiscriminator(args.randomized_dim, hidden_size=1024).to(device)
    else:
        # domain_discri_1 = DomainDiscriminator(384 * num_classes, hidden_size=1024).to(device)
        # domain_discri_2 = DomainDiscriminator(384 * num_classes, hidden_size=1024).to(device)
        domain_discri_1 = DomainDiscriminator(classifier_feature_dim * num_classes, hidden_size=1024).to(device)
        domain_discri_2 = DomainDiscriminator(classifier_feature_dim * num_classes, hidden_size=1024).to(device)
        # domain_discri_2 = DomainDiscriminator(224 * 224, hidden_size=1024).to(device)

    classifier_n_parameters = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    print('number of params_c:', classifier_n_parameters)
    domain_discri_1_n_parameters = sum(p.numel() for p in domain_discri_1.parameters() if p.requires_grad)
    print('number of params_d:', domain_discri_1_n_parameters)
    print('number of params_all:', classifier_n_parameters + domain_discri_1_n_parameters)

    input = torch.randn(1, 3, 224, 224).to(device)
    macs, params = profile(classifier, inputs=(input,))
    macs, params = clever_format([macs, params], "%.3f")
    print(macs, params)

    # param_group = []
    # learning_rate = 1.0
    # for k, v in classifier.named_parameters():
        # if v == classifier.conv_cls_head.parameters():
        #     param_group += classifier.get_cov_cls_parameters()
        # elif v == classifier.bottleneck.parameters():
        #     param_group += classifier.get_cov_bottleneck_parameters()
        # elif v == classifier.trans_cls_head.parameters():
        #     param_group += classifier.get_tra_cls_parameters()
        # else:
        #     param_group += [{'params': v, 'lr': learning_rate * 0.1}]
        # param_group += [{'params': v, 'lr': learning_rate * 0.1}]

    # for k, v in domain_discri_1.named_parameters():
    #     param_group += [{'params': v, 'lr': learning_rate}]
    # for k, v in domain_discri_2.named_parameters():
    #     param_group += [{'params': v, 'lr': learning_rate}]

    # all_parameters = classifier.get_parameters() + domain_discri.get_parameters()
    # define optimizer and lr scheduler
    optimizer = SGD(classifier.get_parameters() + domain_discri_1.get_parameters(), args.lr, momentum=args.momentum,
                    weight_decay=args.weight_decay, nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
    # lr_scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    # lr_scheduler = MultiStepLR(optimizer, milestones=[4, 7, 9], gamma=0.1)

    # define loss function
    domain_adv_1 = ConditionalDomainAdversarialLoss(
        domain_discri_1, entropy_conditioning=args.entropy,
        num_classes=num_classes, features_dim=args.bottleneck_dim, randomized=args.randomized,
        randomized_dim=args.randomized_dim
    ).to(device)
    domain_adv_2 = ConditionalDomainAdversarialLoss(
        domain_discri_2, entropy_conditioning=args.entropy,
        num_classes=num_classes, features_dim=args.bottleneck_dim, randomized=args.randomized,
        randomized_dim=args.randomized_dim
    ).to(device)

    # domain_adv_1 = NuclearWassersteinDiscrepancy(domain_discri_1)
    # domain_adv_2 = NuclearWassersteinDiscrepancy(domain_discri_2)

    mcc_loss = MinimumClassConfusionLoss(temperature=args.temperature)

    # resume from the best checkpoint
    if args.phase != 'train':
        checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
        classifier.load_state_dict(checkpoint)

    # analysis the model
    if args.phase == 'analysis':
        # extract features from both domains
        feature_extractor = nn.Sequential(classifier.backbone, classifier.pool_layer, classifier.bottleneck).to(device)
        source_feature = collect_feature(train_source_loader, feature_extractor, device)
        target_feature = collect_feature(train_target_loader, feature_extractor, device)
        # plot t-SNE
        tSNE_filename = osp.join(logger.visualize_directory, 'TSNE.pdf')
        tsne.visualize(source_feature, target_feature, tSNE_filename)
        print("Saving t-SNE to", tSNE_filename)
        # calculate A-distance, which is a measure for distribution discrepancy
        A_distance = a_distance.calculate(source_feature, target_feature, device)
        print("A-distance =", A_distance)
        return

    if args.phase == 'test':
        acc1 = utils.validate(test_loader, classifier, args, device)
        print(acc1)
        return

    # start training
    best_acc1 = 0.
    for epoch in range(args.epochs):
        print("lr:", lr_scheduler.get_last_lr()[0])
        # train for one epoch
        train(train_source_iter, train_target_iter, classifier, domain_adv_1, domain_adv_2, mcc_loss, optimizer,
              lr_scheduler, epoch, args)
        # lr_scheduler.step()
        # evaluate on validation set
        acc1, acc2 = utils.validate(val_loader, classifier, args, device)
        acc = max(acc1, acc2)
        # remember best acc@1 and save checkpoint
        torch.save(classifier.state_dict(), logger.get_checkpoint_path('latest'))
        if acc > best_acc1:
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
        best_acc1 = max(acc, best_acc1)

    print("best_acc1 = {:3.1f}".format(best_acc1))

    # evaluate on test set
    classifier.load_state_dict(torch.load(logger.get_checkpoint_path('best')))
    acc1, acc2 = utils.validate(test_loader, classifier, args, device)
    acc = max(acc1, acc2)
    print("test_acc1 = {:3.1f}".format(acc))

    logger.close()


def train(train_source_iter: ForeverDataIterator, train_target_iter: ForeverDataIterator, model: Conformer,
          domain_adv_1, domain_adv_2, mcc, optimizer: SGD, lr_scheduler, epoch: int, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':3.1f')
    data_time = AverageMeter('Data', ':3.1f')
    losses = AverageMeter('Loss', ':3.2f')
    trans_losses1 = AverageMeter('Trans Loss1', ':3.2f')
    trans_losses2 = AverageMeter('Trans Loss2', ':3.2f')
    cls_accs1 = AverageMeter('Cls Acc1', ':3.1f')
    cls_accs2 = AverageMeter('Cls Acc2', ':3.1f')
    domain_accs1 = AverageMeter('Domain Acc1', ':3.1f')
    domain_accs2 = AverageMeter('Domain Acc2', ':3.1f')
    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, trans_losses1, trans_losses2, cls_accs1, cls_accs2],
        prefix="Epoch: [{}]".format(epoch))
    # progress = ProgressMeter(
    #     args.iters_per_epoch,
    #     [batch_time, data_time, losses, trans_losses1, cls_accs1],
    #     prefix="Epoch: [{}]".format(epoch))
    # switch to train mode
    model.train()
    domain_adv_1.train()
    domain_adv_2.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        x_s, labels_s = next(train_source_iter)[:2]
        x_t, = next(train_target_iter)[:1]

        x_s = x_s.to(device)
        x_t = x_t.to(device)
        labels_s = labels_s.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # s_cls, s_fea = model(x_s)
        cov_s_cls, cov_s_fea, tra_s_cls, tra_s_fea = model(x_s)
        # cov_ys = ys[0]
        # tra_ys = ys[1]
        # cov_fs = fs[0]
        # tra_fs = fs[1]

        # t_cls, t_fea = model(x_t)
        cov_t_cls, cov_t_fea, tra_t_cls, tra_t_fea = model(x_t)
        # cov_yt = yt[0]
        # tra_yt = yt[1]
        # cov_ft = ft[0]
        # tra_ft = ft[1]

        cov_cls_loss = F.cross_entropy(cov_s_cls, labels_s)
        tra_cls_loss = F.cross_entropy(tra_s_cls, labels_s)
        # mcc_loss_value1 = mcc(cov_t_cls)
        # mcc_loss_value2 = mcc(tra_t_cls)

        # cov_cls_loss = CrossEntropyLabelSmooth(num_classes=num_classes, epsilon=args.smooth)(cov_s_cls, labels_s)
        # tra_cls_loss = CrossEntropyLabelSmooth(num_classes=num_classes, epsilon=args.smooth)(tra_s_cls, labels_s)

        # cov_fea = torch.cat((cov_s_fea, cov_t_fea), dim=0)
        # tra_fea = torch.cat((tra_s_fea, tra_t_fea), dim=0)
        # loss1 = cov_cls_loss + mcc_loss_value1
        # loss2 = tra_cls_loss + mcc_loss_value2
        # loss = loss1 + loss2

        cov_transfer_loss = domain_adv_1(cov_s_cls, cov_s_fea, cov_t_cls, cov_t_fea)
        tra_transfer_loss = domain_adv_1(tra_s_cls, tra_s_fea, tra_t_cls, tra_t_fea)
        # tra_transfer_loss = domain_adv_2(tra_s_cls, tra_s_fea, tra_t_cls, tra_t_fea)
        # fea_transfer_loss = domain_adv_2(cov_s_fea, cov_t_fea, tra_s_fea, tra_t_fea)
        # cov_transfer_loss = domain_adv_1(cov_fea)
        # tra_transfer_loss = domain_adv_2(tra_fea)

        # domain_acc_1 = domain_adv_1.domain_discriminator_accuracy
        # domain_acc_2 = domain_adv_2.domain_discriminator_accuracy

        # loss = cov_cls_loss + tra_cls_loss + cov_transfer_loss + tra_transfer_loss + fea_transfer_loss
        loss = cov_cls_loss + tra_cls_loss + cov_transfer_loss + tra_transfer_loss
        # loss += cov_transfer_loss + tra_transfer_loss
        # loss = cov_cls_loss + tra_cls_loss + fea_transfer_loss

        cls_acc_1 = accuracy(cov_s_cls, labels_s)[0]
        cls_acc_2 = accuracy(tra_s_cls, labels_s)[0]

        # cls_loss = F.cross_entropy(y_s, labels_s)
        # transfer_loss = domain_adv(y_s, f_s, y_t, f_t)
        # domain_acc = domain_adv.domain_discriminator_accuracy
        # loss = cls_loss + transfer_loss * args.trade_off

        # cls_acc = accuracy(y_s, labels_s)[0]

        losses.update(loss.item(), x_s.size(0))
        # domain_accs1.update(domain_acc_1, x_s.size(0))
        # domain_accs2.update(domain_acc_2, x_s.size(0))

        trans_losses1.update(cov_transfer_loss.item(), x_s.size(0))
        trans_losses2.update(tra_transfer_loss.item(), x_s.size(0))
        cls_accs1.update(cls_acc_1, x_s.size(0))
        cls_accs2.update(cls_acc_2, x_s.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CDAN for Unsupervised Domain Adaptation')
    # dataset parameters
    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA', default='Office31', choices=utils.get_dataset_names(),
                        help='dataset: ' + ' | '.join(utils.get_dataset_names()) +
                             ' (default: Office31)')
    parser.add_argument('-s', '--source', help='source domain(s)', nargs='+')
    parser.add_argument('-t', '--target', help='target domain(s)', nargs='+')
    parser.add_argument('--train-resizing', type=str, default='default')
    parser.add_argument('--val-resizing', type=str, default='default')
    parser.add_argument('--resize-size', type=int, default=224,
                        help='the image size after resizing')
    parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                        help='Random resize scale (default: 0.08 1.0)')
    parser.add_argument('--ratio', type=float, nargs='+', default=[3. / 4., 4. / 3.], metavar='RATIO',
                        help='Random resize aspect ratio (default: 0.75 1.33)')
    parser.add_argument('--no-hflip', action='store_true',
                        help='no random horizontal flipping during training')
    parser.add_argument('--norm-mean', type=float, nargs='+',
                        default=(0.485, 0.456, 0.406), help='normalization mean')
    parser.add_argument('--norm-std', type=float, nargs='+',
                        default=(0.229, 0.224, 0.225), help='normalization std')
    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',)
    parser.add_argument('--bottleneck-dim', default=256, type=int,
                        help='Dimension of bottleneck')
    parser.add_argument('--no-pool', action='store_true',
                        help='no pool layer after the feature extractor.')
    parser.add_argument('--scratch', action='store_true', help='whether train from scratch.')
    parser.add_argument('-r', '--randomized', action='store_true',
                        help='using randomized multi-linear-map (default: False)')
    parser.add_argument('-rd', '--randomized-dim', default=1024, type=int,
                        help='randomized dimension when using randomized multi-linear-map (default: 1024)')
    parser.add_argument('--entropy', default=False, action='store_true', help='use entropy conditioning')
    parser.add_argument('--trade-off', default=1., type=float,
                        help='the trade-off hyper-parameter for transfer loss')
    # training parameters
    parser.add_argument('-b', '--batch-size', default=8, type=int,
                        metavar='N', help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-gamma', default=0.001, type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float,
                        metavar='W', help='weight decay (default: 1e-3)',
                        dest='weight_decay')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=1000, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--per-class-eval', action='store_true',
                        help='whether output per-class accuracy during evaluation')
    parser.add_argument("--log", type=str, default='cdan',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    # parser.add_argument('--finetune', default='True', help='finetune from checkpoint')
    parser.add_argument('--smooth', type=float, default=0.1)

    parser.add_argument('--temperature', default=2.0,
                        type=float, help='parameter temperature scaling')

    args = parser.parse_args()
    main(args)


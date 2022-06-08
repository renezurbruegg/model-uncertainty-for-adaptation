#
# SPDX-FileCopyrightText: 2021 Idiap Research Institute
#
# Written by Prabhu Teja <prabhu.teja@idiap.ch>,
#
# SPDX-License-Identifier: MIT


import os
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import wandb
# from datasets import (CityscapesDataset, CrossCityDataset, get_test_transforms,
#                       get_train_transforms)
from generate_pseudo_labels import validate_model
from network import DeeplabMulti as DeepLab
from network import JointSegAuxDecoderModel, NoisyDecoders
from utils import (ScoreUpdater, adjust_learning_rate, cleanup,
                   get_arguments, label_selection, parse_split_list,
                   savelst_tgt, seed_torch, self_training_regularized_infomax,
                   self_training_regularized_infomax_cct, set_logger)

import numpy as np
from train_utils import get_train_transforms, get_valid_transforms, FourierTransform, DataLoaderSegmentation, \
  get_validation_transform

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
osp = os.path

# PATHS
path_testset = "/media/rene/Empty/TestSets/Flat/LivingRoomWithDresser/"
path_datasets= "/media/rene/Empty/airsim_paper/clean/livingroom/Dataset/"
path_cp = "/home/rene/catkin_ws/src/refinenet_fork/light-weight-refinenet/scannet_50_classes_40_clusters.pth"

args = get_arguments()
if not os.path.exists(args.save):
    os.makedirs(args.save)
logger = set_logger(args.save, 'training_logger', False)

from embodied_active_learning_core.utils.pytorch.evaluation import score_model_for_dl



def score(model):
  model.seg_model.evaluation_mode = True
  model.seg_model.eval()

  test_loader_nyu = torch.utils.data.DataLoader(
    DataLoaderSegmentation(path_testset,
                           transform=get_validation_transform(
                             normalize=False,
                             additional_targets={'mask': 'mask'}), num_imgs=120,
                           verbose=False), batch_size=1)
  miou_nyu, miou_per_classes_nyu, acc_nyu = score_model_for_dl(model, test_loader_nyu)
  print(miou_nyu)
  miou_eigen, miou_per_classes_eigen, acc_eigen = score_model_for_dl(model, test_loader_nyu, map_to_eigen=True,  n_classes=13)
  model.seg_model.evaluation_mode = False

  print("Scored: NYU:", miou_nyu, "Eigen:", miou_eigen)
  wandb.log({
    "test/nyu": miou_nyu,
    "test/eigen": miou_eigen
  })

  return miou_nyu, miou_eigen


def make_network(args):
    # MODEL SETUP
    # define segmentation model and predict function
    # model = models.__dict__[args.arch](num_classes=train_source_dataset.num_classes).to(device)
    from embodied_active_learning_core.semseg.models import model_manager
    from embodied_active_learning_core.config.config import NetworkConfig, \
        NETWORK_CONFIG_ONLINE_CLUSTERED_LIGHTWEIGHT_REFINENET

    def refinenet_from_config(num_classes=19, pretrained_backbone=True):
        config = NetworkConfig()
        config.name = NETWORK_CONFIG_ONLINE_CLUSTERED_LIGHTWEIGHT_REFINENET
        config.checkpoint = path_cp
        u_model = model_manager.get_model_for_config(config)
        return u_model.base_model

    model = refinenet_from_config()
    model.evaluation_mode = True

    if args.unc_noise:
        aux_decoders = NoisyDecoders(args.decoders, args.dropout)
        model = JointSegAuxDecoderModel(model, aux_decoders)
    return model


import time

def train(mix_trainloader, model, interp, optimizer, args):
    """Create the model and start the training."""
    tot_iter = len(mix_trainloader)
    for i_iter, batch in enumerate(mix_trainloader):
        n = time.time()
        # images, labels, name = batch
        images = batch['image']
        labels = batch['mask']
        # names = batch['name']
        labels = labels.long()

        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter, tot_iter, args)

        if args.info_max_loss:
            pred = model(images.to(device), training=True)
            loss = self_training_regularized_infomax(pred, labels.to(device), args)
        elif args.unc_noise:
            pred, noise_pred = model(images.to(device), training=True)
            loss = self_training_regularized_infomax_cct(pred, labels.to(device), noise_pred, args)
        else:
            pred = model(images.to(device))
            loss = F.cross_entropy(pred, labels.to(device), ignore_index=255)


        loss.backward()
        optimizer.step()

        wandb.log({
          "train/loss": loss.item(),
        })

        logger.info('iter = {} of {} completed, loss = {:.4f}'.format(i_iter+1, tot_iter, loss.item()))
        elapsed = ((time.time() - n) * (tot_iter - i_iter) )// 60
        logger.info(f"ETA: {elapsed // 60}H, {elapsed - ((elapsed // 60) * 60)}s")

def main():
    # seed_torch(args.randseed)

    wandb.init(project="eal", entity="eal_da", name="mcfa_135_" + args.run_name)
    logger.info('Starting training with arguments')
    logger.info(vars(args))

    save_path = args.save
    save_pseudo_label_path = osp.join(save_path, 'pseudo_label')  # in 'save_path'. Save labelIDs, not trainIDs.
    save_stats_path = osp.join(save_path, 'stats')  # in 'save_path'
    save_lst_path = osp.join(save_path, 'list')

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(save_pseudo_label_path):
        os.makedirs(save_pseudo_label_path)
    if not os.path.exists(save_stats_path):
        os.makedirs(save_stats_path)
    if not os.path.exists(save_lst_path):
        os.makedirs(save_lst_path)

    tgt_portion = args.init_tgt_port
    # image_tgt_list, image_name_tgt_list, _, _ = parse_split_list(args.data_tgt_train_list.format(args.city))

    model = make_network(args).to(device)

    score(model)


    for round_idx in range(args.num_rounds):
        save_round_eval_path = osp.join(args.save, str(round_idx))
        save_pseudo_label_color_path = osp.join(
            save_round_eval_path, 'pseudo_label_color')  # in every 'save_round_eval_path'
        if not os.path.exists(save_round_eval_path):
            os.makedirs(save_round_eval_path)
        if not os.path.exists(save_pseudo_label_color_path):
            os.makedirs(save_pseudo_label_color_path)

        src_portion = args.init_src_port
        ########## pseudo-label generation
        conf_dict, pred_cls_num, save_prob_path, save_pred_path = validate_model(model,
                                                                                 save_round_eval_path,
                                                                                 round_idx, args)
        cls_thresh = label_selection.kc_parameters(
            conf_dict, pred_cls_num, tgt_portion, round_idx, save_stats_path, args)

        label_selection.label_selection(cls_thresh, round_idx, save_prob_path, save_pred_path,
                                        save_pseudo_label_path, save_pseudo_label_color_path, save_round_eval_path, args)

        tgt_portion = min(tgt_portion + args.tgt_port_step, args.max_tgt_port)
        # tgt_train_lst = savelst_tgt(image_tgt_list, image_name_tgt_list, save_lst_path, save_pseudo_label_path)

        rare_id = np.load(save_stats_path + '/rare_id_round' + str(round_idx) + '.npy')
        mine_id = np.load(save_stats_path + '/mine_id_round' + str(round_idx) + '.npy')
        # mine_chance = args.mine_chance

        # src_transforms, tgt_transforms = get_train_transforms(args, mine_id)
        from train_utils import ScanNetDatset, LivingRoomKitchtenDataset
        srcds = ScanNetDatset(transform=get_train_transforms(normalize=False), limit=-135)

        tgtds = LivingRoomKitchtenDataset(path = path_datasets + args.run_name,
                                  pseudo_root=save_pseudo_label_path, transform=get_train_transforms(normalize=False))
        
        if args.no_src_data:
            mixtrainset = tgtds
        else:
            mixtrainset = torch.utils.data.ConcatDataset([srcds, tgtds])

        mix_loader = DataLoader(mixtrainset, batch_size=args.batch_size, shuffle=True,
                                num_workers=args.batch_size, pin_memory=torch.cuda.is_available())
        src_portion = min(src_portion + args.src_port_step, args.max_src_port)
        optimizer = optim.SGD(model.optim_parameters(args), lr=args.learning_rate,
                              momentum=args.momentum, weight_decay=args.weight_decay)
        interp = nn.Upsample(size=args.input_size[::-1], mode='bilinear', align_corners=True)
        torch.backends.cudnn.enabled = True  # enable cudnn
        torch.backends.cudnn.benchmark = True
        start = time.time()
        for epoch in range(args.epr):
            train(mix_loader, model, interp, optimizer, args)
            print('taking snapshot ...')
            torch.save(model.state_dict(), osp.join(args.save,
                                                    '2nthy_round' + str(round_idx) + '_epoch' + str(epoch) + '.pth'))
        end = time.time()
        
        logger.info('###### Finish model retraining dataset in round {}! Time cost: {:.2f} seconds. ######'.format(
            round_idx, end - start))
        score(model)
        cleanup(args.save)
    cleanup(args.save)
    shutil.rmtree(save_pseudo_label_path)
    score(model)


if __name__ == "__main__":
    main()

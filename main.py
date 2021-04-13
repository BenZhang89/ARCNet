# -*- coding: utf-8 -*-
# Implementation of Wang et al 2017: Automatic Brain Tumor Segmentation using Cascaded Anisotropic Convolutional Neural Networks. https://arxiv.org/abs/1709.00382

# Author: Guotai Wang
# Copyright (c) 2017-2018 University College London, United Kingdom. All rights reserved.
# http://cmictig.cs.ucl.ac.uk
#
# Distributed under the BSD-3 licence. Please see the file licence.txt
# This software is not certified for clinical use.
#
from __future__ import absolute_import, print_function
import argparse
import torch
import numpy as np
import random
from scipy import ndimage
import time
import os
import sys
from util.data_loader import DataLoader
from util.parse_config import parse_config
from util.log_utils import LogWriter
from util.ARCNet import ARCNet
import logging
import shutil
from solver import Solver

torch.set_default_tensor_type('torch.FloatTensor')

def get_patient_names(data_names,data_root=None):
    """
    get the list of patient names, if self.data_names id not None, then load patient 
    names from that file, otherwise search all the names automatically in data_root
    """
    # use pre-defined patient names
    if(data_names is not None):
        #load data_names.txt list
        if os.path.isfile(data_names):
            with open(data_names) as f:
                content = f.readlines()
            patient_names = [x.strip() for x in content]
        #load a single assigned case
        else:
            patient_names = [data_names]

    # use all the patient names in data_root
    else:
        patient_names = os.listdir(data_root)
        patient_names = [name for name in patient_names if 'brats' in name.lower()]
    return patient_names

def train(config_file):
    # 1, load configuration parameters
    config = parse_config(config_file)
    config_common = config['common']
    config_data  = config['data']
    config_net   = config['network']
    config_train = config['training']
        
    random.seed(config_train.get('random_seed', 1))
    assert(config_data['with_ground_truth'])

    class_num   = config_net['class_num']

    # 2 load data
    data_names = get_patient_names(config_data["data_names"])
    split_ratio = int(config_train["train_val_ratio"]*len(data_names))
    random.Random(42).shuffle(data_names)
    config_data["train_names"] = data_names[:split_ratio]
    config_data["val_names"] = data_names[split_ratio:]
    dataset_tr = DataLoader("train",config_data)
    dataset_tr.load_data()
    train_loader = torch.utils.data.DataLoader(dataset_tr, batch_size=config_train['train_batch_size'], shuffle=True,
                                               num_workers=4, pin_memory=True)

    dataset_val = DataLoader("validate",config_data)
    dataset_val.load_data()
    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=config_train['val_batch_size'], shuffle=False,
                                             num_workers=4, pin_memory=True)
   
    # 3, load model
    # load pretrained
    empty_model = ARCNet(class_num, vae_enable=config_train['vae_enable'],config=config_data)
    if config_train['model_pre_trained']:
        arcnet_model = torch.load(config_train['model_pre_trained'])
    else:
        arcnet_model = ARCNet(class_num, vae_enable=config_train['vae_enable'],config=config_data)

    # 4, start to train
    solver = Solver(arcnet_model,
                    exp_name=config_train['exp_name'],
                    device=config_common['device'],
                    class_num=config_net['class_num'],
                    optim_args={"lr": config_train['learning_rate'],
                                "betas": config_train['optim_betas'],
                                "eps": config_train['optim_eps'],
                                "weight_decay": config_train['optim_weight_decay']},
                    loss_args={"vae_loss": config_train['vae_enable'],
                                "loss_k1_weight": config_train['loss_k1_weight'],
                                "loss_k2_weight": config_train['loss_k2_weight']},
                    model_name=config_common['model_name'],
                    labels=config_data['labels'],
                    log_nth=config_train['log_nth'],
                    num_epochs=config_train['num_epochs'],
                    lr_scheduler_step_size=config_train['lr_scheduler_step_size'],
                    lr_scheduler_gamma=config_train['lr_scheduler_gamma'],
                    use_last_checkpoint=config_train['use_last_checkpoint'],
                    log_dir=config_common['log_dir'],
                    exp_dir=config_common['exp_dir'])

    solver.train(train_loader, val_loader)
    if not os.path.exists(config_common['save_model_dir']):
        os.makedirs(config_common['save_model_dir'])
    final_model_path = os.path.join(config_common['save_model_dir'], config_train['final_model_file'])
    solver.model = empty_model
    solver.save_best_model(final_model_path)
    print("final model saved @ " + str(final_model_path))


def pred(eval_bulk):
    data_dir = eval_bulk['data_dir']
    prediction_path = eval_bulk['save_predictions_dir']
    volumes_txt_file = eval_bulk['volumes_txt_file']
    device = eval_bulk['device']
    label_names = ["vol_ID", "Background", "Tumor Core", "Edema", "Enhancing"]
    batch_size = eval_bulk['batch_size']
    need_unc = eval_bulk['estimate_uncertainty']
    mc_samples = eval_bulk['mc_samples']
    dir_struct = eval_bulk['directory_struct']
    if 'exit_on_error' in eval_bulk.keys():
        exit_on_error = eval_bulk['exit_on_error']
    else:
        exit_on_error = False

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-m', required=True, help='run mode, valid values are train and eval')
    parser.add_argument('--config_path', '-cp', required=True, help='optional path to config file')
    args = parser.parse_args()

    if args.mode == 'train':
        train(args.config_path)
    elif args.mode == 'pred':
        logging.basicConfig(filename='error.log')
        if args.config_path is not None:
            settings_eval = Settings(args.config_path)
        else:
            settings_eval = Settings('settings_eval.ini')
        evaluate_bulk(settings_eval['EVAL_BULK'])
    else:
        raise ValueError('Invalid value for mode. only support values are train, eval and clear')

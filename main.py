from __future__ import print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tqdm
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset import UnpairedDataset, PairedDataset
from models import AttentiveSwapNet
from datetime import datetime
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream, reset_tensor, dumb_config, \
    one_epoch_train, save_model
import sklearn.metrics as metrics
from configs import get_cfg_defaults


def _init_(cfg):
    infix_enc = f"Encoder{cfg.ATTENTION.ENCODER_N_LAYER}-linEmb{cfg.ATTENTION.LINEAR_EMB}-head{cfg.ATTENTION.N_HEAD}"
    infix_attn = f"_{cfg.TRAIN.N_DIV}_{cfg.TRAIN.MIXUP_LEVEL}Level"
    dir_name = f"{infix_enc}{infix_attn}_BS{cfg.TRAIN.BATCH_SIZE}_Opt-{cfg.TRAIN.OPT}_"
    cfg.EXP.NAME = dir_name + cfg.EXP.NAME
    if not os.path.exists(cfg.EXP.WORKING_DIR):
        os.makedirs(cfg.EXP.WORKING_DIR)
    if not os.path.exists(f'{cfg.EXP.WORKING_DIR}/'+cfg.EXP.NAME):
        os.makedirs(f'{cfg.EXP.WORKING_DIR}/'+cfg.EXP.NAME)
    if not os.path.exists(f'{cfg.EXP.WORKING_DIR}/'+cfg.EXP.NAME+'/'+'models'):
        os.makedirs(f'{cfg.EXP.WORKING_DIR}/'+cfg.EXP.NAME+'/'+'models')
    os.system(f'cp main.py {cfg.EXP.WORKING_DIR}'+'/'+cfg.EXP.NAME+'/'+'main.py.backup')
    os.system(f'cp models/model.py {cfg.EXP.WORKING_DIR}' + '/' + cfg.EXP.NAME + '/' + 'model.py.backup')
    os.system(f'cp models/attentive.py {cfg.EXP.WORKING_DIR}' + '/' + cfg.EXP.NAME + '/' + 'attentive.py.backup')
    os.system(f'cp util.py {cfg.EXP.WORKING_DIR}' + '/' + cfg.EXP.NAME + '/' + 'util.py.backup')
    os.system(f'cp dataset/data_loader.py {cfg.EXP.WORKING_DIR}' + '/' + cfg.EXP.NAME + '/' + 'data_loader.py.backup')
    print("Experiment will be stored in:", f'{cfg.EXP.WORKING_DIR}/'+cfg.EXP.NAME)

def create_data_loader(cfg):
    multi_workers = 4
    dataset_train_paired = PairedDataset(train=True, num_points=cfg.EXP.NUM_POINTS, io=io, n_sample=cfg.TRAIN.N_DIV,
                                         is_uniform=cfg.EXP.UNIFORM_DATASET)

    dataset_train_unpaired = UnpairedDataset(train=True, num_points=cfg.EXP.NUM_POINTS, io=io, is_uniform=cfg.EXP.UNIFORM_DATASET)

    train_loader = DataLoader(dataset_train_unpaired,
                              num_workers=multi_workers,
                              batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, drop_last=False)

    dataset_test_unpaired = UnpairedDataset(train=False, num_points=cfg.EXP.NUM_POINTS,
                                            io=io, is_uniform=cfg.EXP.UNIFORM_DATASET)
    test_loader = DataLoader(dataset_test_unpaired, num_workers=multi_workers,
                             batch_size=cfg.TEST.BATCH_SIZE, shuffle=True, drop_last=False)

    return train_loader, test_loader, dataset_train_paired

def create_network(cfg,device):
    model = AttentiveSwapNet(cfg=cfg, output_channels=40).to(device)
    model = nn.DataParallel(model)
    return model

def train(cfg, io):
    train_loader, test_loader, dataset_train_paired = create_data_loader(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    model = create_network(cfg,device)

    opt = optim.SGD(model.parameters(), lr=cfg.TRAIN.LR, momentum=cfg.TRAIN.MOMENTUM, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(opt, cfg.TRAIN.N_EPOCHS, eta_min=cfg.TRAIN.LR)
    criterion = cal_loss

    is_mix_start = False
    never_start_mixing = True
    print("Train from beginning!")

    best_test_acc = 0
    best_epoch = -1
    for epoch in range(cfg.TRAIN.N_EPOCHS):
        scheduler.step()
        ####################
        # Train
        ####################
        acc_score, balanced_acc_score, acc_score_mixup, balanced_acc_score_mixup, loss_train = one_epoch_train(cfg, model, train_loader, opt, criterion,
                                                                                                               device, epoch, is_mix_start=is_mix_start)
        # warm start for Point MixSwap
        if never_start_mixing:
            is_mix_start = epoch >= 19
            if is_mix_start:
                dataset_train_paired.init_mixing(model)  # to save the attention right before "mixup training"
                train_loader = DataLoader(dataset_train_paired, num_workers=8,
                                          batch_size=(int)(np.floor(cfg.TRAIN.BATCH_SIZE / cfg.TRAIN.N_DIV)),
                                          shuffle=True)
                save_model(cfg, epoch, model, opt, scheduler, loss_train, loss_test=-1,
                           is_mix_start=is_mix_start, never_start_mixing=never_start_mixing,
                           suffix_name=f'_att-{epoch}')
                never_start_mixing = False
        outstr = f'Train {epoch}, loss: {loss_train}, train acc: {acc_score}, train avg acc: {balanced_acc_score}, train acc mixup: {acc_score_mixup}, ' \
                 f'train avg acc mixup: {balanced_acc_score_mixup}'
        reset_tensor([acc_score, balanced_acc_score, acc_score_mixup, balanced_acc_score_mixup, loss_train])
        delta_rounded = 5
        rounded_train_acc = (int)(np.floor(acc_score*100/delta_rounded)*delta_rounded)
        suffix_name = f'_acc{rounded_train_acc}'
        model_path = f'{cfg.EXP.WORKING_DIR}/{cfg.EXP.NAME}/models/model{suffix_name}.t7'
        if not os.path.exists(model_path):
            save_model(cfg, epoch, model, opt, scheduler, loss_train, loss_test=-1,
                       is_mix_start=is_mix_start, never_start_mixing=never_start_mixing,
                       suffix_name=suffix_name)
        io.cprint(outstr)

        ####################
        # Test
        ####################
        loss_test = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        for data, label in tqdm.tqdm(test_loader, desc=f"Test {epoch}/{cfg.TRAIN.N_EPOCHS}"):
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            logits_cls, _ = model(data)
            loss = criterion(logits_cls, label, smoothing=cfg.TRAIN.LABEL_SMOOTH)
            preds = logits_cls.max(dim=1)[1]
            count += batch_size
            loss_test += loss.item() * batch_size
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        loss_test = loss_test * 1.0 / count
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch
            save_model(cfg, epoch, model, opt, scheduler, loss_train, loss_test, is_mix_start, never_start_mixing)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f, best acc: %.4f at epoch %d' % (epoch,
                                                                                                          loss_test,
                                                                                                          test_acc,
                                                                                                          avg_per_class_acc,
                                                                                                          best_test_acc, best_epoch)
        reset_tensor([data, label, logits_cls, loss])
        io.cprint(outstr)


if __name__ == "__main__":
    # training settings
    parser = argparse.ArgumentParser(description='Point MixSwap')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='path of yaml config file')
    args = parser.parse_args()

    # read the config file
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config)

    # torch.manual_seed(args.i)
    torch.backends.cudnn.deterministic = True
    np.random.seed(cfg.EXP.NUMPY_RANDOM_SEED)

    torch.manual_seed(cfg.EXP.TORCH_RANDOM_SEED)
    # dumb related files to working dir
    _init_(cfg)
    io = IOStream(f'{cfg.EXP.WORKING_DIR}/' + cfg.EXP.NAME + '/run.log')
    io.cprint(str(args))
    clean_cfg = dumb_config(cfg, args)
    io.cprint(clean_cfg)

    time_now = datetime.now()
    time_string = time_now.strftime("%Y-%m-%d_%H-%M-%S")
    io.cprint(f"Time: {time_string}")
    if torch.cuda.is_available():
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(
                torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(cfg.EXP.TORCH_RANDOM_SEED)
    else:
        io.cprint('Using CPU')

    train(cfg, io)
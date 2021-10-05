import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from sklearn import metrics
import torch.utils.data as Data
from training.load_data import load_EOD_data, get_batch, get_industry_adj
from loss.batch_loss import Batch_Loss
import pandas as pd
import numpy as np

adj = torch.Tensor(get_industry_adj())
eod_data, ground_truth = load_EOD_data()
loss_func = Batch_Loss(0.0025, 0.001, 0.01, 0.01, device="cuda")

def train_epoch(model,training_data, optimizer, device, smoothing, args):
    ''' Epoch operation in training phase'''

    model.train()
    total_loss = 0
    total_portfolio_value = 0
    batch_size = args.batch_size
    stock = adj.shape[0]
    out_init = torch.ones(batch_size, 1, stock)/stock

    for step, (eod, gt) in enumerate(training_data):

        Eod, Gt, adj_= eod.to(device), gt.to(device), adj.to(device)

        # forward
        optimizer.zero_grad()
        out = model(Eod,  adj_, out_init, args.hidden)

        loss, portfolio_value, SR, MDD= loss_func(out, Gt)
        # backward

        loss.backward()
        optimizer.step_and_update_lr()
        total_loss += loss.item()
        total_portfolio_value += portfolio_value.item()
        SR += SR.item()


    return total_loss, total_portfolio_value, SR, MDD


def eval_epoch(model, validation_data, device, args):
    ''' Epoch operation in evaluation phase '''

    model.eval()

    total_loss = 0
    total_portfolio_value = 0

    batch_size = args.batch_size
    stock = adj.shape[0]
    out = torch.ones(batch_size, 1, stock) / stock


    with torch.no_grad():
        for step, (eod, gt) in enumerate(validation_data):

            Eod, Gt, adj_ = eod.to(device), gt.to(device), adj.to(device)

            out = model(Eod, adj_, out, args.hidden)
            loss, portfolio_value, SR, MDD = loss_func(out, Gt)
            total_loss += loss.item()
            total_portfolio_value += portfolio_value.item()
            SR += SR.item()

    return total_loss, total_portfolio_value, SR, MDD



def train(model, training_data, validation_data, optimizer, device, args):
    ''' Start training '''

    if args.log:
        log_train_file = args.log + '.train.log'
        log_valid_file = args.log + '.valid.log'

        print('[Info] Training performance will be written to file: {} and {}'.format(
            log_train_file, log_valid_file))

        with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
            log_tf.write('epoch, loss,  portfolio_value, SR, MDD\n')
            log_vf.write('epoch, loss,  portfolio_value, SR, MDD\n')

    valid_port = []
    Train_Loss_list = []
    Val_Loss_list = []

    for epoch_i in range(args.epoch):
        print('[ Epoch', epoch_i, ']')
        start = time.time()
        train_loss, train_total_portfolio, train_SR, train_MDD = train_epoch(
            model, training_data, optimizer, device, smoothing=args.label_smoothing, args=args)

        Train_Loss_list.append(train_loss)

        print(' - (Training) loss:{loss}, portfolio_value:{portfolio_value}, SR:{SR}, MDD:{MDD}'\
            'elapse: {elapse:3.3f} min'.format(loss=train_loss,
                                               portfolio_value=train_total_portfolio,
                                               SR=train_SR,
                                               MDD=train_MDD,
                                               elapse=(time.time() - start) / 60))

        start = time.time()
        valid_loss, valid_total_portfolio, valid_SR, valid_MDD= eval_epoch(model, validation_data, device, args=args)
        Val_Loss_list.append(valid_loss)
        print(
            ' - (Validation) loss:{loss}, portfolio_value:{portfolio_value}, SR:{SR},  MDD:{MDD}' \
            'elapse: {elapse:3.3f} min'.format(
                loss=valid_loss,
                portfolio_value=valid_total_portfolio,
                SR=valid_SR,
                MDD=valid_MDD,
                elapse=(time.time() - start) / 60))

        valid_port += [valid_total_portfolio]

        model_state_dict = model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'settings': args,
            'epoch': epoch_i}

        if args.save_model:
            if args.save_mode == 'all':
                model_name = args.save_model + '_port_{port}.chkpt'.format(port=valid_total_portfolio)
                torch.save(checkpoint, model_name)
            elif args.save_mode == 'best':
                model_name = args.save_model + '.chkpt'
                if valid_total_portfolio >= max(valid_port):
                    torch.save(checkpoint, model_name)
                    print('    - [Info] The checkpoint file has been updated.')

        if log_train_file and log_valid_file:
            with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                log_tf.write(
                    '{epoch: 4.0f},{loss},{port},{SR},{MDD}\n'.format(
                        epoch=epoch_i, loss=train_loss, port=train_total_portfolio, SR=train_SR, MDD=train_MDD))
                log_vf.write(
                    '{epoch: 4.0f},{loss},{port},{SR},{MDD}\n'.format(
                        epoch=epoch_i, loss=valid_loss, port=valid_total_portfolio, SR=valid_SR, MDD=valid_MDD))

    if log_valid_file:
        with open(log_valid_file, 'a') as log_vf:
            log_vf.write('{Best}\n'.format(Best= max(valid_port)))
            log_vf.write('{Best_epoch}\n'.format(Best_epoch=valid_port.index(max(valid_port))))


def prepare_dataloaders(eod_data, gt_data, args):
    # ========= Preparing DataLoader =========#
    EOD, GT = [], []
    for i in range(eod_data.shape[1] - args.length):
        eod, gt = get_batch(eod_data, gt_data, i, args.length)
        EOD.append(eod)
        GT.append(gt)

    train_eod, train_gt = EOD[:args.train_index], GT[:args.train_index]
    valid_eod, valid_gt = EOD[args.train_index:args.valid_index], GT[args.train_index:args.valid_index]
    test_eod, test_gt = EOD[args.valid_index:], GT[args.valid_index:]

    train_eod, valid_eod, test_eod = torch.FloatTensor(train_eod), torch.FloatTensor(valid_eod), torch.FloatTensor(test_eod)
    train_gt, valid_gt, test_gt = torch.FloatTensor(train_gt), torch.FloatTensor(valid_gt), torch.FloatTensor(test_gt)
    train_gt, valid_gt, test_gt = train_gt.unsqueeze(2), valid_gt.unsqueeze(2), test_gt.unsqueeze(2)


    train_dataset = Data.TensorDataset(train_eod, train_gt)
    valid_dataset = Data.TensorDataset(valid_eod, valid_gt)
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    valid_loader = Data.DataLoader(dataset=valid_dataset, batch_size=args.batch_size, drop_last=True)


    return train_loader, valid_loader
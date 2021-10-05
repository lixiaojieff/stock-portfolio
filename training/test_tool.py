import torch
import torch.nn.functional as F
import time
from sklearn import metrics
import torch.utils.data as Data
from training.load_data import load_EOD_data, get_batch, get_industry_adj
from loss.batch_loss import Batch_Loss
import pandas as pd
import numpy as np
import os

adj = torch.Tensor(get_industry_adj())
eod_data, ground_truth = load_EOD_data()

loss_func = Batch_Loss(0.0025, 0.001, 0.01, 0.01, device="cuda")


def eval_epoch(model, validation_data, device, args):
    ''' Epoch operation in evaluation phase '''

    model.eval()

    total_loss = 0
    total_portfolio_value = 1
    batch_size = args.batch_size
    stock = adj.shape[0]
    out = torch.ones(batch_size, 1, stock) / stock


    with torch.no_grad():
        for step, (eod, gt) in enumerate(validation_data):

            Eod, Gt, adj_ = eod.to(device), gt.to(device), adj.to(device)

            # forward
            out = model(Eod, adj_, out, args.hidden)
            loss, portfolio_value, SR, MDD = loss_func(out, Gt)
            total_loss += loss.item()
            total_portfolio_value *= portfolio_value.item()
            SR += SR.item()

    return total_loss, portfolio_value, SR, MDD




def train(model, test_data, optimizer, device, args):

    log_dir = args.save_model + '.chkpt'
    if os.path.exists(log_dir):
        checkpoint = torch.load(log_dir)
        model.load_state_dict(checkpoint['model'])
        # args.load_state_dict(checkpoint['settings'])
        start_epoch = checkpoint['epoch']
        print('load epoch {} successfully！'.format(start_epoch))
    else:

        print('no save model, start training from the beginning！')
    start = time.time()
    valid_loss, valid_total_portfolio, valid_SR, valid_MDD = eval_epoch(model, test_data, device,
                                                                                  args=args)

    print(
        ' - (Validation) loss:{loss}, portfolio_value:{portfolio_value}, SR:{SR},  MDD:{MDD}' \
        'elapse: {elapse:3.3f} min'.format(
            loss=valid_loss,
            portfolio_value=valid_total_portfolio,
            SR=valid_SR,
            MDD=valid_MDD,
            elapse=(time.time() - start) / 60))

def prepare_dataloaders(eod_data, gt_data, args):
    # ========= Preparing DataLoader =========#
    EOD, GT = [], []
    for i in range(eod_data.shape[1] - args.length):
        eod, gt = get_batch(eod_data, gt_data, i, args.length)
        EOD.append(eod)
        GT.append(gt)

    test_eod, test_gt = EOD[args.valid_index:], GT[args.valid_index:]

    test_eod =  torch.FloatTensor(test_eod)
    test_gt = torch.FloatTensor(test_gt)
    test_gt = test_gt.unsqueeze(2)
    test_dataset = Data.TensorDataset(test_eod, test_gt)
    test_loader = Data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, drop_last=True)

    return test_loader






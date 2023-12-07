import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import time
import pickle
from tqdm import tqdm, trange
from utils import *
from torch_geometric.utils import to_dense_adj
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from torch_geometric.data import Data
# torch.multiprocessing.set_start_method("spawn")

def train_smoothing_model(model,dataset,optimizer,args):
    endure_count = 0
    best_acc_val = 0
    adj, features, labels, idx_train, idx_val, idx_test = dataset
    for epoch in range(args.epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        output = model.forward_perturb(features, adj)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()
        if epoch %10==0:
            model.eval()
            output = model.forward_perturb(features, adj)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = accuracy(output[idx_val], labels[idx_val])
            print('Epoch: {:04d}'.format(epoch + 1),
                  'loss_train: {:.4f}'.format(loss_train.item()),
                  'acc_train: {:.4f}'.format(acc_train.item()),
                  'loss_val: {:.4f}'.format(loss_val.item()),
                  'acc_val: {:.4f}'.format(acc_val.item()),
                  'time: {:.4f}s'.format(time.time() - t))
            if  acc_val > best_acc_val:
                best_acc_val = acc_val
                endure_count = 0
                if args.save_model==True:
                    torch.save(model, args.model_dir)
                    print('model saved to:',args.model_dir)
            else:
                endure_count += 1
            if endure_count > args.patience:
                print('early stop at epoch:',epoch)
                break


def train_N_models(model,dataset,optimizer,args,save_dir_inc,save_dir_exc):
    best_acc_val = 0
    adj, features, labels, idx_train, idx_val, idx_test = dataset
    n=features.shape[0]
    adj_dense = torch.squeeze(to_dense_adj(adj))
    counts_inc = np.zeros((features.shape[0], model.nclass), dtype=int)
    counts_exc = np.zeros((features.shape[0], model.nclass), dtype=int)

    print('Preparaing smoothing samples:')
    Data_list=[]
    for repeat_time in tqdm(range(int(args.n_smoothing))):
        # sampling the smoothing distribution.
        adj_dense_pert = model.perturbation(adj_dense)
        # delete the isolated/singleton nodes in training set
        deleted_nodes = torch.where(torch.sum(adj_dense_pert, dim=0) == 0)[0].cpu().numpy().tolist()
        keep_index = set(range(n)) - set(deleted_nodes)
        sub_train_index = torch.tensor(list(keep_index.intersection(idx_train)))
        edge_index_pert = torch.nonzero(adj_dense_pert).t()
        if sub_train_index.shape[0]>0:
            Data_list.append(Data(x=features, edge_index=edge_index_pert,train_index=sub_train_index,keep_index=list(keep_index),idx_val=idx_val))
        else: # do it against
            # sampling the smoothing distribution.
            adj_dense_pert = model.perturbation(adj_dense)
            # delete the isolated/singleton nodes in training set
            deleted_nodes = torch.where(torch.sum(adj_dense_pert, dim=0) == 0)[0].cpu().numpy().tolist()
            keep_index = set(range(n)) - set(deleted_nodes)
            sub_train_index = torch.tensor(list(keep_index.intersection(idx_train)))
            edge_index_pert = torch.nonzero(adj_dense_pert).t()
            if sub_train_index.shape[0] > 0:
                Data_list.append(Data(x=features, edge_index=edge_index_pert, train_index=sub_train_index,
                                      keep_index=list(keep_index), idx_val=idx_val))

    # dataloader
    batch_loader = DataLoader(Data_list, batch_size=1, shuffle=False)
    for i,batch in enumerate(tqdm(batch_loader)):
        # train the model
        for epoch in range(args.epochs):
            t = time.time()
            model.train()
            optimizer.zero_grad()
            output = model.forward(batch.x, batch.edge_index)
            loss_train = F.nll_loss(output[batch.train_index], labels[batch.train_index])
            acc_train = accuracy(output[batch.train_index], labels[batch.train_index])
            loss_train.backward()
            optimizer.step()

            if i<10 and epoch % 99 == 0 and epoch>0:
                model.eval()
                output = model.forward(batch.x, batch.edge_index)
                loss_val = F.nll_loss(output[batch.idx_val], labels[batch.idx_val])
                acc_val = accuracy(output[batch.idx_val], labels[batch.idx_val])
                print('Model: {:01d}'.format(i + 1),
                      'epoch: {:04d}'.format(epoch + 1),
                      'loss_train: {:.4f}'.format(loss_train.item()),
                      'acc_train: {:.4f}'.format(acc_train.item()),
                      'loss_val: {:.4f}'.format(loss_val.item()),
                      'acc_val: {:.4f}'.format(acc_val.item()),
                      'time: {:.4f}s'.format(time.time() - t))


        predictions = model.forward(batch.x, batch.edge_index).argmax(1)
        # if args.singleton == 'exclude':
        predictions_exc=predictions[batch.keep_index]
        counts_exc[batch.keep_index,:] += count_arr(predictions_exc.cpu().numpy(), model.nclass)
        # if args.singleton == 'include':
        counts_inc += count_arr(predictions.cpu().numpy(), model.nclass)
        model.reset_parameters()

    top2_exc = counts_exc.argsort()[:, ::-1][:, :2]
    count1_exc = [counts_exc[n, idx] for n, idx in enumerate(top2_exc[:, 0])]
    count2_exc = [counts_exc[n, idx] for n, idx in enumerate(top2_exc[:, 1])]
    f = open(save_dir_exc, 'wb')
    pickle.dump([top2_exc, count1_exc, count2_exc, counts_exc], f)
    f.close()
    print(f'Save result to {save_dir_exc}')

    top2_inc = counts_inc.argsort()[:, ::-1][:, :2]
    count1_inc = [counts_inc[n, idx] for n, idx in enumerate(top2_inc[:, 0])]
    count2_inc = [counts_inc[n, idx] for n, idx in enumerate(top2_inc[:, 1])]

    f = open(save_dir_inc, 'wb')
    pickle.dump([top2_inc, count1_inc, count2_inc, counts_inc], f)
    f.close()
    print(f'Save result to {save_dir_inc}')

    if args.singleton == 'exclude':
        return top2_exc, count1_exc, count2_exc, counts_exc
    else:
        return top2_inc, count1_inc, count2_inc, counts_inc



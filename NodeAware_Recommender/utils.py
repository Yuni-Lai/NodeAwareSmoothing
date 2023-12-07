import numpy as np
import torch
import torch.nn as nn
import os
import scipy.sparse as sp
import pandas as pd
from scipy.stats import bernoulli
import warnings

def init_random_seed(SEED=2021):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    os.environ['PYTHONHASHSEED'] = str(SEED)
    warnings.filterwarnings("ignore")

def get_device(gpuID):
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpuID}')
        print(f"---using GPU---cuda:{gpuID}----")
    else:
        print("---using CPU---")
        device = torch.device("cpu")
    return device

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def count_arr(predictions, nclass):
    nodes_n=predictions.shape[0]
    counts = np.zeros((nodes_n,nclass), dtype=int)
    for n,idx in enumerate(predictions):
        counts[n,idx] += 1
    return counts

def listSubset(A_list,index_list):
    '''take out the elements of a list (A_list) by a index list (index_list)'''
    return [A_list[i] for i in index_list]


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def save_test(test,filename):
    test.to_csv(filename, index=False)
    return


def save_aggregation(frequency_aggregation,filename):
    print("[", end="", file=open(filename, "a"))
    for i in range(len(frequency_aggregation)):
        current_dict = frequency_aggregation[i]
        for j in range(len(list(current_dict.keys()))):
            key = list(current_dict.keys())[j]
            p = [key, current_dict[key]]
            print("(" + str(p[0]) + "," + str(p[1]) + ")", end="", file=open(filename, "a"))
            if j != len(list(current_dict.keys())) - 1:
                print(";", end="", file=open(filename, "a")),
        print("]", file=open(filename, "a"))
    return


def read_frequency_aggregation(root_dir,filename):
    file = root_dir+filename
    with open(file) as f:
        temp = f.readlines()
    frequency_aggregation = []
    for i in range(len(temp)):
        frequency_aggregation.append(process_point(temp[i]))
    return frequency_aggregation

def read_test(root_dir,filename):
    file = root_dir + filename
    return pd.read_csv(file)


def drop(text, char_lst):
    buffer = ""
    for i in range(len(text)):
        if text[i] not in char_lst:
            buffer += text[i]
    return buffer

def process_point(text):
    text = drop(text, ['[', ']', '(', ')', '\n'])
    raw = text.split(';')
    points = {}
    for i in range(len(raw)):
        if raw[i] != '':
            point = raw[i].split(',')
            points[int(point[0])] = int(point[1])
    return points

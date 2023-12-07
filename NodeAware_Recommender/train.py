import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import time
from tqdm import tqdm, trange
from utils import *
from torch_geometric.utils import to_dense_adj
import cornac
from reco_utils.common.python_utils import binarize
from reco_utils.common.timer import Timer
from reco_utils.dataset import movielens
from reco_utils.evaluation.python_evaluation import map_at_k, ndcg_at_k, precision_at_k, recall_at_k
from reco_utils.recommender.cornac.cornac_utils import predict_ranking
from reco_utils.dataset.python_splitters import python_stratified_split
from reco_utils.evaluation.python_evaluation import (
    map_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    rmse,
    mae,
    logloss,
    rsquared,
    exp_var
)
from reco_utils.common.constants import (
    COL_DICT,
    DEFAULT_K,
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_PREDICTION_COL,
    DEFAULT_TIMESTAMP_COL,
    SEED,
)
from reco_utils.recommender.sar import SAR

def sampling(data_after_split, user_num, args):
    # randomly delete some users:
    sampled_users_indices = np.array(range(user_num))[bernoulli.rvs([1-args.p_n]*user_num)>0]
    train_set = data_after_split[sampled_users_indices[0]]['train']
    test_set = data_after_split[sampled_users_indices[0]]['test']
    for index in sampled_users_indices:
        if index != sampled_users_indices[0]:
            train_set = train_set.append(data_after_split[index]['train'], ignore_index=True)
            test_set = test_set.append(data_after_split[index]['test'], ignore_index=True)
    # randomly delete some ratings:
    train_set = train_set[bernoulli.rvs([1-args.p_e]*train_set.shape[0])>0]
    # remove users that do not have training edges:
    users_train = list(set(train_set['userID']))
    test_set = test_set.loc[test_set['userID'].isin(users_train)]
    return train_set, test_set

def get_degree(data_after_split):
    user_num=range(data_after_split.__len__())
    degree_list=[]
    for u in user_num:
        train_set = data_after_split[u]['train']
        # test_set = data_after_split[u]['test']
        degree_list.append(train_set.shape[0])
    return degree_list

def train_N_models(data_after_split,user_num,args):
    frequency_aggregation = [{} for i in range(user_num)]
    print('Training N models from smoothing samples:')
    for i in tqdm(range(args.n_smoothing)):
        # smoothing sampling and get the train set and test set
        train_set, test_set = sampling(data_after_split, user_num, args)
        model = train_model(args.model, train_set)
        top_k = predict(model, test_set, train_set, args.K_prime, args.model, user_num)
        # store the recommended items into frequency_aggregation
        for j in range(len(top_k)):
            # remember user position = user id - 1
            current_user = top_k['userID'][j]
            current_item = top_k['itemID'][j]
            if current_item in frequency_aggregation[current_user - 1]:
                frequency_aggregation[current_user - 1][current_item] += 1
            else:
                frequency_aggregation[current_user - 1][current_item] = 1

    test = data_after_split[0]['test']
    for i in tqdm(range(1, user_num)):
        test = test.append(data_after_split[i]['test'], ignore_index=True)

    test_name = args.output_dir + '/test.csv'
    save_test(test, test_name)

    aggregation_name =  args.output_dir + '/frequency_aggregation.txt'
    save_aggregation(frequency_aggregation, aggregation_name)
    print('Complete smoothing model training')
    return frequency_aggregation, test


# Utils and constants for BPR
EPOCHS = 15
def prepare_training_bpr(train):
    return cornac.data.Dataset.from_uir(
        train.itertuples(index=False), seed=SEED
    )

def train_bpr(params, data):
    model = cornac.models.BPR(**params)
    with Timer() as t:
        model.fit(data)
    return model, t

def prepare_metrics_fastai(train, test):
    data = test.copy()
    data[DEFAULT_USER_COL] = data[DEFAULT_USER_COL].astype("str")
    data[DEFAULT_ITEM_COL] = data[DEFAULT_ITEM_COL].astype("str")
    return train, data


def recommend_k_bpr(model, test, train):
    with Timer() as t:
        topk_scores = predict_ranking(
            model,
            train,
            usercol=DEFAULT_USER_COL,
            itemcol=DEFAULT_ITEM_COL,
            predcol=DEFAULT_PREDICTION_COL,
            remove_seen=True,
        )
    return topk_scores, t


def ranking_metrics_python(test, predictions, k=DEFAULT_K):
    return {
        "MAP": map_at_k(test, predictions, k=k, **COL_DICT),
        "nDCG@k": ndcg_at_k(test, predictions, k=k, **COL_DICT),
        "Precision@k": precision_at_k(test, predictions, k=k, **COL_DICT),
        "Recall@k": recall_at_k(test, predictions, k=k, **COL_DICT),
    }


prepare_training_data = {
    "bpr": prepare_training_bpr,
}

bpr_params = {
    "k": 200,
    "max_iter": EPOCHS,
    "learning_rate": 0.075,
    "lambda_reg": 1e-3,
    "seed": SEED,
    "verbose": False
}

params = {
    "bpr": bpr_params,
}

trainer = {
    "bpr": lambda params, data: train_bpr(params, data),
}

prepare_metrics_data = {
    "fastai": lambda train, test: prepare_metrics_fastai(train, test),
}

ranking_predictor = {
    "bpr": lambda model, test, train: recommend_k_bpr(model, test, train),
}

ranking_evaluator = {
    "bpr": lambda test, predictions, k: ranking_metrics_python(test, predictions, k),
}

def train_model(algorithm, train_set):
    if algorithm == 'sar':
        model = SAR(
            col_user="userID",
            col_item="itemID",
            col_rating="rating",
            col_timestamp="timestamp",
            similarity_type="jaccard",
            time_decay_coefficient=30,
            timedecay_formula=True
        )
        model.fit(train_set)

        return model
    elif algorithm == 'bpr':
        train = prepare_training_data.get(algorithm, lambda x: x)(train_set)
        model_params = params[algorithm]
        model, time_train = trainer[algorithm](model_params, train)
        return model
    else:
        raise NotImplementedError

def predict(model, test_set, train_set, k, algorithm, user_num):
    if algorithm == 'sar':
        top_k = model.recommend_k_items(test_set, top_k=k, remove_seen=True)
        return top_k
    elif algorithm == 'bpr':
        top_k_scores, time_ranking = ranking_predictor[algorithm](model, test_set, train_set)
        top_k = get_k_recommended_items(top_k_scores, k, user_num)
        return top_k
    else:
        raise NotImplementedError


def get_k_recommended_items(top_k_scores, k, user_num):
    result_user = []
    result_item = []

    for userID in range(1, user_num+1):
        if not top_k_scores[top_k_scores['userID'] == userID].empty:
            itemIDs = list(top_k_scores[top_k_scores['userID'] == userID].sort_values('prediction')[-k:]['itemID'])
            for itemID in itemIDs:
                result_user.append(userID)
                result_item.append(itemID)

    return pd.DataFrame({'userID': result_user, 'itemID': result_item})


import os
import numpy as np
import time
from tqdm import tqdm, trange
import concurrent.futures
import pickle
import torch
import torch.optim as optim
from datetime import datetime
from utils import *
from models import *
from train import *
from certify import *
import pprint
pp = pprint.PrettyPrinter(depth=4)
import movielens
import warnings
import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from matplotlib import rc
rc('text', usetex=True)
mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
plt.subplots_adjust(left=0, right=0.1, top=0.1, bottom=0)
plt.style.use('classic')
MEDIUM_SIZE = 25
BIGGER_SIZE = 27
plt.rc('font', size=BIGGER_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)
plt.rcParams['legend.title_fontsize'] = BIGGER_SIZE

# Model Settings=======================================
parser = argparse.ArgumentParser(description='certify GNN node injecttion')
parser.add_argument('-seed', type=int, default=2020)
parser.add_argument('-model', type=str, default='sar',choices=['sar', 'bpr'], help='Graph-based Recommendation model')
# parser.add_argument('-n_hidden', type=int, default=64, help='size of hidden layer')
parser.add_argument('-K_prime', type=int, default=1, help='top-K recommendation for base classifier')
parser.add_argument('-K', type=int, default=30, help='top-K recommendation for smoothed classifier')
parser.add_argument('-force_training', action='store_true', default=False,help="force training even if pretrained model exist")
# Certify setting----------------------
parser.add_argument('-certify_mode', type=str, default='poisoning',
                    choices=['poisoning'], help="perturbation phrase")
parser.add_argument('-p_e', type=float, default=0.5, help='probability of deleting edges')
parser.add_argument('-p_n', type=float, default=0.9, help='probability of deleting nodes')
parser.add_argument('-n_smoothing', type=int, default=100000, help='number of smoothing samples evalute (N)')
parser.add_argument('-conf_alpha', type=float, default=0.01, help='confident alpha for statistic testing')
parser.add_argument('-degree_budget',type=int, default=1, help='number of edges per malicious node can inject (tau)')
parser.add_argument('-strategy', type=str, default='1')
# parser.add_argument('-tag', type=str, default='85_train')
parser.add_argument('-train_split', type=int, default=75, help='percentage of training data split')
# Dir setting--------------------------
parser.add_argument('-dataset', type=str, default='movielens')
parser.add_argument('-datasize', type=str, default='100k', choices=['100k', '1m'])
parser.add_argument('-output_dir', type=str, default='')
parser.add_argument('-tag', type=str, default='_robust',choices=['','_robust'],help="#evaluate the robustness instead of precision/recall")
args = parser.parse_args()
# Others------------------
init_random_seed(args.seed)

#smoothing samples config
sample_config = {'p_e': args.p_e, 'p_n':args.p_n}
args.output_dir = f'./results_{args.dataset}_{args.datasize}/{args.certify_mode}_mode/{sample_config["p_e"]}_{sample_config["p_n"]}_{args.K_prime}_{args.n_smoothing}_{args.train_split}/'
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
pp.pprint(sample_config)
pp.pprint(args)
# =======================================================
## Load load dataset
print('Loading dataset:')
data = movielens.load_pandas_df(
    size=args.datasize,local_cache_path='../Data/MovieLens-100k')
# Convert the float precision to 32-bit in order to reduce memory consumption
data['rating'] = data['rating'].astype(np.float32)
if args.datasize == '1m':
    user_num = 6040
    item_num = 3952
elif args.datasize == '100k':
    user_num = 943
    item_num = 1682
else:
    raise ValueError

print("Spliting the dataset by user:")
train_ratio=args.train_split/100
data_after_split = movielens.split_by_user(data, "userID", user_num, args.model,train_ratio)
user_degree=get_degree(data_after_split)
## Train model with smoothing sampling
if args.certify_mode=='evasion':
    raise ValueError
else:
    if not os.path.exists(f'{args.output_dir}/frequency_aggregation.txt') or args.force_training:
        frequency_aggregation,test = train_N_models(data_after_split, user_num, args)
    else:
        frequency_aggregation = read_frequency_aggregation(args.output_dir,'frequency_aggregation.txt')
        test = read_test(args.output_dir, 'test.csv')


# Certify
max_rho=60
precision_for_plot = []
recall_for_plot = []
n = user_num
V = range(1, item_num + 1)
p_e = sample_config['p_e']
p_n = sample_config['p_n']
tau = args.degree_budget
certified_precision_array = np.zeros([user_num, max_rho],dtype=np.double)
certified_recall_array = np.zeros([user_num, max_rho],dtype=np.double)

print("bound dictionary construction starts...")
bound_dictionary = construct_bound_dictionary(args.n_smoothing, user_num, item_num, args.conf_alpha)
print("bound dictionary construction ends...")

for user in range(1, user_num + 1):
    # ground truth
    print('user:', user)
    start = datetime.now()

    if args.tag=='_robust':#evaluate the robustness
        I_u = prediction_for_user(user, frequency_aggregation, item_num,args.K)
    else:
        I_u = get_ground_truth(test, user)  # items that the users have bought is the ground truth
    # dictionaries: {itemID: lower_bound/upper_bound}
    lower_bounds, upper_bounds, combined_bounds = compute_bound_for_user(user, frequency_aggregation, item_num,
                                                                         bound_dictionary, I_u)
    # In a decsending order
    lower_bounds, upper_bounds, combined_bounds = sorted_bounds(lower_bounds, upper_bounds, combined_bounds)

    V_minus_I_u = []
    for v in V:
        if v not in I_u:
            V_minus_I_u.append(v)
    p_0=p_n+(1-p_n)*np.power(p_e, user_degree[user-1])
    test_items_num = len(I_u)
    for rho in range(max_rho):
        # p_tilde = np.power(np.power(args.p_e, args.degree_budget) + args.p_n - np.power(args.p_e,args.degree_budget)*args.p_n, rho)
        #p_tilde = np.power((1 - p_n) * np.power(p_e + p_n - p_e * p_n, tau) + p_n, rho)
        p_tilde = np.power((1 - p_n) * np.power(p_e, tau) + p_n, rho)
        for r in range(test_items_num):
            item_r = list(lower_bounds.keys())[r]# the top rth item among I_u
            if get_ranking(item_r, combined_bounds) > args.K: # the current r must can not be certified
                break
            lhs = p_tilde*lower_bounds[item_r]

            #rhs 1
            #C_mu_length = args.K - get_ranking(item_r, combined_bounds) + 1
            C_mu_length = args.K - r + 1
            rhs_1_min = math.inf

            if args.strategy == '1':
                C_mu = list(upper_bounds.keys())[:args.K - r + 1] # the top (K - r + 1) items among I\I_u
                for c in range(1,C_mu_length):
                    C_mu_temp=C_mu[-c:]
                    rhs_1 = p_tilde*sum_upper_bounds_in_C_mu(C_mu_temp, upper_bounds)+args.K_prime*(1-p_tilde)*(1-p_0)
                    rhs_1 = rhs_1 / c
                    # print(c,'--',rhs_1)
                    rhs_1_min = min(rhs_1_min, rhs_1)

            elif args.strategy=='2': #pore
                C_mu_length = args.K - get_ranking(item_r, combined_bounds) + 1
                C_mu_temp = get_C_mu(I_u, get_I_mu(V_minus_I_u, item_r, combined_bounds), upper_bounds, C_mu_length)
                rhs_1 = p_tilde*sum_upper_bounds_in_C_mu(C_mu_temp, upper_bounds)+args.K_prime*(1-p_tilde)*(1-p_0)
                rhs_1 = rhs_1 / C_mu_length
                rhs_1_min = rhs_1

            # rhs 2
            item_K_r_1 = list(upper_bounds.keys())[args.K - r + 1] # the top (K - r + 1)th item among I\I_u
            rhs_2 = p_tilde*upper_bounds[item_K_r_1]+(1-p_tilde)*(1-args.p_n)
            rhs = min(rhs_1_min, rhs_2)

            if lhs <= rhs:
                # the current r can not be certified
                break
        certified_precision_array[user - 1, rho] = r / float(args.K)
        certified_recall_array[user - 1, rho] = r / float(test_items_num)

for rho in range(max_rho):
    precision_for_plot.append(np.mean(certified_precision_array[:,rho]))
    recall_for_plot.append(np.mean(certified_recall_array[:,rho]))

print(precision_for_plot)
print(recall_for_plot)


df = {'rho': range(max_rho), 'certified precision': precision_for_plot,'certified recall':recall_for_plot}
f = open(f'{args.output_dir}/certify_result_tau{args.degree_budget}_K{args.K}{args.tag}.pkl', 'wb')
pickle.dump(df, f)
f.close()
print(f'Save result to {args.output_dir}/certify_result_tau{args.degree_budget}_K{args.K}{args.tag}.pkl')

df=pd.DataFrame(df)
ACR_recall = sum([rho * (df.at[rho, 'certified recall'] - df.at[rho + 1, 'certified recall']) for rho in range(max_rho - 1)])
ACR_precision = sum([rho * (df.at[rho, 'certified precision'] - df.at[rho + 1, 'certified precision']) for rho in range(max_rho - 1)])
print(ACR_recall)
print(ACR_precision)


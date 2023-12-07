import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pprint
import time
from tqdm import tqdm, trange
import concurrent.futures
pp = pprint.PrettyPrinter(depth=4)
import pickle
import torch
import torch.nn.functional as F
import torch.optim as optim
from scipy.stats import norm, binom_test
from statsmodels.stats.proportion import proportion_confint
from scipy.stats import ncx2
from utils import *
from models import *
from train import *
from certify import *
import warnings
warnings.filterwarnings("ignore")
import argparse
import matplotlib as mpl
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

# Model Settings=======================================
parser = argparse.ArgumentParser(description='certify GNN node injecttion')
parser.add_argument('-gpuID', type=int, default=8)
parser.add_argument('-seed', type=int, default=2020)
parser.add_argument('-lr', type=float, default=0.001, help='learning rate')
parser.add_argument('-patience', type=int, default=30, help='patience for early stopping')
parser.add_argument('-epochs', type=int, default=1000, help='training epoch')
# parser.add_argument('-epoch', type=int, default=100, help='training epoch')
parser.add_argument('-save_model', action='store_true', default=True,help="save model")
parser.add_argument('-model', type=str, default='GCN',choices=['GCN', 'GAT','APPNP'], help='GNN model')
parser.add_argument('-n_hidden', type=int, default=64, help='size of hidden layer')
parser.add_argument('-drop', type=float, default=0.5, help='dropout rate')
parser.add_argument('-weight_decay', type=float, default=1e-3, help='weight_decay rate')
parser.add_argument('-n_per_class', type=int, default=50, help='sample numebr per class')
parser.add_argument('-force_training', action='store_true', default=False,help="force training even if pretrained model exist")
# Certify setting----------------------
parser.add_argument('-certify_mode', type=str, default='poisoning',
                    choices=['evasion', 'poisoning'])
parser.add_argument('-singleton', type=str, default='exclude',
                    choices=['include', 'exclude'],help='include or exclude singleton node in voting')
parser.add_argument('-p_e', type=float, default=0.8, help='probability of deleting edges')
parser.add_argument('-p_n', type=float, default=0.9, help='probability of deleting nodes')
parser.add_argument('-n_smoothing', type=int, default=10000, help='number of smoothing samples evalute (N)')
parser.add_argument('-conf_alpha', type=float, default=0.01, help='confident alpha for statistic testing')
parser.add_argument('-degree_budget',type=int, default=5, help='number of edges per malicious node can inject (tau)')
# Dir setting-------------------------
parser.add_argument('-dataset', type=str, default='citeseer', choices=['cora', 'citeseer','pubmed'])
parser.add_argument('-output_dir', type=str, default='')
args = parser.parse_args()
# Others------------------
if torch.cuda.is_available():
    args.device = torch.device(f'cuda:{args.gpuID}')
    print(f"---using GPU---cuda:{args.gpuID}----")
else:
    print("---using CPU---")
    args.device = torch.device("cpu")
init_random_seed(args.seed)

if args.certify_mode=='poisoning':
    args.epochs=100
    # args.n_smoothing=1000

else:
    args.singleton = 'include'

if args.dataset == "cora":
    args.data_dir = "../Data/cora_ml.npz"
elif args.dataset == "citeseer":
    args.data_dir = "../Data/citeseer.npz"
elif args.dataset == "pubmed":
    args.data_dir = "../Data/pubmed.npz"
#smoothing samples config
sample_config = {'p_e': args.p_e, 'p_n':args.p_n}
args.output_dir = f'./results_{args.dataset}/{args.certify_mode}_mode_{args.singleton}/{sample_config["p_e"]}_{sample_config["p_n"]}_{args.n_smoothing}/'
args.model_dir=f'{args.output_dir}/{args.model}_{sample_config["p_e"]}_{sample_config["p_n"]}.pth'

if args.certify_mode == 'poisoning':
    if not os.path.exists(f'./results_{args.dataset}/{args.certify_mode}_mode_exclude/{sample_config["p_e"]}_{sample_config["p_n"]}_{args.n_smoothing}/'):
        os.makedirs(f'./results_{args.dataset}/{args.certify_mode}_mode_exclude/{sample_config["p_e"]}_{sample_config["p_n"]}_{args.n_smoothing}/')
    if not os.path.exists(f'./results_{args.dataset}/{args.certify_mode}_mode_include/{sample_config["p_e"]}_{sample_config["p_n"]}_{args.n_smoothing}/'):
        os.makedirs(f'./results_{args.dataset}/{args.certify_mode}_mode_include/{sample_config["p_e"]}_{sample_config["p_n"]}_{args.n_smoothing}/')
else:
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
# =======================================================

## Load load dataset
# pp.pprint(sample_config)
adj, features, labels, n, d, nc = load_data(args.data_dir)
idx_train, idx_val, idx_test = split(labels=labels, n_per_class=args.n_per_class, seed=args.seed)
# np.savetxt(f'./results_{args.dataset}/idx_train.txt',idx_train)
# np.savetxt(f'./results_{args.dataset}/idx_val.txt',idx_val)
# np.savetxt(f'./results_{args.dataset}/idx_test.txt',idx_test)
adj = torch.LongTensor(np.stack(adj.nonzero())).to(args.device)
features = torch.Tensor(features).to(args.device)
labels = torch.LongTensor(labels).to(args.device)
dataset = [adj, features, labels, idx_train, idx_val, idx_test]
node_degrees = get_degrees(adj)
if args.model == 'GCN':
    model = SmoothGCN(in_channels=d, out_channels=nc, hidden_channels=args.n_hidden, dropout=args.drop,config=sample_config,device=args.device).to(args.device)
elif args.model == 'GAT':
    # divide the number of hidden units by the number of heads to match the overall number of paramters
    model = GAT(in_channels=d, out_channels=nc, hidden_channels=args.n_hidden // 8,
                k_heads=8, dropout=args.drop,config=sample_config,device=args.device).to(args.device)
elif args.model == 'APPNP':
    model = APPNPNet(n_features=d, n_classes=nc, n_hidden=args.n_hidden,
                     k_hops=10, alpha=0.15, p_dropout=args.drop).to(args.device)
else:
    raise ValueError(f"choices=['GCN', 'GAT','APPNP']")

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

if args.certify_mode=='evasion':
    ## Training the model with smoothing perturbation samples
    if not os.path.exists(args.model_dir) or args.force_training:
        train_smoothing_model(model,dataset,optimizer,args)
    else:
        model = torch.load(args.model_dir)
        model.to(args.device)

    # Smoothing Testing
    if not os.path.exists(f'{args.output_dir}/smoothing_result.pkl') or args.force_training:
        model.eval()
        top2, count1, count2 = model.smoothed_precit(features, adj, num=args.n_smoothing)
        f=open(f'{args.output_dir}/smoothing_result.pkl','wb')
        pickle.dump([top2, count1, count2],f)
        f.close()
        print(f'Save result to {args.output_dir}/smoothing_result.pkl')
    else:
        f=open(f'{args.output_dir}/smoothing_result.pkl','rb')
        top2, count1, count2=pickle.load(f)
        f.close()

elif args.certify_mode == 'poisoning':
    ## Training N models with smoothing perturbation samples
    if not os.path.exists(f'{args.output_dir}/smoothing_result.pkl') or args.force_training:
        save_dir_inc = f'./results_{args.dataset}/{args.certify_mode}_mode_include/{sample_config["p_e"]}_{sample_config["p_n"]}_{args.n_smoothing}/smoothing_result.pkl'
        save_dir_exc = f'./results_{args.dataset}/{args.certify_mode}_mode_exclude/{sample_config["p_e"]}_{sample_config["p_n"]}_{args.n_smoothing}/smoothing_result.pkl'
        top2, count1, count2, counts = train_N_models(model, dataset, optimizer, args, save_dir_inc, save_dir_exc)

    else:
        f = open(f'{args.output_dir}/smoothing_result.pkl', 'rb')
        top2, count1, count2, counts = pickle.load(f)
        f.close()
    total_votes=counts.sum(1)


test_label=labels.cpu().numpy()[idx_test]
correct = (np.array(top2[idx_test,0]) == test_label)
print("Smoothed classfier accuracy:",np.sum(correct)/len(idx_test))

# Certify
def certify_process(rho):
    if args.singleton == 'include':
        cAHat,certified = certify(rho, top2[idx_test], listSubset(count1,idx_test), listSubset(count2,idx_test),sample_config,model.nclass, args)
    elif args.singleton == 'exclude':
        cAHat,certified = certify(rho, top2[idx_test], listSubset(count1,idx_test), listSubset(count2,idx_test),sample_config,model.nclass,args,node_degrees[idx_test],total_votes[idx_test])

    certified_and_correct=np.array(certified) & correct
    certified_accuracy=np.sum(certified_and_correct)/len(idx_test)
    return certified_accuracy,cAHat,certified

def run(certify_process, rho_list):
    with concurrent.futures.ProcessPoolExecutor(max_workers = 20) as executor:
        results = list(tqdm(executor.map(certify_process, rho_list), total=len(rho_list)))
    return results

max_rho = 160
rho_list=[*range(1,max_rho+1,1)]
certi_acc_list=[]
if False:#args.p_e ==0 or args.p_n==0: # single process
    zero_cer=False
    for rho in tqdm(rho_list, desc='Processing certified radius'):
        if not zero_cer:
            certified_accuracy,cAHat,_=certify_process(rho)
            certi_acc_list.append(certified_accuracy)
            if certified_accuracy==0.0:
                zero_cer=True # Then, do not need further verify for larger radius.
        else:
            certi_acc_list.append(0.0)
else: # multi-threat
    #rho_list = [*range(0, max_rho+1, 20)][1:]
    print('multi-process mode')
    results=run(certify_process, rho_list)
    for res in results:
        certi_acc_list.append(res[0])
        cAHat=res[1]


print('The ratio of ABSTAIN:',np.sum([True if ca==-1 else False for ca in cAHat])/len(cAHat))
df = {'rho': [0]+rho_list, 'certified accuracy': [np.sum(correct)/len(idx_test)]+certi_acc_list}
f = open(f'{args.output_dir}/certify_result_tau{args.degree_budget}.pkl', 'wb')
pickle.dump(df, f)
f.close()
print(f'Save result to {args.output_dir}/certify_result_tau{args.degree_budget}.pkl')

plt.figure(constrained_layout=True)
sns.lineplot(x="rho", y="certified accuracy", data=df)
plt.xlabel(r'$\rho$')
plt.title(fr"$p_e$={sample_config['p_e']},$p_n$={sample_config['p_n']},$\tau$={args.degree_budget}")
plt.savefig(args.output_dir + f'/{sample_config["p_e"]}_{sample_config["p_n"]}_{args.degree_budget}_{args.n_smoothing}_certify_curve.pdf', dpi=300)
print(f'Save result to {args.output_dir}/{sample_config["p_e"]}_{sample_config["p_n"]}_{args.degree_budget}_{args.n_smoothing}_certify_curve.pdf')
plt.show()

if False:
    if args.singleton == 'include':
        yA_list,pA_list,pB_list=get_pA_pB(top2[idx_test], listSubset(count1,idx_test), listSubset(count2,idx_test),sample_config,model.nclass,args)
    elif args.singleton == 'exclude':
        yA_list,pA_list,pB_list=get_pA_pB(top2[idx_test], listSubset(count1,idx_test), listSubset(count2,idx_test),sample_config,model.nclass,args,total_votes[idx_test])
    df = {'y_A': yA_list, 'p_A': pA_list,'p_B':pB_list}
    f = open(f'{args.output_dir}/certify_analysis.pkl', 'wb')
    pickle.dump(df, f)
    f.close()
    print(f'Save result to {args.output_dir}/certify_analysis.pkl')
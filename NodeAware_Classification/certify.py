import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import time
from tqdm import tqdm, trange
from scipy.stats import norm, binom_test
from statsmodels.stats.proportion import proportion_confint
from scipy.stats import ncx2
from utils import *

def certify(rho, top2, count1, count2, sample_config,nclass,args,node_degrees=None,total_votes=None):
    cAHat_list=[]
    certified_list=[]
    p_e = sample_config['p_e']
    p_n = sample_config['p_n']
    for i in range(len(count1)):
        yAHat = top2[i, 0]
        p_tilde = np.power((1-p_n)*np.power(p_e+p_n-p_e*p_n, args.degree_budget) + p_n, rho)
        if args.singleton == 'include':
            pABar = proportion_confint(count1[i], args.n_smoothing, alpha=2 * args.conf_alpha/nclass, method="beta")[0]
            pBBar = proportion_confint(count2[i], args.n_smoothing, alpha=2 * args.conf_alpha/nclass, method="beta")[1]
            certified = (p_tilde*(pABar-pBBar+1)>1)
        elif args.singleton=='exclude':
            # pABar = proportion_confint(count1[i], total_votes[i], alpha=2 * args.conf_alpha / nclass, method="beta")[0]
            # pBBar = proportion_confint(count2[i], total_votes[i], alpha=2 * args.conf_alpha / nclass, method="beta")[1]
            # certified_1 = (p_tilde * (pABar - pBBar + 1) -1 > 0)
            pABar = proportion_confint(count1[i], args.n_smoothing, alpha=2 * args.conf_alpha / nclass, method="beta")[0]
            pBBar = proportion_confint(count2[i], args.n_smoothing, alpha=2 * args.conf_alpha / nclass, method="beta")[1]
            pe_d = np.power(p_e+p_n-p_e*p_n, node_degrees[i])
            # certified = (p_tilde * (pABar - pBBar) - (1 - (pe_d + args.p_n - pe_d * args.p_n)) * (1 - p_tilde) > 0)
            pe_2d = np.power(p_e+p_n-p_e*p_n, 2 * node_degrees[i])
            # certified = (p_tilde * (pABar - pBBar) - (1-args.p_n)*(1-p_tilde) > 0)
            p_0 = pe_d + p_n - pe_d * p_n
            p_2d = pe_2d + p_n - pe_2d * p_n
            frac = (1 - p_2d) / (1 - p_0)
            certified = (p_tilde * (pABar - frac * pBBar + 1 - p_2d) - 1 + p_2d > 0)
            # certified = certified_1 or certified_2


        s = binom_test(count1[i], count1[i] + count2[i], p=0.5)
        if s < args.conf_alpha:
            cAHat_list.append(yAHat)
            certified_list.append(certified)
        else:
            cAHat_list.append(-1)
            certified_list.append(False)
            continue
    return cAHat_list,certified_list

def get_pA_pB(top2, count1, count2, sample_config,nclass,args,total_votes=None):
    pA_list=[]
    pB_list=[]
    yA_list=[]
    for i in range(len(count1)):
        yAHat = top2[i, 0]
        if args.singleton == 'include':
            pABar = proportion_confint(count1[i], args.n_smoothing, alpha=2 * args.conf_alpha/nclass, method="beta")[0]
            pBBar = proportion_confint(count2[i], args.n_smoothing, alpha=2 * args.conf_alpha/nclass, method="beta")[1]
        elif args.singleton=='exclude':
            pABar = proportion_confint(count1[i], args.n_smoothing, alpha=2 * args.conf_alpha / nclass, method="beta")[0]
            pBBar = proportion_confint(count2[i], args.n_smoothing, alpha=2 * args.conf_alpha / nclass, method="beta")[1]
            # pABar = proportion_confint(count1[i], total_votes[i], alpha=2 * args.conf_alpha / nclass, method="beta")[0]
            # pBBar = proportion_confint(count2[i], total_votes[i], alpha=2 * args.conf_alpha / nclass, method="beta")[1]
        yA_list.append(yAHat)
        pA_list.append(pABar)
        pB_list.append(pBBar)
    return yA_list,pA_list,pB_list



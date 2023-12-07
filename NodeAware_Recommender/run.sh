#!/usr/bin/env bash
#conda activate py37
#cd ~/MyProjects/CertifyNodeInjection/03_RecommendCertify/

#nohup bash run.sh > ./run.log 2>&1 &

#datasize=['100k', '1m']
#mode = ['poisoning']

dataset='movielens'
mode='poisoning'
K_prime=1
### grid search###
for tau in 1 #1 2 4
do
  for datasize in '100k' #'1m'
  do
    for p_n in 0.9
    do
      for p_e in 0.5 0.6 0.7 #0.0 0.1 0.2 #0.5 0.6
      do
        echo "The current program (datasize,p_e,p_n,tau) is: ${datasize},${p_e},${p_n},${tau}"
        nohup python -u main.py -datasize $datasize -p_e $p_e -p_n $p_n -K_prime $K_prime -degree_budget $tau > ./results_${dataset}_${datasize}/${mode}_mode/run_${p_e}_${p_n}.log 2>&1 &
      done
    done
    wait
  done
done

echo "Proccess Finished!"

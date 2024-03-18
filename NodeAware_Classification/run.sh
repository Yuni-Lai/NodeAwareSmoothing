#!/usr/bin/env bash
#conda activate py37

#nohup bash run.sh > ./run.log 2>&1 &

#datset=['cora', 'citeseer', 'pubmed']
#mode = ['evasion', 'poisoning']

mode='evasion'
singleton='include'
### grid search###
#for tau in 2 4 6
#do
#  for p_n in 0.9 0.8 #0.0 0.7 0.8
#  do
#    for p_e in 0.0 0.7 0.8 0.9 #0.1 0.2 #0.7 0.8 0.9 # 0.0 0.1 0.2 #0.3 0.4 0.5 0.6 #0.7 0.8 0.5 0.6 0.9
#      do
#      for dataset in 'cora' 'citeseer'
#      do
#        echo "The current program (dataset,mode,p_e,p_n,tau) is: ${dataset},${mode},${p_e},${p_n},${tau}"
#        nohup python -u main.py -dataset $dataset -p_e $p_e -p_n $p_n -degree_budget $tau -certify_mode $mode -singleton $singleton -gpuID 9 > ./results_${dataset}/${mode}_mode_${singleton}/run_${p_e}_${p_n}.log 2>&1 &
#      done
#      wait
#    done
#  done
#done


#N=1000
#for tau in 5 10 #4 6
#do
#  for dataset in 'citeseer' #'cora'
#  do
#    for p_n in 0.9 #0.0 0.7  0.0 0.7 0.8 0.5 0.6
#    do
#      for p_e in 0.9 #0.1 0.2 0.3 #0.7 0.8 0.9 #0.3 0.4 0.5 0.6
#      do
#        echo "The current program (dataset,mode,p_e,p_n,tau) is: ${dataset},${mode},${p_e},${p_n},${tau}"
#        nohup python -u main.py -dataset $dataset -p_e $p_e -p_n $p_n -n_smoothing $N -degree_budget $tau -certify_mode $mode -singleton $singleton -gpuID 6 > ./results_${dataset}/${mode}_mode_${singleton}/run_${p_e}_${p_n}.log 2>&1 &
#      done
#      wait
#    done
#  done
#done

N=1000
for tau in 5 #4 6
do
  for dataset in 'cora' 'citeseer'
  do
    for p_n in 0.7 0.8 0.9 #0.0 0.7  0.0 0.7 0.8 0.5 0.6
    do
      for p_e in 0.1 0.3 0.5 0.6 0.7 0.8 #0.1 0.2 0.3 #0.7 0.8 0.9 #0.3 0.4 0.5 0.6
      do
        echo "The current program (dataset,mode,p_e,p_n,tau) is: ${dataset},${mode},${p_e},${p_n},${tau}"
        nohup python -u main_empirical.py -dataset $dataset -p_e $p_e -p_n $p_n -n_smoothing $N -degree_budget $tau -certify_mode $mode -singleton $singleton -gpuID 6 > ./results_${dataset}/${mode}_mode_${singleton}/run_${p_e}_${p_n}.log 2>&1 &
      done
      wait
    done
  done
done

echo "Proccess Finished!"

# CertifyNodeInjection
Source code for paper under review:  
Node-aware Bi-smoothing: Certified Robustness against Graph Injection Attacks

## Environment

```bash
conda env create -f py37.yml
conda activate py37
pip install -r py37.txt
```
if report: "ResolvePackageNotFound:xxx", or "No matching distribution found for xxx", just open the .yaml or .txt file and delete that line.


## Node-aware Bi-Smoothing
```bash
nohup bash run.sh > ./run.log 2>&1 &
```

or 
```bash
python main.py -dataset 'citeseer' -p_e 0.8 -p_n 0.9 -n_smoothing 10000 -degree_budget 5 -certify_mode 'poisoning' -singleton 'exclude' -gpuID 0
```

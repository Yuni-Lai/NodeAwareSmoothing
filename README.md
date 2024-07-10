# CertifyNodeInjection
This is the official source codes for the paper:  
Node-aware Bi-smoothing: Certified Robustness against Graph Injection Attacks (S&P2024)

## Environment

```bash
cd ./Environments
conda env create -f py37.yml
conda activate py37
pip install -r py37.txt
```
if report: "ResolvePackageNotFound:xxx", or "No matching distribution found for xxx", just open the .yaml or .txt file and delete that line.


## Node-aware Bi-Smoothing
The directory ./NodeAware_Classification is about general node classification tasks, such as Citeseer, Cora-ML, and PubMed datasets.   
The directory ./NodeAware_Recommender is about the recommendation task.   

```bash
nohup bash run.sh > ./run.log 2>&1 &
```

or 
```bash
python main.py -dataset 'citeseer' -p_e 0.8 -p_n 0.9 -n_smoothing 10000 -degree_budget 5 -certify_mode 'poisoning' -singleton 'exclude' -gpuID 0
```

## Citation
```bash
@article{lai2023node,
  title={Node-aware Bi-smoothing: Certified Robustness against Graph Injection Attacks},
  author={Lai, Yuni and Zhu, Yulin and Pan, Bailin and Zhou, Kai},
  journal={IEEE Symposium on Security and Privacy},
  year={2023}
}
```

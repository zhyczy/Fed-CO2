# Fed-CO2: Cooperation of Online and Offline Models for Severe Data Heterogeneity in Federated Learning
This is the PyTorch implemention.
## Usage
### Setup

We explore data heterogeneity issues in FL with label distribution skew and feature skew.

For experiments with Feature Skew:

Run this basic command: 

python main.py  --mode [algorithm] --log  --dataset  [dataset]  --save_path checkpoint/[dataset]

For experiments with Feature Skew and Label Distribution Skew:

Run this basic command:

python main.py  --mode [algorithm] --log  --dataset  [dataset]  --save_path checkpoint/[dataset] --imbalance_train --beta [beta]  --divide [n]

For experiments with Label Distribution Skew:

Run the following basic commands:

cd label-skew

python experiments.py --model=[backbone] --dataset=[dataset] --alg=[algorithm] 

### Dataset

Benchmark datasets in this work include CIFAR10, CIFAR100, Digits, Office-Caltech10 and DomainNet

For CIFAR10 and CIFAR100 dataset, download and unzip data under 'label-skew/data' file catalog.

For Digits, Office-Caltech10 and DomainNet, put data in 'data' file catalog

### Train
Federated Learning

Refer to run.sh for more details


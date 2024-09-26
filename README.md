# Constructing Adversarial Examples for Vertical Federated Learning: Optimal Client Corruption through Multi-Armed Bandit

This repository contains the implementation of the paper:

[Constructing Adversarial Examples for Vertical Federated Learning: Optimal Client Corruption through Multi-Armed Bandit](https://openreview.net/pdf?id=m52uU0dVbH)

Published as a conference paper at ICLR 2024.

## Authors
- Duanyi Yao (HKUST)
- Songze Li (Southeast University)
- Ye Xue (Shenzhen Research Institute of Big Data, CUHK(SZ))
- Jin Liu (HKUST(GZ))

## Abstract

Vertical federated learning (VFL), where each participating client holds a subset of data features, has found numerous applications in finance, healthcare, and IoT systems. However, adversarial attacks, particularly through the injection of adversarial examples (AEs), pose serious challenges to the security of VFL models. In this paper, we investigate such vulnerabilities through developing a novel attack to disrupt the VFL inference process, under a practical scenario where the adversary is able to adaptively corrupt a subset of clients. We formulate the problem of finding optimal attack strategies as an online optimization problem, which is decomposed into an inner problem of adversarial example generation (AEG) and an outer problem of corruption pattern selection (CPS). Specifically, we establish the equivalence between the formulated CPS problem and a multiarmed bandit (MAB) problem, and propose the Thompson sampling with Empirical maximum reward (E-TS) algorithm for the adversary to efficiently identify the optimal subset of clients for corruption. The key idea of E-TS is to introduce an estimation of the expected maximum reward for each arm, which helps to specify a small set of competitive arms, on which the exploration for the optimal arm is performed. This significantly reduces the exploration space, which otherwise can quickly become prohibitively large as the number of clients increases. We analytically characterize the regret bound of E-TS, and empirically demonstrate its capability of efficiently revealing the optimal corruption pattern with the highest attack success rate, under various datasets of popular VFL tasks.


[Read more](https://openreview.net/pdf?id=m52uU0dVbH)

## Installation

To set up the project environment:

1. Clone the repository
2. Install the required packages:

```bash
cd E-TS-in-VFL
pip install -r requirements.txt
```
## Usage

This implementation is based on the FashionMNIST dataset and the model is designed for 7 clients. If you want to try other datasets and client numbers, please carefully adjust the `model.py` file. For table datasets, use `table_dis.py` for data splitting and adjust your models accordingly.

Example usage:

```bash 
python attack.py --clients_num 7 --config ./config.json --constraint 2 --record_rounds 125 --model_training True --targeted True 
```

## Citation
```bash
@inproceedings{duanyi2023constructing,
  title={Constructing Adversarial Examples for Vertical Federated Learning: Optimal Client Corruption through Multi-Armed Bandit},
  author={Duanyi, YAO and Li, Songze and Ye, XUE and Liu, Jin},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2023}
}
```

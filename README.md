# Multiplex Graph Representation Learning via Dual Correlation Reduction

This repository contains the reference code for the paper Multiplex Graph Representation Learning via Dual Correlation Reduction. 

## Contents

0. [Installation](#installation)
0. [Preparation](#Preparation)
0. [Testing](#test)
0. [Training](#train)

## Installation
pip install -r requirements.txt 

## Preparation

weights see >>>[here](saved_model/)<<<.

Configs see >>>[here](args.yaml)<<<.

Dataset (`--dataset-class`, `--dataset-name`,`--Custom-key`)

Important args:
* `--pretrain` Test checkpoints
* `--dataset-name` acm, imdb
* `--custom_key` Node: node classification  Clu: clustering   Sim: similarity SemiNode: semi-supervised learning
## Training
python main.py

## Testing
Choose the custom_key of different downstream tasks


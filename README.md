# FEPaML
## Overview
`PKNAP` is a suite of methods designed for Physics-based and Knowledge-based Neo-Antigen Prediction. Free Energy Perturbation-assisted Machine Learning (FEPaML) is one of them.

`FEPaML` iteratively utilizes a small amount of precise physics-based FEP data to fine-tune the rough knowledge-based pre-trained ML predictions through Bayesian Optimization.

## Dependencies
- Python (some common Python libraries)
- PyTorch

## Installation
```bash
git clone https://github.com/zhongqinglu/FEPaML.git
```

## Usage
```bash
path=`pwd`
```
### 1. pre-train

```bash
cd $path/example/pretrain
# prepare pre-train dataset
touch train.tsv
# pre-train
../../PKNAP.pretrain --ipfn train.tsv
```
If you want to skip Step 1, `A0201.pretrained.model` is a folder of pre-trained models prepared for Step 2.

### 2. fine-tune
```bash
cd $path/example/finetune
# prepare pre-train dataset
touch train.tsv
# prepare fine-tune dataset
touch ddg.csv
# copy/symlink pre-train hyper-parameters
ln -s ../pretrain/model_parameters.pkl
# or use the provided one
ln -s ../pretrain/A0201.pretrained.model/0/model_parameters.pkl
# copy/symlink pre-trained models (e.g. use the provided one)
ln -s ../pretrain/A0201.pretrained.model pretrain_state_dict_fold
# fine-tune
../../PKNAP.finetune --nsample 3 --stoploss 0.04
```

### 3. predict
```bash
cd $path/example/predict
# copy/symlink model hyper-parameters (e.g. use the provided one)
ln -s ../pretrain/A0201.pretrained.model/0/model_parameters.pkl
# copy/symlink models (e.g. use the provided one)
ln -s ../pretrain/A0201.pretrained.model pretrain_state_dict_fold
```
predict ddG
```bash
# prepare samples to be predicted
touch ddg.csv
# predict
../../PKNAP.predict.ddg --ipfn ddg.csv
```
predict dG
```bash
# prepare samples to be predicted
touch dg.csv
# predict
../../PKNAP.predict.dg --ipfn dg.csv
```
mutate residues based on a given peptide (e.g. HMTEVVRHC)
```bash
../../PKNAP.mutate --mutseq HMTEVVRHC --mutnu 1
```

## Contact
If you have any questions, please contact us at [zhongqinglu@foxmail.com]()


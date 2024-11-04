
path=`pwd`

# 1.pretrain
cd $path/example/pretrain
# prepare pretrain dataset
touch train.tsv
# train
../../PKNAP.pretrain --ipfn train.tsv
# A0201.pretrained.model is a provided fold of pretrained models if you want to skip step 1.

# 2.finetune
cd $path/example/finetune
# prepare pretrain dataset
touch train.tsv
# prepare finetune dataset
touch ddg.csv
# copy/symlink pretrain hyper-parameters
ln -s ../pretrain/model_parameters.pkl
# or use the provided one
ln -s ../pretrain/A0201.pretrained.model/0/model_parameters.pkl
# copy/symlink pretrained models (e.g. use the provided one)
ln -s ../pretrain/A0201.pretrained.model pretrain_state_dict_fold
# finetune
../../PKNAP.finetune --nsample 3 --stoploss 0.04

# 3.predict
cd $path/example/predict
# copy/symlink model hyper-parameters (e.g. use the provided one)
ln -s ../pretrain/A0201.pretrained.model/0/model_parameters.pkl
# copy/symlink models (e.g. use the provided one)
ln -s ../pretrain/A0201.pretrained.model pretrain_state_dict_fold

# if predict ddG
# prepare samples to be predicted
touch ddg.csv
# predict
../../PKNAP.predict.ddg --ipfn ddg.csv

# if predict dG
# prepare samples to be predicted
touch dg.csv
# predict
../../PKNAP.predict.dg --ipfn dg.csv

# if mutate residues based on a given peptide (e.g. HMTEVVRHC)
../../PKNAP.mutate --mutseq HMTEVVRHC --mutnu 1


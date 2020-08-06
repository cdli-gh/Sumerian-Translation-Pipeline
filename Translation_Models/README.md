# Checkpoints for Translation

## Vanilla Transformer (Supervised) 

```
wget https://cdlisumerianunmt.s3.us-east-2.amazonaws.com/Transformer/<data_mode>/_step_14000.pt
````
Where, ```data_mode``` is in ```[AllCompSents, UrIIICompSents, UrIIILineByLine, AllLineByLine]``` and ```AllCompSents``` shows the best results:

```
wget https://cdlisumerianunmt.s3.us-east-2.amazonaws.com/Transformer/AllCompSents/_step_14000.pt
````

## Back Translation (Semi-Supervised; Best BLEU) 

```
wget https://cdlisumerianunmt.s3.us-east-2.amazonaws.com/BackTranslation/<shard_num>st/_step_10000.pt
```
Where, ```shard_num``` is in ```{1,8}``` and ```5``` shows the best results:

```
wget https://cdlisumerianunmt.s3.us-east-2.amazonaws.com/BackTranslation/5st/_step_10000.pt
```

## XLM (Semi-Supervised and Unsupervised Versions)

```
wget https://cdlisumerianunmt.s3.us-east-2.amazonaws.com/XLM/experiments/<exp_name>/<method>/checkpoint.pth
```
Where, ```exp_name``` is in ```[AllData_MLM, RetrictedData_MLM_TLM, MixedData_MLM_TLM, AugmentedData]```, ```method``` is in ```[Supervised, Unsupervised]``` and ```AugmentedData``` with ```Supervised``` shows the best results:

```
wget https://cdlisumerianunmt.s3.us-east-2.amazonaws.com/XLM/experiments/SumEn_RetrictedData_MLM_TLM/Supervised/checkpoint.pth
```

## MASS (Semi-Supervised and Unsupervised Versions) 

```
wget https://cdlisumerianunmt.s3.us-east-2.amazonaws.com/MASS/experiments/<exp_name>/PhaseTwo/checkpoint.pth
```
Where, ```exp_name``` is in ```[AllData_MLM, RetrictedData_MLM_TLM, MixedData_MLM_TLM, AugmentedData]```, ```method``` is in ```[Supervised, Unsupervised]``` and ```AugmentedData``` with ```Supervised``` shows the best results:

```
wget https://cdlisumerianunmt.s3.us-east-2.amazonaws.com/MASS/experiments/AugmentedData/Supervised/checkpoint.pth
```

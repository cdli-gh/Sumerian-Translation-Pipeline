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

## XLM (Unsupervised)

```
wget https://cdlisumerianunmt.s3.us-east-2.amazonaws.com/XLM/experiments/<exp_name>/PhaseTwo/checkpoint.pth
```
Where, ```exp_name``` is in ```[SumEn_AllData_MLM, SumEn_RetrictedData_MLM_TLM, SumEn_MixedData_MLM_TLM]``` and ```SumEn_RetrictedData_MLM_TLM``` shows the best results:

```
wget https://cdlisumerianunmt.s3.us-east-2.amazonaws.com/XLM/experiments/SumEn_RetrictedData_MLM_TLM/PhaseTwo/checkpoint.pth
```

## MASS (Semi-Supervised and Unsupervised Versions) 

```
wget https://cdlisumerianunmt.s3.us-east-2.amazonaws.com/MASS/experiments/<exp_name>/PhaseTwo/checkpoint.pth
```
Where, ```exp_name``` is in ```[SumEn_Supervised, SumEn_Unsupervised]``` and ```SumEn_Supervised``` shows the best results:

```
wget https://cdlisumerianunmt.s3.us-east-2.amazonaws.com/MASS/experiments/SumEn_Supervised/PhaseTwo/checkpoint.pth
```

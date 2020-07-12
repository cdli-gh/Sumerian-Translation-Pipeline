# Checkpoints for Translation

## Vanilla Transformer

```
wget https://cdlisumerianunmt.s3.us-east-2.amazonaws.com/Transformer/<data_mode>/_step_14000.pt
````
Where, ```data_mode``` is in ```[AllCompSents, UrIIICompSents, UrIIILineByLine, AllLineByLine]``` and ```AllCompSents``` shows the best results:

```
wget https://cdlisumerianunmt.s3.us-east-2.amazonaws.com/Transformer/AllCompSents/_step_14000.pt
````

## Back Translation 

```
wget https://cdlisumerianunmt.s3.us-east-2.amazonaws.com/BackTranslation/<shard_num>st/_step_10000.pt
```
Where, ```shard_num``` is in ```{1,8}``` and ```5``` shows the best results:

```
wget https://cdlisumerianunmt.s3.us-east-2.amazonaws.com/BackTranslation/5st/_step_10000.pt
```

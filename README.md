# Experimental code of SERM-master
This code repo contains the preprocess, training, inference and evaluation code for SERM-master on TACRED relation extraction dataset. 

## Requirements
* Python 3.6.3
* PyTorch 1.2.0
* Transformers 2.2.1
* CUDA 9.2

## Perparation
First, download and unzip RE dataset from [LDC](https://catalog.ldc.upenn.edu/LDC2018T24).

## Training
Please run 
```
python train.py
```

## Evaluation
Evaluate the trained model on TACRED dataset:
```
python evaluation.py
```
You will get:
| Model | Precision | Recall | micro-F1 |
| ---------------- | ------------ | ------------ | ------------ |
| SERM(ALBERT-large) | 0.828 | 0.811 | 0.819 |

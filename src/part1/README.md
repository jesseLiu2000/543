# Here is part1 of our project

including FastText, TextCNN, RNN and BERT model

**Our environment is Ubuntu=18.04, cuda=11.1, python=3.8, miniconda3**  

when you run the model, you should define the optimizer you want to test by yourself, you should follow the naming convention of the following table to pass in the name of the optimizer

## Optimizer Choice Table

| Input  | True optimizer |
| ------------- |:-------------:|
| Adam     | Adam     |
| AdamW     | AdamW     |
| SGD      | SGD     |

## How to run

```
conda env create -f part1.yaml
```
```
source activate part1

```
### fasttext model
```
python3 fasttext.py Adam
```
### textcnn model
```
python3 textcnn.py Adam
```
### rnn model
```
python3 rnn.py Adam
```
### bert model
```
python3 transformer.py Adam
```
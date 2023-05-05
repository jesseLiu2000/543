# Here is part2 of our project

including Embedding and GCNN model

**Our environment is Ubuntu=20.04**  

when you run the model, you should define the optimizer you want to test by yourself, you should follow the naming convention of the following table to pass in the name of the optimizer

## Optimizer Choice Table

| Input  | True optimizer |
| ------------- |:-------------:|
| Adam     | Adam     |
| AdamW     | AdamW     |
| SGD      | SGD     |

## How to run

```
conda create -n part2 python=3.8
source activate part2
```
```
pip install -r requirements.txt
```
### gcnn model
```
run gcnn.ipynb
```
### embed model
```
run embed.ipynb
```
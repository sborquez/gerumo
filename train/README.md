# Gerumo Train environment

## Requirements

## Setup Dataset

## Train a model

## Evaluate

How to evaluate your trained model and assemble results.

### Evaluate single model

```
(gerumo)$ python evaluate.py --model  "umonna_model.h5"
                             --config "./config/umonna_model.json"
                             --output "./evaluation"
                             --samples --results --predictions
```


### Evaluate experiment folder
```
(gerumo)$ python evaluate.py --experiment "./umonna_model_run_id"
                             --samples --results --predictions
```
### Evaluate Assembler
```
(gerumo)$ python evaluate.py --assembler "./config/umonna_assembler.json"
                             --output "." 
                             --samples --results --predictions
```
## Extras

### tools.py

### debug.py

### prepare_preprocessing.py

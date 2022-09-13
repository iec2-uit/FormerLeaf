# FormerLeaf

This repository is the official implementation of FormerLeaf: An Efficient Vision Transformer for Cassava Leaf Disease Detection.

## Data preparation

Download and extract Cassava Leaf Disease Dataset from https://www.kaggle.com/competitions/cassava-leaf-disease-classification/data. Place the extracted dataset in the input folder. The directory structure is as follows:
```
input/cassava-leaf-disease-classification/
  train_images/
    sample.jpg
  label_num_to_disease_map.json
  train.csv
```


We provide models trained on Cassava Leaf Disease Dataset. Models can be found [here](https://github.com/iec2-uit/FormerLeaf/releases/tag/model_zoo_release).

| Name  | F1 - score | #Params  | Size | URL|
| ------------- | ------------- | ------------- | ------------- |------------- |
| FormerLeaf | 96.82  | 85.8M  | 345.4  | [model](https://github.com/iec2-uit/FormerLeaf/releases/download/model_zoo_release/FormerLeaf) |
| FormerLeaf - 1 | 96.63  | 83.4M  | 336.4  | [model](https://github.com/iec2-uit/FormerLeaf/releases/download/model_zoo_release/FormerLeaf-1)  |
| FormerLeaf - 3 | 98.5  | 78.7M  | 318.4  | [model](https://github.com/iec2-uit/FormerLeaf/releases/download/model_zoo_release/FormerLeaf-3)  |
| FormerLeaf - 5 | 96.42  | 74M  | 300.4  | [model](https://github.com/iec2-uit/FormerLeaf/releases/download/model_zoo_release/FormerLeaf-5)  |
| FormerLeaf - 7 | 96.07  | 69.2M  | 282.4  | [model](https://github.com/iec2-uit/FormerLeaf/releases/download/model_zoo_release/FormerLeaf-7)  |
| FormerLeaf - 9 | 97.3  | 64.5M  | 264.4  | [model](https://github.com/iec2-uit/FormerLeaf/releases/download/model_zoo_release/FormerLeaf-9)  |
| FormerLeaf - 11 | 91.8  | 59.8M  | 246.4  | [model](https://github.com/iec2-uit/FormerLeaf/releases/download/model_zoo_release/FormerLeaf-11)  |
| FormerLeaf + SPMM| 95.3  | 85.5M  | 345.4  | [model](https://github.com/iec2-uit/FormerLeaf/releases/download/model_zoo_release/FormerLeaf_SPMM)  |


attention map LeIAP

| First Header  | Second Header |
| ------------- | ------------- |
| Content Cell  | Content Cell  |
| Content Cell  | Content Cell  |

## Prune function

From the FormerLeaf, we can create FormerLeaf -1 by using the prune function as follows:
D = {0: [4], 
         1: [5], 
         2: [10], 
         3: [5], 
         4: [3], 
         5: [3],
         6: [8], 
         7: [10],
         8: [11], 
         9: [8], 
         10: [0],
         11: [10]}
model.model.vit.prune_heads(D)

## Training

To train FormerLeaf on Cassava Leaf Disease Dataset on a single node with 1 gpu core for 165 epochs run:

## Evaluation

To evaluate a pretrained model FormerLeaf:

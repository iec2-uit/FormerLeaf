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

| Name  | F1 - score | #Params  | Size | Model Complexity | URL|
| ------------- | ------------- | ------------- | ------------- |------------- |------------- |
| FormerLeaf | 96.82  | 85.8M  | 345.4 |$O^(n^2)$ | [model](https://github.com/iec2-uit/FormerLeaf/releases/download/model_zoo_release/FormerLeaf) |
| FormerLeaf - 1 | 96.63  | 83.4M  | 336.4  |$O^(n^2)$ | [model](https://github.com/iec2-uit/FormerLeaf/releases/download/model_zoo_release/FormerLeaf-1)  |
| FormerLeaf - 3 | 98.5  | 78.7M  | 318.4  |$O^(n^2)$ | [model](https://github.com/iec2-uit/FormerLeaf/releases/download/model_zoo_release/FormerLeaf-3)  |
| FormerLeaf - 5 | 96.42  | 74M  | 300.4  |$O^(n^2)$ | [model](https://github.com/iec2-uit/FormerLeaf/releases/download/model_zoo_release/FormerLeaf-5)  |
| FormerLeaf - 7 | 96.07  | 69.2M  | 282.4  |$O^(n^2)$ | [model](https://github.com/iec2-uit/FormerLeaf/releases/download/model_zoo_release/FormerLeaf-7)  |
| FormerLeaf - 9 | 97.3  | 64.5M  | 264.4  |$O^(n^2)$ | [model](https://github.com/iec2-uit/FormerLeaf/releases/download/model_zoo_release/FormerLeaf-9)  |
| FormerLeaf - 11 | 91.8  | 59.8M  | 246.4  |$O^(n^2)$ | [model](https://github.com/iec2-uit/FormerLeaf/releases/download/model_zoo_release/FormerLeaf-11)  |
| FormerLeaf + SPMM| 95.3  | 85.5M  | 345.4  |$O^(n^2/p)$ | [model](https://github.com/iec2-uit/FormerLeaf/releases/download/model_zoo_release/FormerLeaf_SPMM)  |




## Prune function
From the FormerLeaf, we can create the different pruned models from FormerLeaf - 1 to FormerLeaf - 11 by pruning the corresponding head from each layer, according to the table below. This table is created from our proposed algorithm LeIAP.

| Layer  | Head - sort by important values | ^| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| 0 | 4  |7  | 3  |8  |6  |5  |0  |9  |2  |10  | 11 |1  |
| 1 |  5 |4  | 1  |3  |2  |8  |9  |7 |0   |11  | 6  |10 |
| 2 |  10 |1  | 7  |9  |5  |3  |2  |8 |11   |0  | 4  |6 |
| 3 |  5 |1  | 2  |0  |11  |6  |3  |7 |10   |8  | 4  |9 |
| 4 |  3 |11  | 7  |9  |6  |4  |10  |5 |0   |1  | 2  |8 |
| 5 |  3 |9  | 2  |11  |1  |0  |6  |5 |7   |8  | 10  |4 |
| 6 |  8 |4  | 11  |10  |1  |9  |6  |5 |7   |10  |2  |3 |
| 7 |  10 |3  | 9  |4  |2  |0  |7  |1 |6   |11  | 8  |5 |
| 8 |  11 |0  | 6  |9  |7  |2  |10  |8 |1   |5  | 3  |4 |
| 9 |  8 |6  | 2  |10  |9  |3  |4  |0 |7   |11  | 5  |1 |
| 10 |  0 |4  | 5  |10  |1  |6  |7  |3 |8   |11  | 9 |2 |
| 11 |  10 |2  |7  |9  |3  |5  |6  |8 |11   |0  | 1  |4 |

The following source code is used to create FormerLeaf - 1 by pruning one head in each Transformer encoder layer.
```
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
```

## Training

To train FormerLeaf on Cassava Leaf Disease Dataset on a single node with 1 gpu core for 165 epochs run:

## Evaluation

To evaluate a pretrained model FormerLeaf:

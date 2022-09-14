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
From the FormerLeaf, we can create the different pruned models from FormerLeaf - 1 to FormerLeaf - 11 by pruning the corresponding heads from each Transformer  layer, according to the table below. This table is created from our proposed algorithm LeIAP.

<table>
  <tr>
    <td>Layer</td>
    <td  colspan="12">Head - sort by important values</td>
  </tr>
  <tr>
    <td>0</td>
    <td>4</td>
    <td>7</td>
    <td>3</td>
    <td>8</td>
    <td>6</td>
    <td>5</td>
    <td>0</td>
    <td>9</td>
    <td>2</td>
    <td>10</td>
    <td>11</td>
    <td>1</td>
  </tr>
   <tr>
    <td>1</td>
    <td>5</td>
    <td>4</td>
    <td>1</td>
    <td>3</td>
    <td>2</td>
    <td>8</td>
    <td>9</td>
    <td>7</td>
    <td>0</td>
    <td>11</td>
    <td>6</td>
    <td>10</td>
  </tr>
   <tr>
    <td>2</td>
    <td>10</td>
    <td>1</td>
    <td>7</td>
    <td>9</td>
    <td>5</td>
    <td>3</td>
    <td>2</td>
    <td>8</td>
    <td>11</td>
    <td>0</td>
    <td>4</td>
    <td>6</td>
  </tr>
   <tr>
    <td>3</td>
    <td>5</td>
    <td>1</td>
    <td>2</td>
    <td>0</td>
    <td>11</td>
    <td>6</td>
    <td>3</td>
    <td>7</td>
    <td>10</td>
    <td>8</td>
    <td>4</td>
    <td>9</td>
  </tr>
   <tr>
    <td>4</td>
    <td>3</td>
    <td>11</td>
    <td>7</td>
    <td>9</td>
    <td>6</td>
    <td>4</td>
    <td>10</td>
    <td>5</td>
    <td>0</td>
    <td>1</td>
    <td>2</td>
    <td>8</td>
  </tr>
   <tr>
    <td>5</td>
    <td>3</td>
    <td>9</td>
    <td>2</td>
    <td>11</td>
    <td>1</td>
    <td>0</td>
    <td>6</td>
    <td>5</td>
    <td>7</td>
    <td>8</td>
    <td>10</td>
    <td>4</td>
  </tr>
   <tr>
    <td>6</td>
    <td>8</td>
    <td>4</td>
    <td>11</td>
    <td>10</td>
    <td>1</td>
    <td>9</td>
    <td>6</td>
    <td>5</td>
    <td>7</td>
    <td>0</td>
    <td>2</td>
    <td>3</td>
  </tr>
   <tr>
    <td>7</td>
    <td>10</td>
    <td>3</td>
    <td>9</td>
    <td>4</td>
    <td>2</td>
    <td>0</td>
    <td>7</td>
    <td>1</td>
    <td>6</td>
    <td>11</td>
    <td>8</td>
    <td>5</td>
  </tr>
   <tr>
    <td>8</td>
    <td>11</td>
    <td>0</td>
    <td>6</td>
    <td>9</td>
    <td>7</td>
    <td>2</td>
    <td>10</td>
    <td>8</td>
    <td>1</td>
    <td>5</td>
    <td>3</td>
    <td>4</td>
  </tr>
   <tr>
    <td>9</td>
    <td>8</td>
    <td>6</td>
    <td>2</td>
    <td>10</td>
    <td>9</td>
    <td>3</td>
    <td>4</td>
    <td>0</td>
    <td>27</td>
    <td>11</td>
    <td>5</td>
    <td>1</td>
  </tr>
   <tr>
    <td>10</td>
    <td>0</td>
    <td>4</td>
    <td>5</td>
    <td>10</td>
    <td>1</td>
    <td>6</td>
    <td>7</td>
    <td>3</td>
    <td>8</td>
    <td>11</td>
    <td>9</td>
    <td>2</td>
  </tr>
   <tr>
    <td>11</td>
    <td>10</td>
    <td>2</td>
    <td>7</td>
    <td>9</td>
    <td>3</td>
    <td>5</td>
    <td>6</td>
    <td>8</td>
    <td>11</td>
    <td>0</td>
    <td>1</td>
    <td>4</td>
  </tr>
</table>

For example, the following source code is used to create FormerLeaf - 1 by pruning one head in each Transformer encoder layer.
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

To train FormerLeaf and its variants on Cassava Leaf Disease Dataset on Google Colab with 1 gpu core for 165 epochs run:
```
notebooks/
  Train_FormerLeaf_Huggingface_Origin_5fold_33epochs.ipynb //FormerLeaf
  Train_FormerLeaf_1_Huggingface_Origin_5fold_33epochs.ipynb //FormerLeaf-1
  Train_FormerLeaf_3_Huggingface_Origin_5fold_33epochs.ipynb //FormerLeaf-3
  Train_FormerLeaf_5_Huggingface_Origin_5fold_33epochs.ipynb //FormerLeaf-5
  Train_FormerLeaf_7_Huggingface_Origin_5fold_33epochs.ipynb //FormerLeaf-7
  Train_FormerLeaf_9_Huggingface_Origin_5fold_33epochs.ipynb //FormerLeaf-9
  Train_FormerLeaf_11_Huggingface_Origin_5fold_33epochs.ipynb //FormerLeaf-11
  Train_FormerLeaf_Huggingface_sparse_Transformer.ipynb //FormerLeaf + SPMM
```

## Evaluation

To evaluate a pretrained model FormerLeaf:

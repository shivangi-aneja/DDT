## DDT (Deep Distribution Transfer) : Zero and Few-Shot Transfer for Facial Forgery Detection

<p float="left">
<figure>
    <img src='/figures/method.jpg' width="100%"  alt='method' />
</figure>
</p>

### 0. Setup
For setting up environment, please read [SETUP.md](SETUP.md)

### 1. Datasets
<p float="left">
<figure>
    <img src='/figures/datasets.jpg' width="100%"  alt='Datasets' />
</figure>
</p>
   

Please contact the respective authors to get the datasets. Except AIF dataset which is donated by the company, [AI Foundation](https://aifoundation.com/) to the authors and can be downloaded with [this link](https://www.dropbox.com/s/arnx7s13hm129ra/AIF.zip).

| [FaceForensics ++ ](https://arxiv.org/pdf/1901.08971.pdf)  | [Google DFDC](https://ai.googleblog.com/2019/09/contributing-data-to-deepfake-detection.html) | [Dessa](https://www.dessa.com/post/deepfake-detection-that-actually-works) | [Celeb DF(v2)](https://arxiv.org/abs/1909.12962) | [AIF](https://www.dropbox.com/s/arnx7s13hm129ra/AIF.zip) |
| ------------- | ------------- | ------------- | ------------- | ------------- |

#### Dataset Split-up used for experiments
| Dataset  | Train | Val | Test |
| ------------- | ------------- | ------------- | ------------- |
| FaceForensics++  | 720  | 140  | 140 |
| Google DFDC  | - | -  | 28  |
| Dessa  | 70 | -  | 14  |
| Celeb DF  | 500 | -  | 90  |
| AIF  | 12 | -  | 99  |


### 1. Training and Testing (Zero-Shot)

All the paths (dataset, models) , losses and other hyperparameters are specified in `config.py`.

| Method  | File | 
| ------------- | ------------- |
| Classifier  | `train_classifier.py`  |
| [ProtoNet](https://arxiv.org/pdf/1703.05175.pdf)  | `train_protonet.py`  |
| [RelationNet](https://arxiv.org/pdf/1711.06025.pdf)  | `train_relation_net.py`  |
| **DDT (Ours)**  | `train_ddt.py`  |

How to run `ddt.py`
```
usage: train_ddt.py [-h] [-seed RANDOM_SEED]  [-train_mode TRAIN_MODE]
                [-epochs EPOCHS]  [-num-classes NUM_CLASSES]          

optional arguments:
  -h, --help                     show this help message and exit
  
  --seed RANDOM_SEED,            random seed for training (default: 1)
                        
  --epochs EPOCHS,               number of epochs (default: 10000)  
  
  --train_mode TRAIN_MODE,   dataset mode for training (default: 'train') {'train', 'test'}
                        
  --num-classes NUM_CLASSES,     number of classes to train with (default: 2)
```

#### Sample Command for training
```
python3 train_ddt.py --train_mode train
```

#### Sample Command for testing
```
python3 train_ddt.py --train_mode test
```

Pre-trained models for DDT can be downloaded [here](https://drive.google.com/file/d/1Qq8HitP_DQIzz7p679ak3ZFOFO-lmvTN/).

## 2. Fine-tuning (Few-Shot Learning)
For fine-tuning experiments, results are compared with supervised domain adaptation methods as well.

| Transfer Learning Method  | File | 
| ------------- | ------------- |
| Classifier  | `finetune_classifier.py`  |
| [DDC (Deep Domain Confusion)](https://arxiv.org/abs/1412.3474)  | `finetune_ddc.py`  |
| [Deep Correlation Alignment](https://arxiv.org/abs/1607.01719)  | `finetune_coral.py`  |
| [Classification and Contrastive Semantic Alignment Loss](https://arxiv.org/pdf/1709.10190.pdf)  | `finetune_ccsa.py`  |
| [d-SNE: Domain Adaptation using Stochastic Neighborhood Embedding](https://arxiv.org/abs/1905.12775)  | `finetune_dsne.py`  |
| [ProtoNet](https://arxiv.org/pdf/1703.05175.pdf)  | `finetune_protonet.py`  |
| [RelationNet](https://arxiv.org/pdf/1711.06025.pdf)  | `finetune_relation_net.py`  |
| **DDT (Ours)**  | `finetune_ddt.py`  |

#### Scripts 
All fine-tuning results are averaged over 10 runs. So we fine-tuned 10 models by running the following scripts.

Classifier : `./script_classifier_resnet.sh` 

Ours : `./script_vae_resnet.sh`

Other Transfer Methods(change filename if needed) : `./script_other_tl_resnet.sh`




<!--  
## 4. Results

### Zero-Shot Results

##### 4.1 Transfer Accuracy for different manipulation methods
<p float="left">
<img src="/figures/results/zero_shot_acc.png" width="100%" />
</p>

##### 4.2 DF manipulation to other manipulation methods
<p float="left">
<img src="/figures/results/zero_shot/df_zero_shot.png" width="100%" />
</p>

##### 4.3 NT manipulation to other manipulation methods
<p float="left">
<img src="/figures/results/zero_shot/nt_zero_shot.png" width="100%" />
</p>

##### 4.4 Model trained with DF manipulation with and without proposed augmentation strategy and evaluated on NT manipulation method.
<p float="left">
<img src="/figures/results/aug/df_aug.png" width="100%" />
</p>

##### 4.5 Model trained with NT manipulation with and without proposed augmentation strategy and evaluated on DF manipulation method.
<p float="left">
<img src="/figures/results/aug/nt_aug.png" width="100%" />
</p>

##### Model trained with DF + NT manipulation with and without proposed augmentation strategy and evaluated on other datasets
<p float="left">
<img src="/figures/results/other_datasets/other_datasets_spatial_aug.png" width="100%" />
</p>


### Few-Shot Results With Spatial Augmentation

##### 4.6 Fine-tuning NT to DF manipulation 
<p float="left">
<img src="/figures/results/few_shot/nt_to_df_with_aug.png" width="100%" />
</p>

##### 4.7 Fine-tuning DF to NT manipulation 
<p float="left">
<img src="/figures/results/few_shot/df_to_nt_with_aug.png" width="100%" />
</p>

##### 4.8 Fine-tuning (DF + NT) to Dessa dataset
<p float="left">
<img src="/figures/results/other_datasets/dessa_with_spatial_aug.png" width="100%" />
</p>

##### 4.9 Fine-tuning (DF + NT) to AIF dataset
<p float="left">
<img src="/figures/results/other_datasets/aif_with_spatial_aug.png" width="100%" />
</p>




Trello Board [here](https://trello.com/b/wvafcTj1/master-thesis-todos)
-->

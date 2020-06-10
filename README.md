## DDT (Deep Distribution Transfer) : Zero and Few-Shot Transfer for Facial Forgery Detection

<p float="left">
<figure>
    <img src='/figures/method/method.jpg' width="100%"  alt='method' />
</figure>
</p>

### 0. Setup
For setting up environment, please read [SETUP.md](SETUP.md)

## 1. Datasets
Please contact the respective authors to get the 

#### [FaceForensics ++ Dataset](https://arxiv.org/pdf/1901.08971.pdf)
The dataset provides videos for 4 different manipulation methods, namely DeepFakes, Face2Face, FaceSwap, and NeuralTextures and their real counterparts.
<p float="left">
<figure>
    <img src='/figures/dataset/faceforensics.png' width="100%"  alt='FaceForensics++' />
</figure>
</p>

### [Google DFDC Dataset](https://ai.googleblog.com/2019/09/contributing-data-to-deepfake-detection.html)
<p float="left">
<figure>
    <img src='/figures/dataset/dfdc.jpg' width="70%"  alt='Google DFDC' />
</figure>
</p>

### [Dessa Dataset](https://www.dessa.com/post/deepfake-detection-that-actually-works)
<p float="left">
<figure>
    <img src='/figures/dataset/dessa.png' width="70%"  alt='Google DFDC' />
</figure>
</p>

### [AIF Dataset](https://aifoundation.com/)
<p float="left">
<figure>
    <img src='/figures/dataset/aif_images.png' width="100%"  alt='Google DFDC' />
</figure>
</p>

### Dataset Splitting
| Dataset  | Train | Val | Test |
| ------------- | ------------- | ------------- | ------------- |
| FaceForensics++  | 720  | 140  | 140 |
| Google DFDC  | - | -  | 28  |
| Dessa  | 70 | -  | 14  |
| AIF  | 12 | -  | 99  |


### 1. Training and Testing (Zero-Shot)

To train/test Classifier, run the file `train_classifier.py`

To train/test ForensicTransfer, run the file `train_forensic_transfer.py` 

To train/test Our approach, run the file `train_vae_2classes.py`

How to run `train_classifier.py` / `train_forensic_transfer.py` / `train_vae_2classes.py`
```
usage: train_vae_2classes.py [-h] [-seed RANDOM_SEED]  [-div-loss DIVERGENCE_LOSS]
               [--batch-size BATCH_SIZE] [--latent-dim LATENT_DIM] [-dataset_mode DATASET_MODE]
                [-epochs EPOCHS]  [-num-classes NUM_CLASSES]          

optional arguments:
  -h, --help                     show this help message and exit
  
  --seed RANDOM_SEED,            random seed for training (default: 1)
  
  --div-loss DIVERGENCE_LOSS,    divergence loss function (only for train_vae_2classes.py) , {'KL', 'Wasserstein'}
                        
  --batch-size BATCH_SIZE,       input batch size for training (default: 128)
                        
  --epochs EPOCHS,               number of epochs (default: 10000) 
  
  --latent-dim LATENT_DIM,       embedding size (default: 16) 
  
  --dataset_mode DATASET_MODE,   dataset mode for training (default: 'face') {'face', 'face_finetune', 'face_residual', 'lip'}
                        
  --num-classes NUM_CLASSES,     number of classes to train with (default: 2)
```

#### Sample Command
```
python3 train_vae_2classes.py --batch-size 128  --latent-dim 16
```

## 2. Fine-tuning (Few-Shot Learning)

To fine-tune Classifier, run the file `finetune_classifier.py`

To fine-tune ForensicTransfer, run the file `finetune_forensic_transfer.py` 

To fine-tune Our approach, run the file `finetune_vae_2classes.py`

#### Fine-tune with Other Transfer Learning Methods

[DDC (Deep Domain Confusion)](https://arxiv.org/abs/1412.3474) : `finetune_mmd_classifier.py`

[Deep Correlation Alignment](https://arxiv.org/abs/1607.01719) : `finetune_coral_classifier.py`

[Classification and Contrastive Semantic Alignment Loss](https://arxiv.org/pdf/1709.10190.pdf) : `finetune_ccsa_classifier.py`

[d-SNE: Domain Adaptation using Stochastic Neighborhood Embedding](https://arxiv.org/abs/1905.12775) : `finetune_dsne_classifier.py`

#### Scripts 
All fine-tuning results are averaged over 10 runs. So we fine-tuned 10 models by running the following scripts.

Classifier : `./script_classifier_resnet.sh` 

ForensicTransfer : `./script_forensic_transfer_resnet.sh`

Ours : `./script_vae_resnet.sh`

Other Transfer Methods(change filename if needed) : `./script_other_tl_resnet.sh`





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



<!--  
Trello Board [here](https://trello.com/b/wvafcTj1/master-thesis-todos)
-->

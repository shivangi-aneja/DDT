# Generalized Zero and Few Shot Transfer for Facial Forgery Detection

## 0. Links
[Research Log](https://docs.google.com/document/d/16Equq5mI2USyMEJvcCYkBg37eLqGu-ICjZgOuCcptf0/edit)

For setting up environment, please read `SETUP.md`


## 1. Training and Testing (Zero-Shot)

To train Classifier, run the file `train_classifier.py`

To test ForensicTransfer, run the file `train_forensic_transfer.py` 

To test Our approach, run the file `train_vae_2classes.py`

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

To train Classifier, run the file `train_classifier.py`

To test ForensicTransfer, run the file `train_forensic_transfer.py` 

To test Our approach, run the file `train_vae_2classes.py`

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



## 3. Datasets Used

### [FaceForensics ++ Dataset](https://arxiv.org/pdf/1901.08971.pdf)
The dataset provides videos for 4 different manipulation methods, namely DeepFakes, Face2Face, FaceSwap, and NeuralTextures and their real counterparts.
<p float="left">
<figure>
    <img src='/figures/dataset/faceforensics.png' width="100%"  alt='FaceForensics++' />
    <figcaption><a href="">Image Source</a></figcaption>
</figure>
</p>

### [Google DFDC Dataset](https://ai.googleblog.com/2019/09/contributing-data-to-deepfake-detection.html)
<p float="left">
<figure>
    <img src='/figures/dataset/dfdc.jpg' width="100%"  alt='Google DFDC' />
</figure>
</p>

### [Dessa Dataset](https://www.dessa.com/post/deepfake-detection-that-actually-works)
<p float="left">
<figure>
    <img src='/figures/dataset/dessa.png' width="100%"  alt='Google DFDC' />
    <a href="">Image Source</a>
</figure>
</p>

### [AIF Dataset](https://aifoundation.com/)
<p float="left">
<figure>
    <img src='/figures/dataset/aif.png' width="100%"  alt='Google DFDC' />
</figure>
</p>

### Dataset Splitting
| Dataset  | Train | Val | Test |
| ------------- | ------------- | ------------- | ------------- |
| FaceForensics++  | 720  | 140  | 140 |
| Google DFDC  | - | -  | 28  |
| Dessa  | - | -  | 28  |
| AIF  | - | -  | 28  |

## 4. Results

The task here is to train object detection network. We used SSD (Single Shot Multibox detectot to evaluate our results).
Code is available for both `Faster R-CNN` and `SSD`.



To run SSD, migrate to directory SSD-keras, and follow `README.md` for instructions



## 3. Misc

Code is available for both Faster R-CNN and SSD. But I used SSD for the task of detecting polyps.

All the results and models can be downloaded from [here](https://drive.google.com/file/d/1Fb9XrDYKtzJiysEi79dC_NZlsrgUr-9o/view?usp=sharing).
No pretrained models were used. Everything is trained from scratch.



## 4. Results

#### Results on Validation set for Single Shot Multi Box Detector

##### Succesful Cases (Benchmark Dataset CVC Colon)
<p float="left">
<img src="/images/cvc/8.png" width="30%" />
<img src="/images/cvc/2.png" width="30%" />
<img src="/images/cvc/3.png" width="30%" />
</p>

<p float="left">
<img src="/images/cvc/4.png" width="30%" />
<img src="/images/cvc/5.png" width="30%" />
<img src="/images/cvc/111.png" width="30%" />
</p>


<!--
##### Succesful Cases (Hospital Dataset)
<p float="left">
<img src="/images/hospital/h1.png" width="30%" />
<img src="/images/hospital/h2.png" width="30%" />
<img src="/images/hospital/h3.png" width="30%" />
</p>
<p float="left">
<img src="/images/hospital/h4.png" width="30%" />
<img src="/images/hospital/h5.png" width="30%" />
<img src="/images/hospital/h6.png" width="30%" />
</p>
-->





#####  Failure Cases (Benchmark Dataset CVC Colon)
<p float="left">
<img src="/images/cvc/e4.png" width="30%" />
<img src="/images/cvc/112.png" width="30%" />
<img src="/images/cvc/113.png" width="30%" />
</p>

<p float="left">
<img src="/images/cvc/e4.png" width="30%" />
<img src="/images/cvc/e5.png" width="30%" />
<img src="/images/cvc/e6.png" width="30%" />
</p>

<!--
##### Failure Cases (Hospital Dataset)
<p float="left">
<img src="/images/hospital/h1.png" width="30%" />
<img src="/images/hospital/h2.png" width="30%" />
<img src="/images/hospital/h3.png" width="30%" />
</p>
<p float="left">
<img src="/images/hospital/h4.png" width="30%" />
<img src="/images/hospital/h5.png" width="30%" />
<img src="/images/hospital/h6.png" width="30%" />
</p>
-->











<!--  
Trello Board [here](https://trello.com/b/wvafcTj1/master-thesis-todos)
-->

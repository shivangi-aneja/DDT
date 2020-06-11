# Config file for path and other hyperparameters
import os
import torch
from torch import nn
from common.utils.dataset import make_dataset
from torchvision import transforms
from common.losses.custom_losses import wasserstein_distance, kl_with_gaussian_unit_std


batch_size = 128
latent_dim = 16
train_lr = 1e-3
finetune_lr = 1e-5
train_patience = 10
finetune_patience = 30
scheduler_factor = 0.7
scheduler_patience = 5
divergence = 'wasserstein'
BASE_PATH = os.getcwd()
fake_classes = ['nt']
num_classes = len(fake_classes) + 1
orig_weight_factor = num_classes - 1
ft_images_train = 25
alpha = 0.25


# Distribution Means (Only for DDT)
# Real mean
mean1 = torch.zeros(int(latent_dim)).cuda()
mean1[:int(latent_dim / 2)] = 1
mean1[int(latent_dim / 2):] = 0
# Fake mean
mean2 = torch.zeros(int(latent_dim)).cuda()
mean2[:int(latent_dim / 2)] = 0
mean2[int(latent_dim / 2):] = 1


# Losses
# ------ Distribution losses (only for DDT) ----------
DIV_LOSSES = {
    'kl': kl_with_gaussian_unit_std,
    'wasserstein': wasserstein_distance
}
class_weights = torch.Tensor([orig_weight_factor, 1]).cuda()
div_loss = DIV_LOSSES[divergence]
# ------ Classification Loss
classification_loss = nn.CrossEntropyLoss(reduction='mean', weight=class_weights)
# MSE loss
mse = nn.MSELoss().cuda()

train_path = '/home/aneja/Desktop/Data/Masters/master_thesis/data/ff_face_20k/c23/train_20k_c23/'
train_path_ff = '/home/aneja/Desktop/Data/Masters/master_thesis/data/ff_face_20k/c23/train_20k_c23/'
val_path = '/home/aneja/Desktop/Data/Masters/master_thesis/data/ff_face_20k/c23/val_6k_c23/'
test_path = '/home/aneja/Desktop/Data/Masters/master_thesis/data/ff_face_20k/c23/test/xtra/'

MODEL_PATH = os.path.join(os.getcwd(), 'pretrained_models/')


# Dataloaders fro Pre-training
train_dataset = make_dataset(name='ff', base_path=train_path, num_classes=num_classes,
                             mode='face', image_count='all',
                             transform=transforms.Compose([transforms.ToPILImage(),
                                                           transforms.RandomHorizontalFlip(),
                                                           transforms.RandomVerticalFlip(),
                                                           transforms.ToTensor(),
                                                           transforms.Normalize([0.5] * 3, [0.5] * 3),
                                                           ]))


val_dataset = make_dataset(name='ff', base_path=val_path, num_classes=num_classes,
                           mode='face', image_count='all',
                           transform=transforms.Compose(
                                [transforms.ToPILImage(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.5] * 3, [0.5] * 3)]))

test_dataset = make_dataset(name='ff', base_path=test_path, num_classes=num_classes,
                           mode='face', image_count='all',
                           transform=transforms.Compose(
                                [transforms.ToPILImage(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.5] * 3, [0.5] * 3)]))


### For fine-tuning
batch_size_train = min(2*ft_images_train, batch_size)
src_fake_classes = ['df', 'nt']
target_fake_classes = ['dfdc']


source_train_dataset = make_dataset(name='ff', base_path=train_path, num_classes=len(src_fake_classes)+1,
                             mode='face', image_count='all',
                             transform=transforms.Compose([transforms.ToPILImage(),
                                                           transforms.RandomHorizontalFlip(),
                                                           transforms.RandomVerticalFlip(),
                                                           transforms.ToTensor(),
                                                           transforms.Normalize([0.5] * 3, [0.5] * 3)
                                                           ]))


target_train_dataset = make_dataset(name='ff', base_path=train_path, num_classes=len(target_fake_classes) + 1,
                                    mode='face_finetune', image_count=ft_images_train, base_path_ff=train_path_ff,
                                    transform=transforms.Compose([transforms.ToPILImage(),
                                                                  transforms.RandomHorizontalFlip(),
                                                                  transforms.RandomVerticalFlip(),
                                                                  transforms.ToTensor(),
                                                                  transforms.Normalize([0.5] * 3, [0.5] * 3)]))


target_test_dataset = make_dataset(name='ff', base_path=val_path, num_classes=len(target_fake_classes) + 1,
                                   mode='face_finetune', image_count='all',
                                   transform=transforms.Compose(
                                       [transforms.ToPILImage(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.5] * 3, [0.5] * 3)]))
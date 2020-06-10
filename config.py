# Config file for path and other hyperparameters
import os
import torch
from torch import nn
from common.utils.dataset import make_dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from common.losses.custom_losses import wasserstein_distance, kl_with_gaussian_unit_std

fake_classes = ['nt']
dataset_mode = 'face'
num_classes = len(fake_classes) + 1
batch_size = 128
latent_dim = 16
train_lr = 1e-3
train_patience = 10
scheduler_factor = 0.7
scheduler_patience = 5
divergence = 'wasserstein'
BASE_PATH = os.getcwd()
orig_weight_factor = num_classes - 1

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

train_path = '/home/shivangi/Desktop/Projects/master_thesis/data/ff_face_20k/c23/train_20k_c23/'
val_path = '/home/shivangi/Desktop/Projects/master_thesis/data/ff_face_20k/c23/val_6k_c23/'
train_path_ff = ''

MODEL_PATH = os.path.join(os.getcwd(), 'models/')


train_dataset = make_dataset(name='ff', base_path=train_path, num_classes=num_classes,
                             mode='face', image_count='all',
                             transform=transforms.Compose([transforms.ToPILImage(),
                                                           transforms.RandomHorizontalFlip(),
                                                           transforms.RandomVerticalFlip(),
                                                           transforms.ToTensor(),
                                                           transforms.Normalize([0.5] * 3, [0.5] * 3),
                                                           ]))


test_dataset = make_dataset(name='ff', base_path=val_path, num_classes=num_classes,
                            mode='face', image_count='all',
                            transform=transforms.Compose(
                                [transforms.ToPILImage(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.5] * 3, [0.5] * 3)]))

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=16, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=32, shuffle=False)


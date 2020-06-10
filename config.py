# Config file for path and other hyperparameters
import os
from os import makedirs
from common.utils.dataset import make_dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from common.losses.custom_losses import wasserstein_distance, kl_with_gaussian_unit_std

fake_classes = ['nt']
dataset_mode = 'face'
BASE_PATH = os.getcwd()

train_path = '/home/shivangi/Desktop/Projects/master_thesis/data/ff_face_20k/c23/train_20k_c23/'
val_path = '/home/shivangi/Desktop/Projects/master_thesis/data/ff_face_20k/c23/val_6k_c23/'

MODEL_PATH = os.path.join(os.getcwd(), 'models/')
best_path = MODEL_PATH + 'vae/' + dataset_mode + '/2classes/best/'
if not os.path.isdir(best_path):
    makedirs(best_path)

DIV_LOSSES = {
    'kl': kl_with_gaussian_unit_std,
    'wasserstein': wasserstein_distance
}

train_lr = 1e-3

train_dataset = make_dataset(name='ff', base_path=train_path, num_classes=num_classes, fake_classes=fake_classes,
                             mode='face', image_count='all',
                             transform=transforms.Compose([transforms.ToPILImage(),
                                                           transforms.RandomHorizontalFlip(),
                                                           transforms.RandomVerticalFlip(),
                                                           transforms.ToTensor(),
                                                           transforms.Normalize([0.5] * 3, [0.5] * 3),
                                                           ]))


test_dataset = make_dataset(name='ff', base_path=val_path, num_classes=num_classes, fake_classes=fake_classes,
                            mode='face', image_count='all',
                            transform=transforms.Compose(
                                [transforms.ToPILImage(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.5] * 3, [0.5] * 3)]))

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=16, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=32, shuffle=False)

train_patience = 10
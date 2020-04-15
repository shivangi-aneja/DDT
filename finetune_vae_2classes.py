from __future__ import print_function

import argparse
import os
import random
from os import makedirs

import numpy as np
import torch
import torch.utils.data
from create_plot import print_confusion_matrix
from sklearn.metrics import confusion_matrix
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from common.logging.tf_logger import Logger
from common.losses.custom_losses import wasserstein_distance, kl_with_gaussian_unit_std, wasserstein_distance_vector
from common.models.resnet_subset_models import VariationalEncoder1 as Encoder
from common.utils.dataset import make_dataset

parser = argparse.ArgumentParser(description='VAE FaceForensics++ Transfer Learning')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('-i', '--ft_images', type=int, default=1, metavar='N',
                    help='number of images for fine_tuning (default: 10)')
parser.add_argument('--latent_dim', type=int, default=16, metavar='N',
                    help='latent embedding size (default: 128)')
parser.add_argument('--epochs', type=int, default=10000, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--num_classes', type=int, default=2, metavar='N',
                    help='Number of classes (N fakes + 1 real)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('-m', '--model_name', type=str,
                    default='NONE', help='name for model')
parser.add_argument('-r', '--run', type=str,
                    default='1', help='run number')
parser.add_argument('-div_loss', '--div_loss', type=str,
                    default='wasserstein', help='Divergence Loss')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
random.seed(int(args.run))
torch.manual_seed(int(args.run))
device = torch.device("cuda" if args.cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
patience = 30
ft_images_train = args.ft_images

train_path = '/home/shivangi/Desktop/Projects/master_thesis/data/ff_face_20k/c23/train_20k_c23/'
val_path = '/home/shivangi/Desktop/Projects/master_thesis/data/ff_face_20k/c23/val_6k_c23/'

src_fake_classes = ['df', 'nt']
target_fake_classes = ['dfdc']
tsne_fake_classes = ['dfdc']

target_train_dataset = make_dataset(name='ff', base_path=train_path, num_classes=len(target_fake_classes) + 1,
                                    fake_classes=target_fake_classes,
                                    mode='face_finetune', image_count=ft_images_train,
                                    transform=transforms.Compose([transforms.ToPILImage(),
                                                                  transforms.RandomHorizontalFlip(),
                                                                  transforms.RandomVerticalFlip(),
                                                                  # transforms.RandomResizedCrop(256),
                                                                  transforms.ToTensor(),
                                                                  transforms.Normalize([0.5] * 3, [0.5] * 3)]))

# source_test_dataset = make_dataset(name='ff', base_path=val_path, num_classes=len(src_fake_classes)+1, fake_classes=src_fake_classes,
#                                    mode='face_finetune', image_count='all',
#                                    transform=transforms.Compose(
#                                        [transforms.ToPILImage(),
#                                         transforms.ToTensor(),
#                                         transforms.Normalize([0.5] * 3, [0.5] * 3)]))


target_test_dataset = make_dataset(name='ff', base_path=val_path, num_classes=len(target_fake_classes) + 1,
                                   fake_classes=target_fake_classes,
                                   mode='face_finetune', image_count='all',
                                   transform=transforms.Compose(
                                       [transforms.ToPILImage(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.5] * 3, [0.5] * 3)]))

tsne_target_test_dataset = make_dataset(name='ff', base_path=val_path, num_classes=len(tsne_fake_classes) + 1,
                                        fake_classes=tsne_fake_classes,
                                        mode='face_finetune', image_count='all',
                                        transform=transforms.Compose(
                                            [transforms.ToPILImage(),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.5] * 3, [0.5] * 3)]))

batch_size = args.batch_size

target_train_loader = DataLoader(dataset=target_train_dataset, batch_size=batch_size, num_workers=8, shuffle=True)
# source_test_loader = DataLoader(dataset=source_test_dataset, batch_size=batch_size, num_workers=8, shuffle=False)
target_test_loader = DataLoader(dataset=target_test_dataset, batch_size=batch_size, num_workers=8, shuffle=False)
tsne_loader = DataLoader(dataset=tsne_target_test_dataset, batch_size=batch_size, num_workers=8, shuffle=False)

logger = Logger(model_name='vae_model', data_name='ff', log_path=os.path.join(os.getcwd(),
                                                                              'tf_logs/vae/2classes_finetune/' + str(
                                                                                  ft_images_train) + '_images/' + 'run_' + args.run + '/' + args.model_name))
vae_source = 'vae_train_20k_val3k_mean1_std1_c23_latent16_3blocks_2classes_mixup_flip_normalize_df_nt.pt'
vae_target = args.model_name + '.pt'

transfer_dir = 'df_nt_to_aif_no_mixup'
# Paths
MODEL_PATH = os.path.join(os.getcwd(), 'models/')
src_path_vae = MODEL_PATH + 'vae/face/2classes/best/'
tgt_path_vae_best = MODEL_PATH + 'vae_finetune/2classes_' + str(
    ft_images_train) + 'images/' + transfer_dir + '/' + args.run + '_run/'

if not os.path.isdir(tgt_path_vae_best):
    makedirs(tgt_path_vae_best)

latent_dim = args.latent_dim
orig_weight_factor = 1
var_inv = 1

x = int(latent_dim / 2)
y = int(latent_dim / 4)

# Real
mean1 = torch.zeros(int(latent_dim)).cuda()
mean1[:x] = 1
mean1[x:] = 0

# Fake
mean2 = torch.zeros(int(latent_dim)).cuda()
mean2[:x] = 0
mean2[x:] = 1

DIV_LOSSES = {
    'kl': kl_with_gaussian_unit_std,
    'wasserstein': wasserstein_distance
}
div_loss = DIV_LOSSES[args.div_loss]

# Create model objects
tgt_vae_model = Encoder(latent_dim=latent_dim).to(device)

# Freeze all layers of target vae model
# for param in tgt_vae_model.parameters():
#     param.requires_grad = False
#
# # set the FC layer + last block gradients to true
# for param in tgt_vae_model.fc1.parameters():
#     param.requires_grad = True
# for param in tgt_vae_model.fc2.parameters():
#     param.requires_grad = True
# for param in tgt_vae_model.avgpool.parameters():
#     param.requires_grad = True
# for param in tgt_vae_model.layer3.parameters():
#     param.requires_grad = True
# print(sum(p.numel() for p in tgt_vae_model.parameters() if p.requires_grad))


# Optmizers
tgt_vae_optimizer = optim.Adam(tgt_vae_model.parameters(), lr=1e-5)


# scheduler = ReduceLROnPlateau(tgt_vae_optimizer, mode='min', factor=0.9, patience=10, verbose=True)


def loss_function(mu, logvar, labels):
    mu = torch.abs(mu)

    orig_indices = (labels == 0).nonzero().squeeze(1)
    fake_indices = (labels == 1).nonzero().squeeze(1)

    mu_orig = torch.index_select(mu, dim=0, index=orig_indices, out=None)
    mu_fake = torch.index_select(mu, dim=0, index=fake_indices, out=None)

    logvar_orig = torch.index_select(logvar, dim=0, index=orig_indices, out=None)
    logvar_fake = torch.index_select(logvar, dim=0, index=fake_indices, out=None)

    kl_orig = orig_weight_factor * div_loss(mu=mu_orig, logvar=logvar_orig, mean=mean1)
    kl_fake = div_loss(mu=mu_fake, logvar=logvar_fake, mean=mean2)

    real_count = mu_orig.shape[0]
    fake_count = mu_fake.shape[0]
    kl_loss = (kl_orig + kl_fake) / (fake_count + orig_weight_factor * real_count)

    return kl_loss


def train_vae(epoch, tgt_vae_model):
    tgt_vae_model.train()
    train_loss = 0
    tbar = tqdm(target_train_loader)
    last_desc = 'Train'

    for batch_idx, (data, labels) in enumerate(tbar):
        data = data.to(device)
        labels = labels.to(device)
        labels[labels == 2] = 1
        labels[labels == 3] = 1
        labels[labels == 4] = 1
        tgt_vae_optimizer.zero_grad()
        z, mu, logvar = tgt_vae_model(data)
        loss = loss_function(mu, logvar, labels)
        loss.backward()
        train_loss += loss.item()
        tgt_vae_optimizer.step()
        if batch_idx % args.log_interval == 0:
            last_desc = 'Epoch: {} [({:.0f}%)] | Loss: {:.6f}'.format(epoch,
                                                                      100. * batch_idx / len(data),
                                                                      loss.item())
        tbar.set_description(last_desc)
    logger.log(mode="train", error=train_loss / len(target_train_loader), epoch=epoch, n_batch=0, num_batches=1,
               scalar='avg_loss')
    print('====> Train Epoch: {} Avg loss: {:.4f} '.format(epoch, train_loss / len(target_train_loader)))
    return train_loss / len(target_train_loader)


def fine_tune_vae_on_target(src_model_name, tgt_model_name):
    best_loss = np.Inf
    early_stop = False
    counter = 0
    try:
        print("Loading Source Models")
        checkpoint_vae_src = torch.load(src_path_vae + src_model_name)
        # Copy weights from source encoder to target encoder
        tgt_vae_model.load_state_dict(checkpoint_vae_src)
        print("Saved Model(s) successfully loaded")
    except:
        print("Model(s) not found.")
        exit()

    for epoch in range(1, args.epochs + 1):
        tgt_vae_model.train()
        train_loss = train_vae(epoch, tgt_vae_model)
        # scheduler.step(train_loss)
        if train_loss <= best_loss:
            counter = 0
            best_loss = train_loss
            torch.save(tgt_vae_model.state_dict(), tgt_path_vae_best + tgt_model_name)
            print("Best model saved/updated..")
        else:
            counter += 1
            print("EarlyStopping counter: " + str(counter) + " out of " + str(patience))
            if counter >= patience:
                early_stop = True
        # If early stopping flag is true, then stop the training
        if early_stop:
            print("Early stopping")
            break

        # if epoch % 10 == 0:
        #     visualize_latent_tsne(loader=tsne_loader, file_name=tsne_dir+"/abc_" + str(epoch), best_path=tgt_path_vae_best, model_name=vae_target, model=tgt_vae_model)


def test_classifier_vae_style(data_loader):
    tgt_vae_model.eval()
    total = 0
    correct = 0
    predictions = torch.tensor([], dtype=torch.long)
    labels_all = torch.tensor([], dtype=torch.long)
    with torch.no_grad():
        for i, (data, labels) in enumerate(tqdm(data_loader, desc='')):
            data = data.to(device)
            labels = labels.to(device)
            labels[labels == 2] = 1
            labels[labels == 3] = 1
            labels[labels == 4] = 1

            z, mu, logvar = tgt_vae_model(data)

            mu = torch.abs(mu)
            orig_indices = (labels == 0).nonzero().squeeze(1)
            fake_indices = (labels == 1).nonzero().squeeze(1)
            mu_orig = torch.index_select(mu, dim=0, index=orig_indices, out=None)
            mu_fake = torch.index_select(mu, dim=0, index=fake_indices, out=None)
            logvar_orig = torch.index_select(logvar, dim=0, index=orig_indices, out=None)
            logvar_fake = torch.index_select(logvar, dim=0, index=fake_indices, out=None)

            if mu_orig.shape[0] > 0:
                dist_orig_real = wasserstein_distance_vector(mu=mu_orig, logvar=logvar_orig, mean=mean1)
                dist_orig_fake = wasserstein_distance_vector(mu=mu_orig, logvar=logvar_orig, mean=mean2)
                dist_orig = torch.stack([dist_orig_real, dist_orig_fake], dim=1)
                _, predicted_real = torch.min(dist_orig, 1)
                labels_real = torch.zeros(mu_orig.shape[0]).long().cuda()

                total += mu_orig.shape[0]
                correct += (predicted_real == labels_real).sum().item()
                predictions = torch.cat([predictions, predicted_real.cpu()], dim=0)
                labels_all = torch.cat([labels_all, labels_real.cpu()], dim=0)

            if mu_fake.shape[0] > 0:
                dist_manipulated_real = wasserstein_distance_vector(mu=mu_fake, logvar=logvar_fake, mean=mean1)
                dist_manipulated_fake = wasserstein_distance_vector(mu=mu_fake, logvar=logvar_fake, mean=mean2)
                dist_manipulated = torch.stack([dist_manipulated_real, dist_manipulated_fake], dim=1)
                _, predicted_fake = torch.min(dist_manipulated, 1)
                labels_fake = torch.ones(mu_fake.shape[0]).long().cuda()

                total += mu_fake.shape[0]
                correct += (predicted_fake == labels_fake).sum().item()
                predictions = torch.cat([predictions, predicted_fake.cpu()], dim=0)
                labels_all = torch.cat([labels_all, labels_fake.cpu()], dim=0)

        accuracy = 100 * correct / total
        f = open("results.txt", "a+")
        f.write("%.3f" % (accuracy))
        f.write("\n")
        f.close()
    print('====>Accuracy: {:.4f}'.format(accuracy))
    cm = confusion_matrix(y_true=labels_all.cpu().numpy(), y_pred=predictions.cpu().numpy())
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print_confusion_matrix(confusion_matrix=cm, class_names=['Real', 'Fake'])


if __name__ == "__main__":
    # TRAIN
    # Method 1 : Directly fine-tune the network. Update all the layers
    fine_tune_vae_on_target(src_model_name=vae_source, tgt_model_name=vae_target)

    # VALIDATION
    # Evaluate after training
    # Load the target model
    # ************** TARGET **********************
    checkpoint_vae_tgt = torch.load(tgt_path_vae_best + vae_target)
    tgt_vae_model.load_state_dict(checkpoint_vae_tgt)
    test_classifier_vae_style(data_loader=target_test_loader)

    # Load the source classifier
    # visualize_latent_tsne(loader=tsne_loader, file_name="abc_"+str(args.run) , best_path=tgt_path_vae_best, model_name=vae_target, model=tgt_vae_model)

    # K-NN Accuracy
    # knn_classification(model=tgt_vae_model, k=7)

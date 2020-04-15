from __future__ import print_function
import argparse
import torch
import os
import torch.utils.data
import numpy as np
from common.utils.dataset import make_dataset
from torch.utils.data import DataLoader
from torch import optim
from os import makedirs
from torchvision import transforms
from common.logging.tf_logger import Logger
from tqdm import tqdm
import random
from common.utils.common_utils import calc_activation_vector
from common.models.resnet_subset_models import ForensicEncoder1
from common.losses.custom_losses import ActivationLoss

parser = argparse.ArgumentParser(description='Forensic Transfer')

parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')

parser.add_argument('--latent_dim', type=int, default=16, metavar='N',
                    help='latent embedding size (default: 128)')

parser.add_argument('-i', '--ft_images', type=int, default=10, metavar='N',
                    help='number of images for fine_tuning (default: 10)')

parser.add_argument('--epochs', type=int, default=10000, metavar='N',
                    help='number of epochs to train (default: 10)')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')

parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

parser.add_argument('-m', '--model_name', type=str,
                    default='NONE', help='name for model')

parser.add_argument('-r', '--run', type=str,
                    default='1', help='run number')

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


target_train_dataset = make_dataset(name='ff', base_path=train_path, num_classes=len(target_fake_classes)+1, fake_classes=target_fake_classes,
                                    mode='face_finetune', image_count=ft_images_train,
                                   transform=transforms.Compose([transforms.ToPILImage(),
                                                                 transforms.RandomHorizontalFlip(),
                                                                 transforms.RandomVerticalFlip(),
                                                                 transforms.ToTensor(),
                                                                 transforms.Normalize([0.5] * 3, [0.5] * 3)]))


target_test_dataset = make_dataset(name='ff', base_path=val_path, num_classes=len(target_fake_classes)+1, fake_classes=target_fake_classes,
                                    mode='face_finetune', image_count='all',
                                   transform=transforms.Compose(
                                       [transforms.ToPILImage(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.5] * 3, [0.5] * 3)]))

tsne_target_test_dataset = make_dataset(name='ff', base_path=val_path, num_classes=len(tsne_fake_classes) + 1, fake_classes=tsne_fake_classes,
                                        mode='face_finetune', image_count='all',
                                        transform=transforms.Compose(
                                            [transforms.ToPILImage(),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.5] * 3, [0.5] * 3)]))

batch_size = args.batch_size
target_train_loader = DataLoader(dataset=target_train_dataset, batch_size=batch_size, num_workers=8, shuffle=True)
target_test_loader = DataLoader(dataset=target_test_dataset, batch_size=batch_size, num_workers=8, shuffle=False)
tsne_loader = DataLoader(dataset=tsne_target_test_dataset, batch_size=batch_size, num_workers=8, shuffle=False)

logger = Logger(model_name='vae_model', data_name='ff', log_path=os.path.join(os.getcwd(), 'tf_logs/forensic_transfer/2classes_finetune/' + str(ft_images_train) + 'images/' + 'run_' + args.run + '/' + args.model_name))
src_model_name = 'ft_train_20k_val3k_mean1_std1_c23_latent16_3blocks_2classes_mixup_flip_normalize_df_nt.pt'
tgt_model_name = args.model_name + '.pt'

transfer_dir = 'df_nt_to_dessa_mixup'
# Paths
MODEL_PATH = os.path.join(os.getcwd(), 'models/')
src_path_ft = MODEL_PATH + 'forensic_transfer/2classes/best/'
tgt_path_ft_best = MODEL_PATH + 'forensic_transfer_finetune/2classes_' + str(ft_images_train) + 'images/' + transfer_dir +'/' + args.run + '_run/'

if not os.path.isdir(tgt_path_ft_best):
    makedirs(tgt_path_ft_best)

latent_dim = args.latent_dim
var_inv = 1

# Create model objects
tgt_ft_model = ForensicEncoder1(latent_dim=latent_dim).to(device)

# Freeze all layers of target vae model
# for param in tgt_ft_model.parameters():
#     param.requires_grad = False
#
# # set the FC layer + last block gradients to true
# for param in tgt_ft_model.fc1.parameters():
#     param.requires_grad = True
# for param in tgt_ft_model.avgpool.parameters():
#     param.requires_grad = True
# for param in tgt_ft_model.layer3.parameters():
#     param.requires_grad = True
# print(sum(p.numel() for p in tgt_ft_model.parameters() if p.requires_grad))

# Optmizers
tgt_vae_optimizer = optim.Adam(tgt_ft_model.parameters(), lr=1e-5)


def loss_function(z, y):
    act_vector = calc_activation_vector(latent_dim, z)
    y_onehot = torch.FloatTensor(y.size()[0], 2).cuda()
    y_onehot.zero_()
    y_onehot.scatter_(1, y.view(-1, 1), 1)
    weight = torch.ones(y.size()[0])
    act_loss = ActivationLoss(act_vector, y_onehot, weight)
    return act_loss


def train_forensic_transfer_epoch(epoch, tgt_ft_model):
    tgt_ft_model.train()
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
        z = tgt_ft_model(data)
        loss = loss_function(z, labels)
        loss.backward()
        train_loss += loss.item()
        tgt_vae_optimizer.step()
        if batch_idx % args.log_interval == 0:
            last_desc = 'Epoch: {} [({:.0f}%)] | Loss: {:.6f}'.format(epoch,
                                                                           50. * batch_idx / len(data),
                                                                           loss.item())
        tbar.set_description(last_desc)
    logger.log(mode="train", error=train_loss / len(target_train_loader), epoch=epoch, n_batch=0, num_batches=1,
               scalar='avg_loss')
    print('====> Train Epoch: {} Avg loss: {:.4f} '.format(epoch, train_loss / len(target_train_loader)))
    return train_loss / len(target_train_loader)


def fine_tune_forensic_transfer_on_target():
    best_loss = np.Inf
    early_stop = False
    counter = 0
    try:
        print("Loading Source Models")
        checkpoint_ft_src = torch.load(src_path_ft + src_model_name)
        # checkpoint_classifier_src = torch.load(src_path_classifier + src_model_name)
        # Load source encoder
        # src_ft_model.load_state_dict(checkpoint_vae_src)
        # Copy weights from source encoder to target encoder
        tgt_ft_model.load_state_dict(checkpoint_ft_src)
        # Load source classifier
        # classifier_model.load_state_dict(checkpoint_classifier_src)
        print("Saved Model(s) successfully loaded")
    except:
        print("Model(s) not found.")
        exit()

    for epoch in range(1, args.epochs + 1):
        tgt_ft_model.train()
        train_loss = train_forensic_transfer_epoch(epoch, tgt_ft_model)
        if train_loss <= best_loss:
            counter = 0
            best_loss = train_loss
            torch.save(tgt_ft_model.state_dict(), tgt_path_ft_best + tgt_model_name)
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


def test_classifier_forensic_style(data_loader):
    tgt_ft_model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for i, (data, labels) in enumerate(tqdm(data_loader, desc='')):
            data = data.to(device)
            labels = labels.to(device)
            labels[labels == 2] = 1
            labels[labels == 3] = 1
            labels[labels == 4] = 1
            z = tgt_ft_model(data)
            act_vector = calc_activation_vector(latent_dim, z)

            # Calculate correct predictions
            total += labels.size(0)
            _, predicted = torch.max(act_vector, 1)
            predicted[predicted == 2] = 1
            predicted[predicted == 3] = 1
            predicted[predicted == 4] = 1
            correct += (predicted == labels).sum().item()

        # Calculate accuracy for current epoch
        accuracy = 100 * correct / total
    print('====>Accuracy: {:.4f}'.format(accuracy))


if __name__ == "__main__":

    # TRAIN
    # Method 1 : Directly fine-tune the network.
    fine_tune_forensic_transfer_on_target()

    # VALIDATION
    # Evaluate after training
    # Load the target model and source classifier

    # ************** TARGET **********************
    checkpoint_vae_tgt = torch.load(tgt_path_ft_best + tgt_model_name)
    tgt_ft_model.load_state_dict(checkpoint_vae_tgt)
    test_classifier_forensic_style(data_loader=target_test_loader)
    # visualize_latent_tsne(loader=tsne_loader, file_name="abc_"+str(args.run), best_path=tgt_path_ft_best, model_name=tgt_model_name,
    #                       model=tgt_ft_model, mode='ft')

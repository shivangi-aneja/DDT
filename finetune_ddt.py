from __future__ import print_function
import argparse
import random
from os import makedirs
import numpy as np
import torch.utils.data
from torch import optim
from tqdm import tqdm
from config import *
from torch.utils.data import DataLoader
from common.utils.ddt_utils import ddt_loss
from common.logging.tf_logger import Logger
from common.losses.custom_losses import wasserstein_distance_vector
from common.models.resnet_subset_models import DDTEncoder1 as Encoder

parser = argparse.ArgumentParser(description='DDT Transfer: FaceForensics++ to others')
parser.add_argument('--train_mode', type=str, default='train', metavar='N',
                    help='training mode (train, test)')
parser.add_argument('--epochs', type=int, default=500, metavar='N',
                    help='number of epochs to train (default: 500)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
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
train_mode = args.train_mode

# Dataloaders
target_train_loader = DataLoader(dataset=target_train_dataset, batch_size=batch_size, num_workers=8, shuffle=True)
target_test_loader = DataLoader(dataset=target_test_dataset, batch_size=batch_size, num_workers=8, shuffle=False)


logger = Logger(model_name='ddt_model', data_name='ff', log_path=os.path.join(os.getcwd(), 'tf_logs/ddt/2classes_finetune/' + str(ft_images_train) + '_images/' + 'run_' + args.run + '/' + args.model_name))
ddt_source = 'ddt_train_20k_val3k_mean1_std1_c23_latent16_3blocks_2classes_mixup_flip_normalize_df_nt.pt'
ddt_target = args.model_name + '.pt'

# Paths
transfer_dir = 'df_nt_to_dessa'
src_path_ddt = MODEL_PATH + 'ddt/face/2classes/best/'
tgt_path_ddt_best = MODEL_PATH + 'ddt_finetune/2classes_' + str(ft_images_train) + 'images/' + transfer_dir + '/' + args.run + '_run/'

if not os.path.isdir(tgt_path_ddt_best):
    makedirs(tgt_path_ddt_best)


# Models and Optmiziers
tgt_ddt_model = Encoder(latent_dim=latent_dim).to(device)
tgt_ddt_optimizer = optim.Adam(tgt_ddt_model.parameters(), lr=finetune_lr)


def train_ddt(epoch, tgt_ddt_model):
    tgt_ddt_model.train()
    train_loss = 0
    tbar = tqdm(target_train_loader)
    last_desc = 'Train'

    for batch_idx, (data, labels) in enumerate(tbar):
        data = data.to(device)
        labels = labels.to(device)
        labels[labels == 2] = 1
        labels[labels == 3] = 1
        labels[labels == 4] = 1
        tgt_ddt_optimizer.zero_grad()
        z, mu, logvar = tgt_ddt_model(data)
        loss = ddt_loss(mu, logvar, labels)
        loss.backward()
        train_loss += loss.item()
        tgt_ddt_optimizer.step()
        if batch_idx % args.log_interval == 0:
            last_desc = 'Epoch: {} [({:.0f}%)] | Loss: {:.6f}'.format(epoch,
                                                                      100. * batch_idx / len(data),
                                                                      loss.item())
        tbar.set_description(last_desc)
    logger.log(mode="train", error=train_loss / len(target_train_loader), epoch=epoch, n_batch=0, num_batches=1,
               scalar='avg_loss')
    print('====> Train Epoch: {} Avg loss: {:.4f} '.format(epoch, train_loss / len(target_train_loader)))
    return train_loss / len(target_train_loader)


def fine_tune_ddt_on_target(src_model_name, tgt_model_name):
    best_loss = np.Inf
    early_stop = False
    counter = 0
    try:
        print("Loading Source Models")
        checkpoint_ddt_src = torch.load(src_path_ddt + src_model_name)
        # Copy weights from source encoder to target encoder
        tgt_ddt_model.load_state_dict(checkpoint_ddt_src)
        print("Saved Model(s) successfully loaded")
    except:
        print("Model(s) not found.")
        exit()

    for epoch in range(1, args.epochs + 1):
        tgt_ddt_model.train()
        train_loss = train_ddt(epoch, tgt_ddt_model)
        # scheduler.step(train_loss)
        if train_loss <= best_loss:
            counter = 0
            best_loss = train_loss
            torch.save(tgt_ddt_model.state_dict(), tgt_path_ddt_best + tgt_model_name)
            print("Best model saved/updated..")
        else:
            counter += 1
            print("EarlyStopping counter: " + str(counter) + " out of " + str(finetune_patience))
            if counter >= finetune_patience:
                early_stop = True
        # If early stopping flag is true, then stop the training
        if early_stop:
            print("Early stopping")
            break


def test_classifier_ddt_style(data_loader):
    tgt_ddt_model.eval()
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

            z, mu, logvar = tgt_ddt_model(data)

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


if __name__ == "__main__":
    # TRAIN
    # Method 1 : Directly fine-tune the network.
    if train_mode == 'train':
        fine_tune_ddt_on_target(src_model_name=ddt_source, tgt_model_name=ddt_target)

    # VALIDATION
    # Evaluate after training
    # Load the target model
    # ************** TARGET **********************
    elif train_mode == 'test':
        checkpoint_ddt_tgt = torch.load(tgt_path_ddt_best + ddt_target)
        tgt_ddt_model.load_state_dict(checkpoint_ddt_tgt)
        test_classifier_ddt_style(data_loader=target_test_loader)
    else:
        print("Sorry!! Invalid Mode..")


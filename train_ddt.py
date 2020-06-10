from __future__ import print_function
import argparse
import torch.utils.data
import numpy as np
from os import makedirs
from torch import optim
from common.logging.tf_logger import Logger
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from config import *
from common.utils.ddt_utils import ddt_loss
from common.models.resnet_subset_models import DDTEncoder1 as Encoder
from common.losses.custom_losses import wasserstein_distance_vector

parser = argparse.ArgumentParser(description='DDT FaceForensics++')
parser.add_argument('--train_mode', type=str, default='train', metavar='N',
                    help='training mode (train, test)')
parser.add_argument('--epochs', type=int, default=500, metavar='N',
                    help='number of epochs to train (default: 500)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

# Parse Arguments
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_mode = args.train_mode

# Models
model_name = 'ddt_train_20k_val3k_mean1_std1_c23_latent' + str(latent_dim) + '_3blocks_2classes_flip_normalize_nt'
logger = Logger(model_name='ddt_model', data_name='ff',
                log_path=os.path.join(os.getcwd(), 'tf_logs/ddt/2classes/' + model_name))
model_name = model_name + '.pt'
best_path = MODEL_PATH + 'ddt/' + dataset_mode + '/2classes/best/'
if not os.path.isdir(best_path):
    makedirs(best_path)

# Models and optimizers
model = Encoder(latent_dim=latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=train_lr)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience, verbose=True)


def test_classifier_ddt_style():
    try:
        print("Loading Saved Models")
        checkpoint_ddt = torch.load(best_path + model_name)
        model.load_state_dict(checkpoint_ddt)
        print("Saved Model successfully loaded")
    except:
        print("Model(s) not found.")
        exit()
    model.eval()
    total = 0
    correct = 0
    predictions = torch.tensor([], dtype=torch.long)
    labels_all = torch.tensor([], dtype=torch.long)
    with torch.no_grad():
        for i, (data, labels) in enumerate(tqdm(test_loader, desc='')):
            data = data.to(device)
            labels = labels.to(device)

            z, mu, logvar = model(data)

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
    print('====>Accuracy: {:.4f}'.format(accuracy))


def train_ddt(epoch, train_loader):
    model.train()
    train_loss = 0
    tbar = tqdm(train_loader)
    last_desc = 'Train'
    for batch_idx, (data, labels) in enumerate(tbar):
        data = data.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        z, mu, logvar = model(data)
        loss = ddt_loss(mu, logvar, labels)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            last_desc = 'Epoch: {} [({:.0f}%)] | Loss: {:.6f}'.format(epoch,
                                                                      100. * batch_idx / len(data),
                                                                      loss.item())
        tbar.set_description(last_desc)
    logger.log(mode="train", error=train_loss / len(train_loader), epoch=epoch, n_batch=0, num_batches=1,
               scalar='avg_loss')
    print('====> Train Epoch: {} Avg loss: {:.4f}   KL loss: {:.4f}   '.format(epoch, train_loss / len(train_loader),
                                                                               train_loss / len(train_loader)))
    return train_loss / len(train_loader)


def test_ddt(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, labels) in enumerate(tqdm(test_loader, desc='')):
            data = data.to(device)
            labels = labels.to(device)
            z, mu, logvar = model(data)
            loss = ddt_loss(mu, logvar, labels)
            test_loss += loss.item()

    logger.log(mode="test", error=test_loss / len(test_loader), epoch=epoch, n_batch=0, num_batches=1,
               scalar='avg_loss')
    logger.log(mode="test", error=float(optimizer.state_dict()['param_groups'][0]['lr']), epoch=epoch, n_batch=0,
               num_batches=1,
               scalar='lr')
    print('====> Val Epoch: {} Avg loss: {:.4f} '.format(epoch, test_loss / len(test_loader)))
    return test_loss / len(test_loader)


def train_model_ddt():
    best_loss = np.Inf
    early_stop = False
    counter = 0
    for epoch in range(1, args.epochs + 1):

        model.train()
        train_ddt(epoch, train_loader)

        model.eval()
        avg_test_loss = test_ddt(epoch)
        scheduler.step(avg_test_loss)

        # Save the current model
        # torch.save(model.state_dict(), current_path + model_name)
        if avg_test_loss <= best_loss:
            counter = 0
            model.train()
            best_loss = avg_test_loss
            torch.save(model.state_dict(), best_path + model_name)
            print("Best model saved/updated..")
        else:
            counter += 1
            print("EarlyStopping counter: " + str(counter) + " out of " + str(train_patience))
            if counter >= train_patience:
                early_stop = True
        # If early stopping flag is true, then stop the training
        if early_stop:
            print("Early stopping")
            break


if __name__ == "__main__":
    if train_mode == 'train':
        train_model_ddt()
    elif train_mode == 'test':
        test_classifier_ddt_style()
    else:
        print("Sorry!! Invalid Mode..")

from __future__ import print_function
import argparse
import torch
import os
import torch.utils.data
import numpy as np
from common.utils.dataset import make_dataset
from torch.utils.data import DataLoader
from torch import optim
from create_plot import print_confusion_matrix
from os import makedirs
from torchvision import transforms
from common.logging.tf_logger import Logger
from common.utils.common_utils import calc_activation_vector
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from common.utils.common_utils import visualize_latent_tsne
from common.models.resnet_subset_models import VariationalEncoder1 as Encoder
# from common.models.resnet_models import ResNet18VariationalEncoder
from common.models.classifiers import CLASSIFIER
from common.losses.custom_losses import wasserstein_distance, kl_with_gaussian_unit_std, wasserstein_distance_vector

parser = argparse.ArgumentParser(description='VAE FaceForensics++ ')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--latent_dim', '-l', type=int, default=16, metavar='N',
                    help='latent embedding size (default: 128)')
parser.add_argument('--dataset_mode', type=str, default='face', metavar='N',
                    help='dataset mode (face, face_residual, lip)')
parser.add_argument('--epochs', type=int, default=10000, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--num_classes', type=int, default=2, metavar='N',
                    help='Number of classes (N fakes + 1 real)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('-div_loss', '--div_loss', type=str,
                    default='wasserstein', help='Divergence Loss')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
dataset_mode = args.dataset_mode

fake_classes = ['nt']
print(fake_classes)
num_classes = len(fake_classes) + 1
DIV_LOSSES = {
    'kl': kl_with_gaussian_unit_std,
    'wasserstein': wasserstein_distance
}

# Data Path and Loaders
train_path = '/home/shivangi/Desktop/Projects/master_thesis/data/ff_face_20k/c23/train_20k_c23/'
val_path = '/home/shivangi/Desktop/Projects/master_thesis/data/ff_face_20k/c23/val_6k_c23/'
train_dataset = make_dataset(name='ff', base_path=train_path, num_classes=num_classes, fake_classes=fake_classes,
                             mode='face', image_count='all',
                             transform=transforms.Compose([transforms.ToPILImage(),
                                                           # transforms.RandomResizedCrop(224),
                                                           # transforms.ColorJitter(hue=.25, saturation=.25),
                                                           transforms.RandomHorizontalFlip(),
                                                           transforms.RandomVerticalFlip(),
                                                           transforms.ToTensor(),
                                                           transforms.Normalize([0.5] * 3, [0.5] * 3),
                                                           # transforms.RandomErasing()
                                                           ]))

test_dataset = make_dataset(name='ff', base_path=val_path, num_classes=num_classes, fake_classes=fake_classes,
                            mode='face', image_count='all',
                            transform=transforms.Compose(
                                [transforms.ToPILImage(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.5] * 3, [0.5] * 3)]))

tsne_dataset = make_dataset(name='ff', base_path=val_path, num_classes=num_classes, fake_classes=fake_classes,
                            mode='face', image_count=1000,
                            transform=transforms.Compose(
                                [transforms.ToPILImage(),
                                 transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)]))
batch_size = args.batch_size
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=16, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=32, shuffle=False)
tsne_loader = DataLoader(dataset=tsne_dataset, batch_size=batch_size, num_workers=8, shuffle=False)

# Paths
MODEL_PATH = os.path.join(os.getcwd(), 'models/')
best_path = MODEL_PATH + 'vae/' + dataset_mode + '/2classes/best/'

if not os.path.isdir(best_path):
    makedirs(best_path)

latent_dim = args.latent_dim
orig_weight_factor = num_classes - 1

model_name = 'vae_train_20k_val3k_mean1_std1_c23_latent'+ str(latent_dim) + '_3blocks_2classes_flip_normalize_nt'
logger = Logger(model_name='vae_model', data_name='ff', log_path=os.path.join(os.getcwd(), 'tf_logs/vae/2classes/'+model_name))
model_name = model_name + '.pt'

# Real
mean1 = torch.zeros(int(latent_dim)).cuda()
mean1[:int(latent_dim / 2)] = 1
mean1[int(latent_dim / 2):] = 0
# Fake
mean2 = torch.zeros(int(latent_dim)).cuda()
mean2[:int(latent_dim / 2)] = 0
mean2[int(latent_dim / 2):] = 1

# Losses
class_weights = torch.Tensor([orig_weight_factor, 1]).cuda()
div_loss = DIV_LOSSES[args.div_loss]

# Models and optimizers
model = Encoder(latent_dim=latent_dim).to(device)
classifier = CLASSIFIER(latent_dim=latent_dim).to(device)
# VAE optimizer
vae_lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=vae_lr)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=10, verbose=True)


# divergence loss summed over all elements and batch
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

    loss = kl_loss
    return loss


def test_classifier_vae_style():

    try:
        print("Loading Saved Models")
        checkpoint_vae = torch.load(best_path + model_name)
        model.load_state_dict(checkpoint_vae)
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
            labels[labels == 2] = 1
            labels[labels == 3] = 1
            labels[labels == 4] = 1

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
    cm = confusion_matrix(y_true=labels_all.cpu().numpy(), y_pred=predictions.cpu().numpy())
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print_confusion_matrix(confusion_matrix=cm, class_names=['Real', 'Fake'], filename='vae_cm')


def train_vae(epoch, train_loader):
    model.train()
    train_loss = 0
    tbar = tqdm(train_loader)
    last_desc = 'Train'
    for batch_idx, (data, labels) in enumerate(tbar):
        data = data.to(device)
        labels = labels.to(device)
        labels[labels == 2] = 1
        labels[labels == 3] = 1
        labels[labels == 4] = 1
        optimizer.zero_grad()
        z, mu, logvar = model(data)
        loss = loss_function(mu, logvar, labels)
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


def test_vae(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, labels) in enumerate(tqdm(test_loader, desc='')):
            data = data.to(device)
            labels = labels.to(device)
            labels[labels == 2] = 1
            labels[labels == 3] = 1
            labels[labels == 4] = 1
            z, mu, logvar = model(data)
            loss = loss_function(mu, logvar, labels)
            test_loss += loss.item()

    logger.log(mode="test", error=test_loss / len(test_loader), epoch=epoch, n_batch=0, num_batches=1,
               scalar='avg_loss')
    logger.log(mode="test", error=float(optimizer.state_dict()['param_groups'][0]['lr']), epoch=epoch, n_batch=0,
               num_batches=1,
               scalar='lr')
    print('====> Val Epoch: {} Avg loss: {:.4f} '.format(epoch, test_loss / len(test_loader)))
    return test_loss / len(test_loader)


def test_vae_after_training():
    try:
        print("Loading Saved Model")
        print(best_path)
        checkpoint = torch.load(best_path + model_name)
        model.load_state_dict(checkpoint)
        print("Saved Model successfully loaded")
    except:
        print("Model not found.")
        exit()
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, labels) in enumerate(tqdm(test_loader, desc='')):
            data = data.to(device)
            labels = labels.to(device)
            z, mu, logvar = model(data)
            loss = loss_function(mu, logvar, labels)
            test_loss += loss.item()
    print(
        '====> Avg loss: {:.4f}   KL loss: {:.4f}  '.format(test_loss / len(test_loader), test_loss / len(test_loader)))
    return test_loss / len(test_loader)


def train_model_vae():
    patience = 10
    best_loss = np.Inf
    early_stop = False
    counter = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = train_vae(epoch, train_loader)
        model.eval()
        avg_test_loss = test_vae(epoch)
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
            print("EarlyStopping counter: " + str(counter) + " out of " + str(patience))
            if counter >= patience:
                early_stop = True
        # If early stopping flag is true, then stop the training
        if early_stop:
            print("Early stopping")
            break
        # if epoch % 10 == 0:
        #     visualize_latent_tsne(loader=tsne_loader, file_name="abc_" + str(epoch), best_path=best_path, model_name=model_name, model=model)

def test_classifier_forensic_style():
    try:
        print("Loading Saved Models")
        checkpoint_vae = torch.load(best_path + model_name)
        model.load_state_dict(checkpoint_vae)
        print("Saved Model successfully loaded")
    except:
        print("Model(s) not found.")
        exit()
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for i, (data, labels) in enumerate(tqdm(test_loader, desc='')):
            data = data.to(device)
            labels = labels.to(device)
            labels[labels == 2] = 1
            labels[labels == 3] = 1
            labels[labels == 4] = 1
            z, _, _ = model(data)
            act_vector = calc_activation_vector(latent_dim, z)

            # Calculate correct predictions
            total += labels.size(0)
            _, predicted = torch.max(act_vector, 1)
            # predicted[predicted == fake_label] = 1
            correct += (predicted == labels).sum().item()

        # Calculate accuracy for current epoch
        accuracy = 100 * correct / total
    print('====>Accuracy: {:.4f}'.format(accuracy))


if __name__ == "__main__":

    train_model_vae()
    test_vae_after_training()
    test_classifier_vae_style()
    visualize_latent_tsne(loader=tsne_loader, file_name="vae_vis", best_path=best_path, model_name=model_name, model=model)

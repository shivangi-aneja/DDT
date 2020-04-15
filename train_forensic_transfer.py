from __future__ import print_function
import argparse
import torch
import os
import torch.utils.data
import numpy as np
from common.utils.common_utils import calc_activation_vector
from common.utils.dataset import make_dataset
from torch.utils.data import DataLoader
from torch import nn, optim
from os import makedirs
from torchvision import transforms
from common.logging.tf_logger import Logger
from tqdm import tqdm
from common.utils.common_utils import visualize_latent_tsne
from common.models.resnet_subset_models import ForensicEncoder1
# from common.models.resnet_models import ResNet18Encoder
# from common.models.networks import Autoencoder3 as AutoencoderNetwork
from torch.optim.lr_scheduler import ReduceLROnPlateau
from common.models.classifiers import CLASSIFIER
from common.losses.custom_losses import ActivationLoss
from sklearn.metrics import confusion_matrix
from create_plot import print_confusion_matrix

parser = argparse.ArgumentParser(description='Forensic Transfer FaceForensics++ Dataset')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--latent_dim', '-l', type=int, default=16, metavar='N',
                    help='latent embedding size (default: 128)')
parser.add_argument('--epochs', type=int, default=10000, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
# decay_epochs = 40

train_path = '/home/shivangi/Desktop/Projects/master_thesis/data/ff_face_20k/c23/train_20k_c23/'
val_path = '/home/shivangi/Desktop/Projects/master_thesis/data/ff_face_20k/c23/test/'

fake_classes = ['df']
num_classes = len(fake_classes) + 1
print(fake_classes)

train_dataset = make_dataset(name='ff', base_path=train_path, num_classes=num_classes, fake_classes=fake_classes,
                             mode='face', image_count='all',
                             transform=transforms.Compose([transforms.ToPILImage(),
                                                           transforms.RandomHorizontalFlip(),
                                                           transforms.RandomVerticalFlip(),
                                                           transforms.ToTensor(),
                                                           transforms.Normalize([0.5] * 3, [0.5] * 3)
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

batch_size = 128
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=32, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=32, shuffle=False)
tsne_loader = DataLoader(dataset=tsne_dataset, batch_size=batch_size, num_workers=8, shuffle=False)


# Paths
MODEL_PATH = os.path.join(os.getcwd(), 'models/')

best_path = MODEL_PATH + 'forensic_transfer/2classes/best/'
best_path_classifier = MODEL_PATH + 'forensic_transfer_classifier/2classes/best/'

if not os.path.isdir(best_path):
    makedirs(best_path)

if not os.path.isdir(best_path_classifier):
    makedirs(best_path_classifier)

latent_dim = args.latent_dim
orig_weight_factor = num_classes -1
model_name = 'ft_train_20k_val3k_mean1_std1_c23_latent' + str(latent_dim) + '_3blocks_2classes_flip_normalize_nt'
logger = Logger(model_name='vae_model', data_name='ff', log_path=os.path.join(os.getcwd(), 'tf_logs/forensic_transfer/2classes/'+model_name))
model_name = model_name + '.pt'

# Losses
class_weights = torch.Tensor([orig_weight_factor, 1]).cuda()
# classification_loss = nn.CrossEntropyLoss(reduction='mean', weight=class_weights)

# Create model objects
ft_model = ForensicEncoder1(latent_dim=latent_dim).to(device)
ft_classifier = CLASSIFIER(latent_dim=latent_dim).to(device)

# VAE optimizer
vae_lr = 1e-3
optimizer = optim.Adam(ft_model.parameters(), lr=vae_lr)
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=10, verbose=True)
# optimizer_classifier = optim.Adam(ft_classifier.parameters(), lr=1e-4)


def loss_function(z, y):
    act_vector = calc_activation_vector(latent_dim, z)
    orig_indices = (y == 0).nonzero().squeeze(1)
    weight = torch.ones(y.size()[0])
    weight.put_(orig_indices.cpu(), torch.Tensor(len(orig_indices)).fill_(orig_weight_factor)).cuda()
    y_onehot = torch.FloatTensor(y.size()[0], 2).cuda()
    y_onehot.zero_()
    y_onehot.scatter_(1, y.view(-1, 1), 1)
    act_loss = ActivationLoss(act_vector, y_onehot, weight)
    return act_loss


def train_forensic_transfer_epoch(epoch, train_loader):
    ft_model.train()
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
        z = ft_model(data)
        loss = loss_function(z, labels)
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
    print('====> Train Epoch: {} Avg loss: {:.4f}  '.format(epoch, train_loss / len(train_loader)))
    return train_loss / len(train_loader)


def test_forensic_transfer_epoch(epoch):
    ft_model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, labels) in enumerate(tqdm(test_loader, desc='')):
            data = data.to(device)
            labels = labels.to(device)
            labels[labels == 2] = 1
            labels[labels == 3] = 1
            labels[labels == 4] = 1
            z = ft_model(data)
            loss = loss_function(z, labels)
            test_loss += loss.item()

    logger.log(mode="test", error=test_loss / len(test_loader), epoch=epoch, n_batch=0, num_batches=1,
               scalar='avg_loss')
    logger.log(mode="test", error=float(optimizer.state_dict()['param_groups'][0]['lr']), epoch=epoch, n_batch=0,
               num_batches=1,
               scalar='lr')
    print('====> Val Epoch: {} Avg loss: {:.4f} '.format(epoch, test_loss / len(test_loader)))
    return test_loss / len(test_loader)


def test_forensic_transfer_after_training():
    try:
        print("Loading Saved Model")
        print(best_path)
        checkpoint = torch.load(best_path + model_name)
        ft_model.load_state_dict(checkpoint)
        print("Saved Model successfully loaded")
    except:
        print("Model not found.")
        exit()
    ft_model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, labels) in enumerate(tqdm(test_loader, desc='')):
            data = data.to(device)
            labels = labels.to(device)
            z = ft_model(data)
            loss = loss_function(z, labels)
            test_loss += loss.item()
    print(
        '====> Avg loss: {:.4f}   KL loss: {:.4f}  '.format(test_loss / len(test_loader), test_loss / len(test_loader)))
    return test_loss / len(test_loader)


def train_forensic_transfer():
    patience = 10
    best_loss = np.Inf
    early_stop = False
    counter = 0
    for epoch in range(1, args.epochs + 1):
        ft_model.train()
        train_loss = train_forensic_transfer_epoch(epoch, train_loader)
        ft_model.eval()
        avg_test_loss = test_forensic_transfer_epoch(epoch)
        # scheduler.step(avg_test_loss)
        if avg_test_loss <= best_loss:
            counter = 0
            best_loss = avg_test_loss
            torch.save(ft_model.state_dict(), best_path + model_name)
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
        #     visualize_latent_tsne(loader=tsne_loader, file_name="abc_" + str(epoch), best_path=best_path, model_name=model_name, model=ft_model)


def train_classifier_epoch(epoch):
    train_loss = 0
    total = 0
    correct = 0
    ft_classifier.train()
    # Training loop
    for batch_idx, (data, labels) in enumerate(train_loader):
        data = data.to(device)
        labels = labels.to(device)
        labels[labels == 2] = 1
        labels[labels == 3] = 1
        labels[labels == 4] = 1
        z = ft_model(data)
        label_hat = ft_classifier(z)
        loss = classification_loss(label_hat, labels)
        loss.backward()
        train_loss += loss.item()
        optimizer_classifier.step()
        _, predicted = torch.max(label_hat, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item() / len(train_loader)))
    # Calculate accuracy for current epoch
    accuracy = 100 * correct / total
    logger.log(mode="train", error=train_loss / len(train_loader), epoch=epoch, n_batch=0, num_batches=1,
               scalar='avg_classification_loss')
    logger.log(mode="train", error=accuracy, epoch=epoch, n_batch=0, num_batches=1,
               scalar='classification_accuracy')

    print(
        '====> Train Epoch: {} Loss: {:.4f}   Accuracy: {:.4f}'.format(epoch, train_loss / len(train_loader), accuracy))


def test_classifier_epoch(epoch):
    ft_classifier.eval()
    test_loss = 0
    total = 0
    correct = 0
    with torch.no_grad():
        for i, (data, labels) in enumerate(test_loader):
            data = data.to(device)
            labels = labels.to(device)
            labels[labels == 2] = 1
            labels[labels == 3] = 1
            labels[labels == 4] = 1
            z = ft_model(data)
            label_hat = ft_classifier(z)
            loss = classification_loss(label_hat, labels)
            test_loss += loss.item()

            _, predicted = torch.max(label_hat, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        # Calculate accuracy for current epoch
        accuracy = 100 * correct / total
    logger.log(mode="test", error=test_loss / len(train_loader), epoch=epoch, n_batch=0, num_batches=1,
               scalar='avg_classification_loss')
    logger.log(mode="test", error=accuracy, epoch=epoch, n_batch=0, num_batches=1,
               scalar='classification_accuracy')
    print(
        '====> Val Epoch: {} Avg loss: {:.4f}  Accuracy: {:.4f}'.format(epoch, test_loss / len(train_loader), accuracy))
    return test_loss


def test_classifier_after_training():
    try:
        print("Loading Saved Models")
        checkpoint_vae = torch.load(best_path + model_name)
        ft_model.load_state_dict(checkpoint_vae)
        checkpoint_classifier = torch.load(best_path_classifier + model_name)
        ft_classifier.load_state_dict(checkpoint_classifier)
        print("Saved Model successfully loaded")
    except:
        print("Model(s) not found.")
        exit()
    ft_model.eval()
    ft_classifier.eval()
    test_loss = 0
    total = 0
    correct = 0
    with torch.no_grad():
        for i, (data, labels) in enumerate(tqdm(test_loader, desc='')):
            data = data.to(device)
            labels = labels.to(device)
            labels[labels == 2] = 1
            labels[labels == 3] = 1
            labels[labels == 4] = 1
            z = ft_model(data)
            label_hat = ft_classifier(z)
            loss = classification_loss(label_hat, labels)
            test_loss += loss.item()

            _, predicted = torch.max(label_hat, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            test_loss += loss.item()

        # Calculate accuracy for current epoch
        accuracy = 100 * correct / total
    print('====> Avg loss: {:.4f}   Accuracy: {:.4f}'.format(test_loss / len(test_loader), accuracy))
    return test_loss


def test_classifier_forensic_style():
    try:
        print("Loading Saved Models")
        checkpoint_vae = torch.load(best_path + model_name)
        ft_model.load_state_dict(checkpoint_vae)
        print("Saved Model successfully loaded")
    except:
        print("Model(s) not found.")
        exit()
    ft_model.eval()
    ft_classifier.eval()
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
            z = ft_model(data)
            act_vector = calc_activation_vector(latent_dim, z)

            # Calculate correct predictions
            total += labels.size(0)
            _, predicted = torch.max(act_vector, 1)
            # predicted[predicted == fake_label] = 1
            correct += (predicted == labels).sum().item()
            predictions = torch.cat([predictions, predicted.cpu()], dim=0)
            labels_all = torch.cat([labels_all, labels.cpu()], dim=0)

        # Calculate accuracy for current epoch
        accuracy = 100 * correct / total
    print('====>Accuracy: {:.4f}'.format(accuracy))
    cm = confusion_matrix(y_true=labels_all.cpu().numpy(), y_pred=predictions.cpu().numpy())
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print_confusion_matrix(confusion_matrix=cm, class_names=['Real', 'Fake'], filename='forensic_transfer_cm')


def train_classifier():
    try:
        print("Loading Saved VAE Model")
        print(best_path)
        checkpoint = torch.load(best_path + model_name)
        ft_model.load_state_dict(checkpoint)
        print("Saved Model successfully loaded")
    except:
        print("Model not found.")
        exit()
    # Freeze all layers of resnet model
    for param in ft_model.parameters():
        param.requires_grad = False
    ft_model.eval()
    patience = 30
    best_loss = np.Inf
    early_stop = False
    counter = 0
    for epoch in range(1, args.epochs + 1):
        # Training epoch
        train_classifier_epoch(epoch)
        # Validation epoch
        avg_test_loss = test_classifier_epoch(epoch)
        if avg_test_loss <= best_loss:
            counter = 0
            best_loss = avg_test_loss
            torch.save(ft_classifier.state_dict(), best_path_classifier + model_name)
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


if __name__ == "__main__":
    # train_forensic_transfer()
    # train_classifier()
    # test_classifier_after_training()
    # test_classifier_forensic_style()
    visualize_latent_tsne(loader=tsne_loader, file_name="df_ft", best_path=best_path, model_name=model_name, model=ft_model, mode='ft')

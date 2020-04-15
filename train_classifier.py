from __future__ import print_function
import argparse
import torch
import os
import torch.utils.data
import numpy as np
from common.utils.dataset import make_dataset
from common.utils.common_utils import visualize_latent_tsne_classifier
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch import nn, optim
from os import makedirs
from torchvision import transforms
from common.logging.tf_logger import Logger
from tqdm import tqdm
from common.models.resnet_subset_models import EncoderLatent as Encoder1
from sklearn.metrics import confusion_matrix
from create_plot import print_confusion_matrix

parser = argparse.ArgumentParser(description='FaceForensics++ ResNet Classifier')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10000, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
decay_epochs = 40

train_path = '/home/shivangi/Desktop/Projects/master_thesis/data/ff_face_20k/c23/train_20k_c23/'
val_path = '/home/shivangi/Desktop/Projects/master_thesis/data/ff_face_20k/c23/val_6k_c23/'

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
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=8, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=32, shuffle=False)
tsne_loader = DataLoader(dataset=tsne_dataset, batch_size=batch_size, num_workers=8, shuffle=False)

model = 'vae_train_20k_val3k_mean1_std1_c23_latent16_3blocks_2classes_flip_normalize_nt'
logger = Logger(model_name='vae_model', data_name='ff', log_path=os.path.join(os.getcwd(), 'tf_logs/classifier/2classes/'+model))
model_name = model + '.pt'

# Paths
MODEL_PATH = os.path.join(os.getcwd(), 'models/')
best_path = MODEL_PATH + 'classifier/face/2classes/best/'

if not os.path.isdir(best_path):
    makedirs(best_path)

orig_class_weight = num_classes - 1

# Losses
class_weights = torch.Tensor([orig_class_weight, 1]).cuda()
classification_loss = nn.CrossEntropyLoss(reduction='mean', weight=class_weights)

# Create model objects
resnet_classifier = Encoder1(latent_dim=16).to(device)

# VAE optimizer
lr = 1e-4
optimizer = optim.Adam(resnet_classifier.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5, verbose=True)


def train_classifier_epoch(epoch):
    train_loss = 0
    total = 0
    correct = 0
    tbar = tqdm(train_loader)
    last_desc = 'Train'
    resnet_classifier.train()
    # Training loop
    for batch_idx, (data, labels) in enumerate(tbar):
        data = data.to(device)
        labels = labels.to(device)
        labels[labels == 2] = 1
        labels[labels == 3] = 1
        labels[labels == 4] = 1
        _, label_hat = resnet_classifier(data)
        loss = classification_loss(label_hat, labels)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        _, predicted = torch.max(label_hat, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # Calculate accuracy for current epoch
    accuracy = 100 * correct / total
    logger.log(mode="train", error=train_loss / len(train_loader), epoch=epoch, n_batch=0, num_batches=1,
               scalar='avg_classification_loss')
    logger.log(mode="train", error=accuracy, epoch=epoch, n_batch=0, num_batches=1,
               scalar='classification_accuracy')

    print(
        '====> Train Epoch: {} Loss: {:.4f}   Accuracy: {:.4f}'.format(epoch, train_loss / len(train_loader), accuracy))


def test_classifier_epoch(epoch):
    resnet_classifier.eval()
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
            _, label_hat = resnet_classifier(data)
            loss = classification_loss(label_hat, labels)
            test_loss += loss.item()
            _, predicted = torch.max(label_hat, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        # Calculate accuracy for current epoch
        accuracy = 100 * correct / total
    logger.log(mode="test", error=test_loss / len(test_loader), epoch=epoch, n_batch=0, num_batches=1,
               scalar='avg_classification_loss')
    logger.log(mode="test", error=accuracy, epoch=epoch, n_batch=0, num_batches=1,
               scalar='classification_accuracy')
    print(
        '====> Val Epoch: {} Avg loss: {:.4f}  Accuracy: {:.4f}'.format(epoch, test_loss / len(test_loader), accuracy))
    return test_loss


def test_classifier_after_training():
    try:
        print("Loading Saved Classifier Model")
        checkpoint_classifier = torch.load(best_path + model_name)
        resnet_classifier.load_state_dict(checkpoint_classifier)
        print("Saved Model successfully loaded")
    except:
        print("Model(s) not found.")
        exit()
    resnet_classifier.eval()
    test_loss = 0
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
            _, label_hat = resnet_classifier(data)
            loss = classification_loss(label_hat, labels)
            test_loss += loss.item()

            _, predicted = torch.max(label_hat, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            test_loss += loss.item()
            predictions = torch.cat([predictions, predicted.cpu()], dim=0)
            labels_all = torch.cat([labels_all, labels.cpu()], dim=0)

        # Calculate accuracy for current epoch
        accuracy = 100 * correct / total
    print('====> Avg loss: {:.4f}   Accuracy: {:.4f}'.format(test_loss / len(test_loader), accuracy))
    cm = confusion_matrix(y_true=labels_all.cpu().numpy(), y_pred=predictions.cpu().numpy())
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print_confusion_matrix(confusion_matrix=cm, class_names=['Real', 'Fake'], filename='classifier_cm')


def train_classifier():

    patience = 10
    best_loss = np.Inf
    early_stop = False
    counter = 0
    for epoch in range(1, args.epochs + 1):
        # Training epoch
        train_classifier_epoch(epoch)
        # Validation epoch
        avg_test_loss = test_classifier_epoch(epoch)
        scheduler.step(avg_test_loss)
        if avg_test_loss <= best_loss:
            counter = 0
            best_loss = avg_test_loss
            torch.save(resnet_classifier.state_dict(), best_path + model_name)
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
    train_classifier()
    test_classifier_after_training()
    visualize_latent_tsne_classifier(loader=tsne_loader, file_name="classifier", best_path=best_path, model_name=model_name, model=resnet_classifier)

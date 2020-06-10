from __future__ import print_function
import argparse
import torch.utils.data
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import optim
from os import makedirs
from config import *
from common.logging.tf_logger import Logger
from tqdm import tqdm
from common.models.resnet_subset_models import EncoderLatent as Encoder1

parser = argparse.ArgumentParser(description='Classifier FaceForensics++')
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
model = 'classifier_train_20k_val3k_mean1_std1_c23_latent16_3blocks_2classes_flip_normalize_nt'
logger = Logger(model_name='classifier_model', data_name='ff', log_path=os.path.join(os.getcwd(), 'tf_logs/classifier/2classes/'+model))
model_name = model + '.pt'
MODEL_PATH = os.path.join(os.getcwd(), 'models/')
best_path = MODEL_PATH + 'classifier/' + dataset_mode + '/2classes/best/'
if not os.path.isdir(best_path):
    makedirs(best_path)

# Models and optimizers
resnet_classifier = Encoder1(latent_dim=latent_dim).to(device)
optimizer = optim.Adam(resnet_classifier.parameters(), lr=train_lr)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience, verbose=True)


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
    if train_mode == 'train':
        train_classifier()
    elif train_mode == 'test':
        test_classifier_after_training()
    else:
        print("Sorry!! Invalid Mode..")

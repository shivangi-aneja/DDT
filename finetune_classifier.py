from __future__ import print_function
import argparse
import torch.utils.data
import numpy as np
from torch import optim
from os import makedirs
from torch.utils.data import DataLoader
from config import *
from common.logging.tf_logger import Logger
from tqdm import tqdm
from common.models.resnet_subset_models import EncoderLatent as Encoder1
import random

parser = argparse.ArgumentParser(description='Classifier FineTuning')
parser.add_argument('--train_mode', type=str, default='train', metavar='N',
                    help='training mode (train, test)')
parser.add_argument('--epochs', type=int, default=500, metavar='N',
                    help='number of epochs to train (default: 500)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('-m', '--model_name', type=str,
                    default='classifier_model', help='name for model')
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


logger = Logger(model_name='classifier_model', data_name='ff', log_path=os.path.join(os.getcwd(), 'tf_logs/classifier/2classes_finetune/' + str(ft_images_train) + 'images/' + 'run_' + args.run + '/' + args.model_name))
src_classifier_name = 'classifier_c23_latent16_3blocks_2classes_flip_normalize_ff.pt'
tgt_classifier_name = args.model_name + '.pt'

transfer_dir = 'df_nt_to_dessa'

# Paths
src_path_classifier = MODEL_PATH + 'classifier/face/2classes/best/'
tgt_path_classifier = MODEL_PATH + 'classifier_finetune/' + transfer_dir + '/' + str(ft_images_train) + 'images/' + args.run + '_run/'

if not os.path.isdir(tgt_path_classifier):
    makedirs(tgt_path_classifier)

# Models and Optmiziers
classifier_model = Encoder1(latent_dim=latent_dim).to(device)
optimizer = optim.Adam(classifier_model.parameters(), lr=finetune_lr)


def train_classifier(epoch):
    classifier_model.train()
    train_loss = 0
    tbar = tqdm(target_train_loader)
    last_desc = 'Train'

    for batch_idx, (data, labels) in enumerate(tbar):
        data = data.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        _, label_hat = classifier_model(data)
        loss = classification_loss(label_hat, labels)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        tbar.set_description(last_desc)
    logger.log(mode="train", error=train_loss / len(target_train_loader), epoch=epoch, n_batch=0, num_batches=1,
               scalar='avg_loss')
    print('====> Train Epoch: {} Avg loss: {:.4f} '.format(epoch, train_loss / len(target_train_loader)))
    return train_loss / len(target_train_loader)


def fine_tune_classifier_on_target():
    best_loss = np.Inf
    early_stop = False
    counter = 0
    try:
        print("Loading Source Models")
        checkpoint_src = torch.load(src_path_classifier + src_classifier_name)
        classifier_model.load_state_dict(checkpoint_src)
        print("Saved Model successfully loaded")
    except:
        print("Model(s) not found.")
        exit()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_classifier(epoch)
        if train_loss <= best_loss:
            counter = 0
            best_loss = train_loss
            torch.save(classifier_model.state_dict(), tgt_path_classifier + tgt_classifier_name)
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


def test_classifier_after_training(data_loader):

    classifier_model.eval()
    test_loss = 0
    total = 0
    correct = 0
    with torch.no_grad():
        for i, (data, labels) in enumerate(tqdm(data_loader, desc='')):
            data = data.to(device)
            labels = labels.to(device)
            _, label_hat = classifier_model(data)
            loss = classification_loss(label_hat, labels)
            test_loss += loss.item()

            _, predicted = torch.max(label_hat, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            test_loss += loss.item()

        # Calculate accuracy for current epoch
        accuracy = 100 * correct / total
        f = open("results.txt", "a+")
        f.write("%.3f" % (accuracy))
        f.write("\n")
        f.close()
    print('====> Avg loss: {:.4f}   Accuracy: {:.4f}'.format(test_loss / len(data), accuracy))
    return accuracy, test_loss / len(data)


if __name__ == "__main__":

    # TRAIN
    # Method 1 : Directly fine-tune the network
    if train_mode == 'train':
        fine_tune_classifier_on_target()

    # ************** TARGET **********************
    elif train_mode == 'test':
        checkpoint_vae_tgt = torch.load(tgt_path_classifier + tgt_classifier_name)
        classifier_model.load_state_dict(checkpoint_vae_tgt)

        print("After fine-tuning, Target")
        test_classifier_after_training(data_loader=target_test_loader)

    else:
        print("Sorry!! Invalid Mode..")


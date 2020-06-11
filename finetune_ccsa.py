from __future__ import print_function
import argparse
import torch.utils.data
import numpy as np
from torch.utils.data import DataLoader
from torch import optim
from os import makedirs
from common.logging.tf_logger import Logger
import tqdm
from config import *
from common.losses.custom_losses import csa_loss
from common.models.resnet_subset_models import EncoderLatent as Encoder1
import random

parser = argparse.ArgumentParser(description='CCSA FineTuning')

parser.add_argument('--train_mode', type=str, default='train', metavar='N',
                    help='training mode (train, test)')
parser.add_argument('--epochs', type=int, default=500, metavar='N',
                    help='number of epochs to train (default: 500)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('-m', '--model_name', type=str,
                    default='ccsa_model', help='name for model')
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
source_train_loader = DataLoader(dataset=source_train_dataset, batch_size=batch_size_train, num_workers=8, shuffle=True)
target_train_loader = DataLoader(dataset=target_train_dataset, batch_size=batch_size_train, num_workers=8, shuffle=True)
target_test_loader = DataLoader(dataset=target_test_dataset, batch_size=batch_size, num_workers=8, shuffle=False)

logger = Logger(model_name='classifier_model', data_name='ff', log_path=os.path.join(os.getcwd(), 'tf_logs/ccsa_finetune/2classes_finetune/' + str(ft_images_train) + 'images/' + 'run_' + args.run + '/' + args.model_name))
src_classifier_name = 'classifier_c23_latent16_3blocks_2classes_flip_normalize_ff.pt'
tgt_classifier_name = args.model_name + '.pt'

transfer_dir = 'df_nt_to_dessa'

# Paths
src_path_classifier = MODEL_PATH + 'classifier/face/2classes/best/'
tgt_path_classifier = MODEL_PATH + 'ccsa_finetune/' + transfer_dir + '/' + str(ft_images_train) + 'images/' + args.run + '_run/'

if not os.path.isdir(tgt_path_classifier):
    makedirs(tgt_path_classifier)

# Create model objects
classifier_model = Encoder1(latent_dim=latent_dim).to(device)
optimizer = optim.Adam(classifier_model.parameters(), lr=finetune_lr)


def train_classifier(epoch):
    classifier_model.train()
    best_loss = np.Inf
    early_stop = False
    counter = 0
    last_desc = 'Train'
    len_source_loader = len(source_train_loader)
    len_target_loader = len(target_train_loader)
    iter_source = iter(source_train_loader)
    iter_target = iter(target_train_loader)
    num_iter = len_source_loader
    tbar = tqdm.trange(num_iter)

    for i, _ in enumerate(range(num_iter)):
        x_src, y_src = iter_source.next()
        x_tgt, y_tgt = iter_target.next()
        if i % len_target_loader == 0:
            iter_target = iter(target_train_loader)

        x_src = x_src.to(device)
        y_src = y_src.to(device)
        x_tgt = x_tgt.to(device)
        y_tgt = y_tgt.to(device)

        if x_src.shape[0] > x_tgt.shape[0]:
            x_src = x_src[:x_tgt.shape[0]]
            y_src = y_src[:x_tgt.shape[0]]

        if x_tgt.shape[0] > x_src.shape[0]:
            x_tgt = x_tgt[:x_src.shape[0]]
            y_tgt = y_tgt[:x_src.shape[0]]


        optimizer.zero_grad()
        z_src, y_hat_src = classifier_model(x_src)
        z_tgt, y_hat_tgt = classifier_model(x_tgt)

        ce_loss = classification_loss(y_hat_src, y_src)
        csa = csa_loss(z_src, z_tgt, (y_src == y_tgt).float().cuda())
        # loss = (1 - alpha) * ce_loss + alpha * csa
        loss = ce_loss + csa
        loss.backward()
        train_loss = loss.item()
        optimizer.step()
        tbar.set_description(last_desc)
        logger.log(mode="train", error=train_loss / len_target_loader, epoch=epoch, n_batch=0, num_batches=1,
               scalar='avg_loss')
        print('====> Train Epoch: {} Avg loss: {:.4f} '.format(epoch, train_loss / len_target_loader))

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
            return True

def fine_tune_ccsa_on_target():

    try:
        print("Loading Source Models")
        checkpoint_src = torch.load(src_path_classifier + src_classifier_name)
        classifier_model.load_state_dict(checkpoint_src)
        print("Saved Model successfully loaded")
    except:
        print("Model(s) not found.")
        exit()

    for epoch in range(1, args.epochs + 1):
        is_exit = train_classifier(epoch)
        if is_exit:
            break


def test_classifier_after_training(data_loader):
    """
    Test model accuracy after training
    :param data_loader: data loader to be tested
    :return: model accuracy
    """

    classifier_model.eval()
    test_loss = 0
    total = 0
    correct = 0
    with torch.no_grad():
        for i, (data, labels) in enumerate(tqdm.tqdm(data_loader, desc='')):
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

    # ************* TRAIN **********************
    # Method 1 :  Fine-tune the network on target data with CCSA loss
    if train_mode == 'train':
        fine_tune_ccsa_on_target()

    # ************** TEST **********************
    elif train_mode == 'test':
        checkpoint_vae_tgt = torch.load(tgt_path_classifier + tgt_classifier_name)
        classifier_model.load_state_dict(checkpoint_vae_tgt)
        print("After fine-tuning, Target")
        tgt_acc, tgt_loss = test_classifier_after_training(data_loader=target_test_loader)
    else:
        print("Sorry!! Invalid Mode..")


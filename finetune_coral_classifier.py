from __future__ import print_function
import argparse
import torch
import os
import torch.utils.data
import numpy as np
from common.utils.dataset import make_dataset
from torch.utils.data import DataLoader
from torch import nn, optim
from os import makedirs
from torchvision import transforms
from common.logging.tf_logger import Logger
import tqdm
from common.losses.custom_losses import coral_loss
from common.models.resnet_subset_models import EncoderLatent as Encoder1
import random

parser = argparse.ArgumentParser(description='FaceForensics++ CORAL FineTuning')

parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')

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

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
random.seed(int(args.run))
torch.manual_seed(int(args.run))
device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
patience = 30
ft_images_train = args.ft_images
alpha = 0.25

train_path = '/home/shivangi/Desktop/Projects/master_thesis/data/ff_face_20k/c23/train_20k_c23/'
val_path = '/home/shivangi/Desktop/Projects/master_thesis/data/ff_face_20k/c23/val_6k_c23/'

src_fake_classes = ['df', 'nt']
target_fake_classes = ['dfdc']
tsne_fake_classes = ['dfdc']


source_train_dataset = make_dataset(name='ff', base_path=train_path, num_classes=len(src_fake_classes)+1, fake_classes=src_fake_classes,
                             mode='face', image_count='all',
                             transform=transforms.Compose([transforms.ToPILImage(),
                                                           transforms.RandomHorizontalFlip(),
                                                           transforms.RandomVerticalFlip(),
                                                           transforms.ToTensor(),
                                                           transforms.Normalize([0.5] * 3, [0.5] * 3)
                                                           ]))

target_train_dataset = make_dataset(name='ff', base_path=train_path, num_classes=len(target_fake_classes)+1, fake_classes=target_fake_classes,
                                    mode='face_finetune', image_count=ft_images_train,
                                   transform=transforms.Compose([transforms.ToPILImage(),
                                                                 transforms.RandomHorizontalFlip(),
                                                                 transforms.RandomVerticalFlip(),
                                                                 transforms.ToTensor(),
                                                                 transforms.Normalize([0.5] * 3, [0.5] * 3)]))

# source_test_dataset = make_dataset(name='ff', base_path=val_path, num_classes=len(src_fake_classes)+1, fake_classes=src_fake_classes,
#                                    mode='face_finetune', image_count='all',
#                                    transform=transforms.Compose(
#                                        [transforms.ToPILImage(),
#                                         transforms.ToTensor(),
#                                         transforms.Normalize([0.5] * 3, [0.5] * 3)]))


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

batch_size_train = min(2*ft_images_train, 128)
batch_size_test = 128


source_train_loader = DataLoader(dataset=source_train_dataset, batch_size=batch_size_train, num_workers=8, shuffle=True)
target_train_loader = DataLoader(dataset=target_train_dataset, batch_size=batch_size_train, num_workers=8, shuffle=True)
target_test_loader = DataLoader(dataset=target_test_dataset, batch_size=batch_size_test, num_workers=8, shuffle=False)
tsne_loader = DataLoader(dataset=tsne_target_test_dataset, batch_size=batch_size_test, num_workers=8, shuffle=False)

logger = Logger(model_name='classifier_model', data_name='ff', log_path=os.path.join(os.getcwd(), 'tf_logs/coral_finetune/2classes_finetune/' + str(ft_images_train) + 'images/' + 'run_' + args.run + '/' + args.model_name))
src_classifier_name = 'vae_train_20k_val3k_mean1_std1_c23_latent16_3blocks_2classes_mixup_flip_normalize_df_nt.pt'
tgt_classifier_name = args.model_name + '.pt'

transfer_dir = 'df_nt_to_dessa'

# Paths
MODEL_PATH = os.path.join(os.getcwd(), 'models/')
src_path_classifier = MODEL_PATH + 'classifier/face/2classes/best/'
tgt_path_classifier = MODEL_PATH + 'coral_finetune/2classes_' + str(ft_images_train) + 'images/' + transfer_dir +'/' + args.run + '_run/'

if not os.path.isdir(tgt_path_classifier):
    makedirs(tgt_path_classifier)

# Losses
class_weights = torch.Tensor([1, 1]).cuda()
classification_loss = nn.CrossEntropyLoss(reduction='mean', weight=class_weights)

# Create model objects
classifier_model = Encoder1(latent_dim=16).to(device)

# # Freeze all layers of classifier model
# for param in classifier_model.parameters():
#     param.requires_grad = False

# # set the FC layer + last block gradients to true
# for param in classifier_model.fc1.parameters():
#     param.requires_grad = True
# for param in classifier_model.fc2.parameters():
#     param.requires_grad = True
# for param in classifier_model.avgpool.parameters():
#     param.requires_grad = True
# for param in classifier_model.layer3.parameters():
#     param.requires_grad = True
# print(sum(p.numel() for p in classifier_model.parameters() if p.requires_grad))


# Optmizers
optimizer = optim.Adam(classifier_model.parameters(), lr=1e-5)


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

        y_src[y_src == 2] = 1
        y_src[y_src == 3] = 1
        y_src[y_src == 4] = 1
        y_tgt[y_tgt == 2] = 1
        y_tgt[y_tgt == 3] = 1
        y_tgt[y_tgt == 4] = 1


        optimizer.zero_grad()
        z_src, y_hat_src = classifier_model(x_src)
        z_tgt, y_hat_tgt = classifier_model(x_tgt)
        c_loss = coral_loss(z_src, z_tgt)
        loss = classification_loss(y_hat_src, y_src) + classification_loss(y_hat_tgt, y_tgt) + alpha*c_loss
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
            print("EarlyStopping counter: " + str(counter) + " out of " + str(patience))
            if counter >= patience:
                early_stop = True
        # If early stopping flag is true, then stop the training
        if early_stop:
            print("Early stopping")
            return True

def fine_tune_coral_on_target():

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
        # if epoch % 10 == 0:
        #     visualize_latent_tsne(loader=tsne_loader, file_name=tsne_dir+"/abc_" + str(epoch), best_path=tgt_path_vae_best, model_name=vae_target, model=tgt_vae_model)


def test_classifier_after_training(data_loader):

    classifier_model.eval()
    test_loss = 0
    total = 0
    correct = 0
    with torch.no_grad():
        for i, (data, labels) in enumerate(tqdm.tqdm(data_loader, desc='')):
            data = data.to(device)
            labels = labels.to(device)
            labels[labels == 2] = 1
            labels[labels == 3] = 1
            labels[labels == 4] = 1
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
    fine_tune_coral_on_target()

    # ************** TARGET **********************
    checkpoint_vae_tgt = torch.load(tgt_path_classifier + tgt_classifier_name)
    classifier_model.load_state_dict(checkpoint_vae_tgt)

    print("After fine-tuning, Target")
    tgt_acc, tgt_loss = test_classifier_after_training(data_loader=target_test_loader)



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
from config import *
from common.models.resnet_subset_models import ForensicEncoder1 as Encoder
from common.models.resnet_subset_models import RelationNetwork

parser = argparse.ArgumentParser(description='Relation FaceForensics++ Few Shot')
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


source_train_loader = DataLoader(dataset=source_train_dataset, batch_size=batch_size, num_workers=8, shuffle=True)
target_train_loader = DataLoader(dataset=target_train_dataset, batch_size=batch_size, num_workers=8, shuffle=True)
target_val_loader = DataLoader(dataset=target_val_dataset, batch_size=batch_size, num_workers=8, shuffle=False)
target_test_loader = DataLoader(dataset=target_test_dataset, batch_size=batch_size, num_workers=8, shuffle=False)

logger = Logger(model_name='relation_net', data_name='ff', log_path=os.path.join(os.getcwd(), 'tf_logs/relation_net/2classes_finetune/' + str(ft_images_train) + '_images/' + 'run_' + args.run + '/' + args.model_name))
relation_net_source = 'train_20k_val3k_latent16_3blocks_2classes_mixup_flip_normalize_nt_df.pt'
relation_net_target = args.model_name + '.pt'
relation_net_file_name = args.model_name + '.npy'

transfer_dir = 'df_nt_to_dessa'
src_path_relation_net = MODEL_PATH + 'relation_net/face/2classes/best_mixup/'
tgt_path_relation_net_best = MODEL_PATH + 'relation_net_finetune/2classes_' + str(ft_images_train) + 'images/' + transfer_dir +'/' + args.run + '_run/'
if not os.path.isdir(tgt_path_relation_net_best):
    makedirs(tgt_path_relation_net_best)


FEATURE_DIM = 8
BATCH_NUM_PER_CLASS = batch_size//2
CLASS_NUM = 2
SAMPLE_NUM_PER_CLASS = 1

# Models and optimizers
tgt_encoder_model = Encoder(latent_dim=latent_dim).to(device)
tgt_relation_model = RelationNetwork(input_size=latent_dim*2, hidden_size=FEATURE_DIM).to(device)
optimizer_encoder_net = optim.Adam(tgt_encoder_model.parameters(), lr=finetune_lr)
optimizer_relation_net = optim.Adam(tgt_relation_model.parameters(), lr=finetune_lr)


def get_embeddings(custom_loader):
    z_real = torch.tensor([])
    z_fake = torch.tensor([])

    tgt_encoder_model.eval()
    tbar = tqdm(custom_loader)
    with torch.no_grad():
        for i, (x_t, y_t) in enumerate(tbar):
            x_t = x_t.to(device)
            y_t = y_t.to(device)
            z_t = tgt_encoder_model(x_t)

            # classes = torch.unique(y_t)
            idx_real = y_t.eq(0).nonzero().squeeze(1)
            idx_fake = y_t.eq(1).nonzero().squeeze(1)

            z_real = torch.cat([z_real, z_t[idx_real].cpu()], dim=0)
            z_fake = torch.cat([z_fake, z_t[idx_fake].cpu()], dim=0)

    return z_real.mean(0), z_fake.mean(0)


def train_relationnet(epoch, tgt_encoder_model, tgt_relation_model):
    iter_acc = []
    train_loss = 0
    tbar = tqdm(target_train_loader)
    last_desc = 'Train'
    # Get protoype embeddings of source data
    proto_real, proto_fake = get_embeddings(source_train_loader)
    z_prototype = torch.tensor([])
    z_prototype = torch.cat([z_prototype, proto_real.unsqueeze(0)], dim=0)
    z_prototype = torch.cat([z_prototype, proto_fake.unsqueeze(0)], dim=0)
    z_prototype = z_prototype.cuda()

    tgt_encoder_model.train()
    tgt_relation_model.train()
    for batch_idx, (x_tgt, y_tgt) in enumerate(tbar):

        x_tgt = x_tgt.to(device)
        y_tgt = y_tgt.to(device)
        z_tgt = tgt_encoder_model(x_tgt)

        iter_size = len(y_tgt)
        z_prototype_ext = z_prototype.unsqueeze(0).repeat(iter_size, 1, 1)
        z_val_ext = z_tgt.unsqueeze(0).repeat(CLASS_NUM, 1, 1)
        z_val_ext = torch.transpose(z_val_ext, 0, 1)

        relation_pairs = torch.cat((z_prototype_ext, z_val_ext), 2).view(-1, 16 * 2)
        relations = tgt_relation_model(relation_pairs).view(-1, CLASS_NUM)

        one_hot_labels = torch.zeros(iter_size, CLASS_NUM).scatter_(1, y_tgt.cpu().view(-1, 1),
                                                                                          1).cuda()
        loss = mse(relations, one_hot_labels)
        _, y_hat = relations.max(1)
        acc = y_hat.eq(y_tgt).float().mean()

        train_loss += loss.item()
        optimizer_encoder_net.zero_grad()
        optimizer_relation_net.zero_grad()
        loss.backward()
        optimizer_encoder_net.step()
        optimizer_relation_net.step()

        iter_acc.append(acc)

        if batch_idx % args.log_interval == 0:
            last_desc = 'Epoch: {} [({:.0f}%)] | Loss: {:.6f}'.format(epoch,
                                                                           100. * batch_idx / len(x_tgt),
                                                                           loss.item())
        tbar.set_description(last_desc)
    logger.log(mode="train", error=train_loss / len(target_train_loader), epoch=epoch, n_batch=0, num_batches=1,
               scalar='avg_loss')
    print('====> Train Epoch: {} Avg loss: {:.4f} '.format(epoch, train_loss / len(target_train_loader)))
    return train_loss / len(target_train_loader)


def fine_tune_relation_net_on_target(src_model_name, tgt_model_name):
    best_loss = np.Inf
    early_stop = False
    counter = 0
    try:
        print("Loading Source Models")
        checkpoint_src_enc = torch.load(src_path_relation_net + 'encoder_'+ src_model_name)
        checkpoint_src_rnet = torch.load(src_path_relation_net + 'relation_'+ src_model_name)
        # Copy weights from source encoder to target encoder
        tgt_encoder_model.load_state_dict(checkpoint_src_enc)
        tgt_relation_model.load_state_dict(checkpoint_src_rnet)
        print("Saved Model(s) successfully loaded")
    except:
        print("Model(s) not found.")
        exit()

    for epoch in range(1, args.epochs + 1):
        tgt_encoder_model.train()
        tgt_relation_model.train()
        train_loss = train_relationnet(epoch, tgt_encoder_model, tgt_relation_model)
        if train_loss <= best_loss:
            counter = 0
            best_loss = train_loss
            torch.save(tgt_encoder_model.state_dict(), tgt_path_relation_net_best + 'encoder_' + tgt_model_name)
            torch.save(tgt_relation_model.state_dict(), tgt_path_relation_net_best + 'relation_' + tgt_model_name)
            print("Best encoder_model saved/updated..")
        else:
            counter += 1
            print("EarlyStopping counter: " + str(counter) + " out of " + str(finetune_patience))
            if counter >= finetune_patience:
                early_stop = True
        # If early stopping flag is true, then stop the training
        if early_stop:
            print("Early stopping")
            break


def save_prototype_embeddings():

    tgt_encoder_model.eval()
    prototype = torch.tensor([])

    train_local_dataset = make_dataset(name='ff', base_path=src_train_path, num_classes=len(src_fake_classes)+1,
                                 mode='face', image_count='all',
                                 transform=transforms.Compose([transforms.ToPILImage(),
                                                               transforms.ToTensor(),
                                                               transforms.Normalize([0.5] * 3, [0.5] * 3),
                                                               ]))

    train_local_loader = DataLoader(dataset=train_local_dataset, batch_size=128, num_workers=8, shuffle=False)

    with torch.no_grad():
        train_real, train_fake = get_embeddings(train_local_loader)
        val_real, val_fake = get_embeddings(target_train_loader)
        prototype = torch.cat([prototype, ((train_real + val_real)/2).unsqueeze(0)], dim=0)
        prototype = torch.cat([prototype, ((train_fake + val_fake)/2).unsqueeze(0)], dim=0)
        torch.save(prototype, tgt_path_relation_net_best+relation_net_file_name)


def test_protonet_after_finetuning(data_loader):

    tgt_encoder_model.eval()
    tgt_relation_model.eval()
    correct = 0
    total = 0
    z_prototype = torch.load(tgt_path_relation_net_best+relation_net_file_name).cuda()
    with torch.no_grad():
        for i, (x_test, y_test) in enumerate(tqdm(data_loader, desc='')):
            x_test = x_test.to(device)
            y_test = y_test.to(device)
            y_test[y_test == 2] = 1
            y_test[y_test == 3] = 1
            y_test[y_test == 4] = 1
            z_test = tgt_encoder_model(x_test)

            iter_size = len(y_test)
            z_prototype_ext = z_prototype.unsqueeze(0).repeat(iter_size, 1, 1)
            z_val_ext = z_test.unsqueeze(0).repeat(CLASS_NUM, 1, 1)
            z_val_ext = torch.transpose(z_val_ext, 0, 1)

            relation_pairs = torch.cat((z_prototype_ext, z_val_ext), 2).view(-1, 16 * 2)
            relations = tgt_relation_model(relation_pairs).view(-1, CLASS_NUM)

            # Calculate correct predictions
            total += y_test.size(0)
            _, y_hat = relations.max(1)
            correct += (y_hat == y_test).sum().item()
        accuracy = 100 * correct / total
    print('====> Test Acc: {:.4f} '.format(accuracy))
    f = open("results.txt", "a+")
    f.write("%.3f" % (accuracy))
    f.write("\n")
    f.close()

if __name__ == "__main__":

    # TRAIN
    # Method 1 : Directly fine-tune the network. Update all the layers
    if train_mode == 'train':
        fine_tune_relation_net_on_target(src_model_name=relation_net_source, tgt_model_name=relation_net_target)
        checkpoint_encoder_tgt = torch.load(tgt_path_relation_net_best + 'encoder_' + relation_net_target)
        checkpoint_relation_tgt = torch.load(tgt_path_relation_net_best + 'relation_' + relation_net_target)
        tgt_encoder_model.load_state_dict(checkpoint_encoder_tgt)
        tgt_relation_model.load_state_dict(checkpoint_relation_tgt)
        save_prototype_embeddings()
        test_protonet_after_finetuning(data_loader=target_val_loader)

    elif train_mode == 'test':
        # VALIDATION
        # Evaluate after training
        # Load the target encoder_model
        # ************** TARGET **********************
        checkpoint_encoder_tgt = torch.load(tgt_path_relation_net_best + 'encoder_' + relation_net_target)
        checkpoint_relation_tgt = torch.load(tgt_path_relation_net_best + 'relation_' + relation_net_target)
        tgt_encoder_model.load_state_dict(checkpoint_encoder_tgt)
        tgt_relation_model.load_state_dict(checkpoint_relation_tgt)
        test_protonet_after_finetuning(data_loader=target_val_loader)
    else:
        print("Sorry!! Invalid Mode..")

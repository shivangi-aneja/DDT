from __future__ import print_function
import argparse
import torch.utils.data
import numpy as np
from torch.utils.data import DataLoader
from torch import optim
from os import makedirs
from common.logging.tf_logger import Logger
from tqdm import tqdm
import random
from config import *
import torch.nn.functional as F
from common.losses.custom_losses import prototypical_loss_full
from common.utils.common_utils import euclidean_dist
from common.models.resnet_subset_models import ForensicEncoder1 as Encoder

parser = argparse.ArgumentParser(description='ProtoNet FaceForensics++ Few Shot')
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

logger = Logger(model_name='protonet', data_name='ff', log_path=os.path.join(os.getcwd(), 'tf_logs/protonet/2classes_finetune/' + str(ft_images_train) + '_images/' + 'run_' + args.run + '/' + args.model_name))
protonet_source = 'protonet_train_20k_val3k_latent16_3blocks_2classes_mixup_flip_normalize_nt_df.pt'
protonet_target = args.model_name + '.pt'
proto_file_name = args.model_name + '.npy'

transfer_dir = 'df_nt_to_dessa'
src_path_protonet = MODEL_PATH + 'protonet/face/2classes/best_mixup/'
tgt_path_protonet_best = MODEL_PATH + 'protonet_finetune/2classes_' + str(ft_images_train) + 'images/' + transfer_dir +'/' + args.run + '_run/'
if not os.path.isdir(tgt_path_protonet_best):
    makedirs(tgt_path_protonet_best)


tgt_protonet_model = Encoder(latent_dim=latent_dim).to(device)
optimizer = optim.Adam(tgt_protonet_model.parameters(), lr=finetune_lr)


def get_embeddings(custom_loader):

    z_real = torch.tensor([])
    z_fake = torch.tensor([])

    tgt_protonet_model.eval()
    tbar = tqdm(custom_loader)
    with torch.no_grad():
        for i, (x_t, y_t) in enumerate(tbar):
            x_t = x_t.to(device)
            y_t = y_t.to(device)

            z_t = tgt_protonet_model(x_t)

            # classes = torch.unique(y_t)
            idx_real = y_t.eq(0).nonzero().squeeze(1)
            idx_fake = y_t.eq(1).nonzero().squeeze(1)

            z_real = torch.cat([z_real, z_t[idx_real].cpu()], dim=0)
            z_fake = torch.cat([z_fake, z_t[idx_fake].cpu()], dim=0)

    return z_real.mean(0), z_fake.mean(0)


def train_protonet(epoch, tgt_protonet_model):
    iter_acc = []
    train_loss = 0
    tbar = tqdm(target_train_loader)
    last_desc = 'Train'
    # Get protoype embeddings of source data
    proto_real, proto_fake = get_embeddings(source_train_loader)
    prototype = torch.tensor([])
    prototype = torch.cat([prototype, proto_real.unsqueeze(0)], dim=0)
    prototype = torch.cat([prototype, proto_fake.unsqueeze(0)], dim=0)

    tgt_protonet_model.train()
    for batch_idx, (x_tgt, y_tgt) in enumerate(tbar):

        x_tgt = x_tgt.to(device)
        y_tgt = y_tgt.to(device)

        z_tgt = tgt_protonet_model(x_tgt)
        loss, acc = prototypical_loss_full(prototype.cuda(), z_tgt, y_tgt)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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


def fine_tune_protonet_on_target(src_model_name, tgt_model_name):
    best_loss = np.Inf
    early_stop = False
    counter = 0
    try:
        print("Loading Source Models")
        checkpoint_src = torch.load(src_path_protonet + src_model_name)
        # Copy weights from source encoder to target encoder
        tgt_protonet_model.load_state_dict(checkpoint_src)
        print("Saved Model(s) successfully loaded")
    except:
        print("Model(s) not found.")
        exit()

    for epoch in range(1, args.epochs + 1):
        tgt_protonet_model.train()
        train_loss = train_protonet(epoch, tgt_protonet_model)
        # scheduler_encoder_net.step(train_loss)
        if train_loss <= best_loss:
            counter = 0
            best_loss = train_loss
            torch.save(tgt_protonet_model.state_dict(), tgt_path_protonet_best + tgt_model_name)
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

    tgt_protonet_model.eval()
    prototype = torch.tensor([])

    train_local_dataset = make_dataset(name='ff', base_path=src_train_path, num_classes=len(src_fake_classes)+1,
                                 mode='face', image_count='all',
                                 transform=transforms.Compose([transforms.ToPILImage(),
                                                               transforms.ToTensor(),
                                                               transforms.Normalize([0.5] * 3, [0.5] * 3),
                                                               ]))

    train_local_loader = DataLoader(dataset=train_local_dataset, batch_size=batch_size, num_workers=8, shuffle=False)

    with torch.no_grad():
        train_real, train_fake = get_embeddings(train_local_loader)
        val_real, val_fake = get_embeddings(target_train_loader)
        prototype = torch.cat([prototype, ((train_real + val_real)/2).unsqueeze(0)], dim=0)
        prototype = torch.cat([prototype, ((train_fake + val_fake)/2).unsqueeze(0)], dim=0)
        torch.save(prototype, tgt_path_protonet_best+proto_file_name)


def test_protonet_after_finetuning(data_loader):

    tgt_protonet_model.eval()
    correct = 0
    total = 0
    prototypes = torch.load(tgt_path_protonet_best+proto_file_name).cuda()
    with torch.no_grad():
        for i, (x_test, y_test) in enumerate(tqdm(data_loader, desc='')):
            x_test = x_test.to(device)
            y_test = y_test.to(device)
            z_test = tgt_protonet_model(x_test)

            dists = euclidean_dist(z_test, prototypes)
            log_p_y = F.log_softmax(-dists, dim=1)

            # Calculate correct predictions
            total += y_test.size(0)
            _, y_hat = log_p_y.max(1)
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
        fine_tune_protonet_on_target(src_model_name=protonet_source, tgt_model_name=protonet_target)
        checkpoint_protonet_tgt = torch.load(tgt_path_protonet_best + protonet_target)
        tgt_protonet_model.load_state_dict(checkpoint_protonet_tgt)
        save_prototype_embeddings()
        test_protonet_after_finetuning(data_loader=target_val_loader)

    elif train_mode == 'test':
        # VALIDATION
        # Evaluate after training
        # Load the target encoder_model
        # ************** TARGET **********************
        checkpoint_protonet_tgt = torch.load(tgt_path_protonet_best + protonet_target)
        tgt_protonet_model.load_state_dict(checkpoint_protonet_tgt)
        test_protonet_after_finetuning(data_loader=target_test_loader)
    else:
        print("Sorry!! Invalid Mode..")

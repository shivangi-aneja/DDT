from __future__ import print_function
import argparse
import torch.utils.data
import numpy as np
from torch.utils.data import DataLoader
from torch import optim
from os import makedirs
from common.logging.tf_logger import Logger
from common.utils.data_samplers import BalancedBatchSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from  tqdm import trange
from config import *
import torch.nn.functional as F
from common.losses.custom_losses import prototypical_loss_full
from common.utils.common_utils import euclidean_dist
from common.models.resnet_subset_models import ForensicEncoder1 as Encoder

parser = argparse.ArgumentParser(description='Prototypical Network FaceForensics++ ')
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


# Both train and val loader are shuffled by default
train_batch_sampler = BalancedBatchSampler(train_dataset.labels, n_classes=num_classes, n_samples=batch_size//2)
val_batch_sampler = BalancedBatchSampler(val_dataset.labels, n_classes=num_classes, n_samples=batch_size//2)
train_loader_balanced = DataLoader(dataset=train_dataset, batch_sampler=train_batch_sampler)
val_loader_balanced = DataLoader(dataset=test_dataset, batch_sampler=val_batch_sampler)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=8, shuffle=False)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=8, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=8, shuffle=False)


# Models
model_name = 'protonet_train_20k_val3k_latent' + str(latent_dim) + '_3blocks_2classes_mixup_flip_normalize_nt'
proto_file_name = 'protonet_train_20k_val3k_latent' + str(latent_dim) + '_3blocks_2classes_mixup_flip_normalize_nt_df.npy'
logger = Logger(model_name='protonet', data_name='ff', log_path=os.path.join(os.getcwd(), 'tf_logs/protonet/2classes/'+model_name))
model_name = model_name + '.pt'
best_path = MODEL_PATH + 'protonet/face/2classes/best_mixup/'
if not os.path.isdir(best_path):
    makedirs(best_path)

# Models and optimizers
model = Encoder(latent_dim=latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=train_lr)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience, verbose=True)


def train_protonet(epoch):

    train_loss = 0
    last_desc = 'Train'
    len_val_loader = len(val_loader_balanced)
    iter_val = iter(val_loader_balanced)
    num_iter = len_val_loader
    tbar = trange(num_iter)

    prototype = torch.tensor([])
    train_real, train_fake = get_embeddings(train_loader)
    prototype = torch.cat([prototype, train_real.unsqueeze(0)], dim=0)
    prototype = torch.cat([prototype, train_fake.unsqueeze(0)], dim=0)

    iter_acc = []

    model.train()

    for i, _ in enumerate(tbar):
        x_val, y_val = iter_val.next()
        x_val = x_val.to(device)
        y_val = y_val.to(device)

        z_val = model(x_val)

        loss, acc = prototypical_loss_full(prototype.cuda(), z_val, y_val)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iter_acc.append(acc)

        if i % args.log_interval == 0:
            last_desc = 'Epoch: {} [({:.0f}%)] | Loss: {:.6f}'.format(epoch,
                                                                      100. * i / len_val_loader,
                                                                      loss.item())

        tbar.set_description(last_desc)

    accuracy = torch.mean(torch.stack(iter_acc))
    logger.log(mode="train", error=train_loss/len_val_loader, epoch=epoch, n_batch=0, num_batches=1,
           scalar='avg_loss')
    print('====> Train Epoch: {} Avg loss: {:.4f}  Val Acc: {:.4f}'.format(epoch, train_loss, accuracy))
    return train_loss/len_val_loader


def test_protonet_after_training():
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
    correct = 0
    total = 0
    prototypes = torch.load(best_path + proto_file_name).cuda()
    with torch.no_grad():
        for i, (x_test, y_test) in enumerate(tqdm(test_loader, desc='')):
            x_test = x_test.to(device)
            y_test = y_test.to(device)
            z_test = model(x_test)

            dists = euclidean_dist(z_test, prototypes)
            log_p_y = F.log_softmax(-dists, dim=1)

            # Calculate correct predictions
            total += y_test.size(0)
            _, y_hat = log_p_y.max(1)
            correct += (y_hat == y_test).sum().item()
        accuracy = 100 * correct / total
    print('====> Test Acc: {:.4f} '.format(accuracy))


def save_prototype_embeddings():
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
    prototype = torch.tensor([])

    train_local_dataset = make_dataset(name='ff', base_path=train_path, num_classes=num_classes,
                                 mode='face', image_count='all',
                                 transform=transforms.Compose([transforms.ToPILImage(),
                                                               transforms.ToTensor(),
                                                               transforms.Normalize([0.5] * 3, [0.5] * 3),
                                                               ]))

    train_local_loader = DataLoader(dataset=train_local_dataset, batch_size=batch_size, num_workers=8, shuffle=False)

    with torch.no_grad():
        train_real, train_fake = get_embeddings(train_local_loader)
        prototype = torch.cat([prototype, train_real.unsqueeze(0)], dim=0)
        prototype = torch.cat([prototype, train_fake.unsqueeze(0)], dim=0)
        torch.save(prototype, best_path+proto_file_name)


def get_embeddings(custom_loader):

    z_real = torch.tensor([])
    z_fake = torch.tensor([])

    model.eval()
    with torch.no_grad():
        for i, (x_t, y_t) in enumerate(tqdm(custom_loader, desc='')):
            x_t = x_t.to(device)
            y_t = y_t.to(device)
            z_t = model(x_t)

            idx_real = y_t.eq(0).nonzero().squeeze()
            idx_fake = y_t.eq(1).nonzero().squeeze()

            z_real = torch.cat([z_real, z_t[idx_real].cpu()], dim=0)
            z_fake = torch.cat([z_fake, z_t[idx_fake].cpu()], dim=0)

    return z_real.mean(0), z_fake.mean(0)


def train_model_protonet():
    best_loss = np.Inf
    patience = 5
    early_stop = False
    counter = 0
    for epoch in range(1, args.epochs + 1):

        epoch_loss = train_protonet(epoch)
        if epoch_loss <= best_loss:
            counter = 0
            best_loss = epoch_loss
            torch.save(model.state_dict(), best_path + model_name)
            print("Best encoder_model saved/updated..")
        else:
            counter += 1
            print("EarlyStopping counter: " + str(counter) + " out of " + str(patience))
            if counter >= patience:
                early_stop = True
        # If early stopping flag is true, then stop the training
        if early_stop:
            print("Early stopping")
            return True


if __name__ == "__main__":
    if train_mode == 'train':
        train_model_protonet()
        save_prototype_embeddings()
    elif train_mode == 'test':
        test_protonet_after_training()

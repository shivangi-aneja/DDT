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
from tqdm import tqdm, trange
from config import *
from common.models.resnet_subset_models import ForensicEncoder1 as Encoder
from common.models.resnet_subset_models import RelationNetwork

parser = argparse.ArgumentParser(description='Relation Network FaceForensics++ ')
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


# Data Path and Loaders
train_batch_sampler = BalancedBatchSampler(train_dataset.labels, n_classes=num_classes, n_samples=batch_size//2)
val_batch_sampler = BalancedBatchSampler(val_dataset.labels, n_classes=num_classes, n_samples=batch_size//2)
train_loader_balanced = DataLoader(dataset=train_dataset, batch_sampler=train_batch_sampler)
val_loader_balanced = DataLoader(dataset=val_dataset, batch_sampler=val_batch_sampler)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=8, shuffle=False)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=8, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=8, shuffle=False)

# Paths
best_path = MODEL_PATH + 'relation_net/face/2classes/best_mixup/'
if not os.path.isdir(best_path):
    makedirs(best_path)


model_name = 'train_20k_val3k_latent' + str(latent_dim) + '_3blocks_2classes_mixup_flip_normalize_nt'
relation_file_name = 'relation_net_train_20k_val3k_latent' + str(latent_dim) + '_3blocks_2classes_mixup_flip_normalize_nt_df.npy'
logger = Logger(model_name='relation_net', data_name='ff', log_path=os.path.join(os.getcwd(), 'tf_logs/relation_net/2classes/'+model_name))
model_name = model_name + '.pt'


# Models and optimizers
FEATURE_DIM = 8
BATCH_NUM_PER_CLASS = batch_size//2
CLASS_NUM = 2
SAMPLE_NUM_PER_CLASS = 1
encoder_model = Encoder(latent_dim=latent_dim).to(device)
relation_model = RelationNetwork(input_size=latent_dim*2, hidden_size=FEATURE_DIM).to(device)

# Model optimizer_encoder_net
optimizer_encoder_net = optim.Adam(encoder_model.parameters(), lr=train_lr)
scheduler_encoder_net = ReduceLROnPlateau(optimizer_encoder_net, mode='min', factor=scheduler_factor, patience=scheduler_patience, verbose=True)
optimizer_relation_net = optim.Adam(relation_model.parameters(), lr=train_lr)
scheduler_relation_net = ReduceLROnPlateau(optimizer_relation_net, mode='min', factor=scheduler_factor, patience=scheduler_patience, verbose=True)


def train_relation_net(epoch):

    train_loss = 0
    last_desc = 'Train'
    len_val_loader = len(val_loader_balanced)
    iter_val = iter(val_loader_balanced)
    num_iter = len_val_loader
    tbar = trange(num_iter)
    z_prototype = torch.tensor([])
    train_real, train_fake = get_embeddings(train_loader)
    z_prototype = torch.cat([z_prototype, train_real.unsqueeze(0)], dim=0)
    z_prototype = torch.cat([z_prototype, train_fake.unsqueeze(0)], dim=0)
    z_prototype = z_prototype.cuda()    # 2 X 16

    iter_acc = []

    encoder_model.train()
    relation_model.train()

    for i, _ in enumerate(tbar):
        x_val, y_val = iter_val.next()

        x_val = x_val.to(device)
        y_val = y_val.to(device)

        z_val = encoder_model(x_val)    # 128 X 16 dim tensor

        # Calculate relations with relation module
        # each prototype link to every validation sample to calculate relations
        # to form a (num_classes*samples_per_class)x16 matrix for relation network
        z_prototype_ext = z_prototype.unsqueeze(0).repeat(CLASS_NUM * BATCH_NUM_PER_CLASS, 1, 1)


        z_val_ext = z_val.unsqueeze(0).repeat(CLASS_NUM, 1, 1)
        z_val_ext = torch.transpose(z_val_ext, 0, 1)

        relation_pairs = torch.cat((z_prototype_ext, z_val_ext), 2).view(-1, 16*2)
        relations = relation_model(relation_pairs).view(-1, CLASS_NUM)

        one_hot_labels = torch.zeros(BATCH_NUM_PER_CLASS * CLASS_NUM, CLASS_NUM).scatter_(1, y_val.cpu().view(-1, 1), 1).cuda()
        loss = mse(relations, one_hot_labels)

        _, y_hat = relations.max(1)
        acc = y_hat.eq(y_val).float().mean()

        train_loss += loss.item()
        optimizer_encoder_net.zero_grad()
        optimizer_relation_net.zero_grad()
        loss.backward()
        optimizer_encoder_net.step()
        optimizer_relation_net.step()

        iter_acc.append(acc)

        if i % args.log_interval == 0:
            last_desc = 'Epoch: {} [({:.0f}%)] | Loss: {:.6f} | Acc: {:.6f}'.format(epoch,
                                                                      100. * i / len_val_loader,
                                                                      loss.item(), acc)

        tbar.set_description(last_desc)

    accuracy = torch.mean(torch.stack(iter_acc))
    logger.log(mode="train", error=train_loss/len_val_loader, epoch=epoch, n_batch=0, num_batches=1,
           scalar='avg_loss')
    print('====> Train Epoch: {} Avg loss: {:.4f}  Val Acc: {:.4f}'.format(epoch, train_loss, accuracy))
    return train_loss/len_val_loader


def test_relationnet_after_training():
    try:
        print("Loading Saved Model")
        print(best_path)
        checkpoint_e = torch.load(best_path + 'encoder_' + model_name)
        checkpoint_r = torch.load(best_path + 'relation_' + model_name)
        encoder_model.load_state_dict(checkpoint_e)
        relation_model.load_state_dict(checkpoint_r)
        print("Saved Model successfully loaded")
    except:
        print("Model not found.")
        exit()
    encoder_model.eval()
    correct = 0
    total = 0
    z_prototype = torch.load(best_path + relation_file_name).cuda()
    with torch.no_grad():
        for i, (x_test, y_test) in enumerate(tqdm(test_loader, desc='')):
            x_test = x_test.to(device)
            y_test = y_test.to(device)
            y_test[y_test == 2] = 1
            y_test[y_test == 3] = 1
            y_test[y_test == 4] = 1
            z_test = encoder_model(x_test)

            iter_size = len(y_test)
            z_prototype_ext = z_prototype.unsqueeze(0).repeat(iter_size, 1, 1)
            z_val_ext = z_test.unsqueeze(0).repeat(CLASS_NUM, 1, 1)
            z_val_ext = torch.transpose(z_val_ext, 0, 1)

            relation_pairs = torch.cat((z_prototype_ext, z_val_ext), 2).view(-1, 16 * 2)
            relations = relation_model(relation_pairs).view(-1, CLASS_NUM)

            # Calculate correct predictions
            total += y_test.size(0)
            _, y_hat = relations.max(1)
            correct += (y_hat == y_test).sum().item()
        accuracy = 100 * correct / total
    print('====> Test Acc: {:.4f} '.format(accuracy))


def save_prototype_embeddings():
    try:
        print("Loading Saved Model")
        print(best_path)
        checkpoint = torch.load(best_path + 'encoder_' + model_name)
        encoder_model.load_state_dict(checkpoint)
        print("Saved Model successfully loaded")
    except:
        print("Model not found.")
        exit()

    encoder_model.eval()
    prototype = torch.tensor([])

    train_local_dataset = make_dataset(name='ff', base_path=train_path, num_classes=num_classes,
                                 mode='face', image_count='all',
                                 transform=transforms.Compose([transforms.ToPILImage(),
                                                               transforms.ToTensor(),
                                                               transforms.Normalize([0.5] * 3, [0.5] * 3),
                                                               ]))

    train_local_loader = DataLoader(dataset=train_local_dataset, batch_size=128, num_workers=8, shuffle=False)

    with torch.no_grad():
        train_real, train_fake = get_embeddings(train_local_loader)
        prototype = torch.cat([prototype, train_real.unsqueeze(0)], dim=0)
        prototype = torch.cat([prototype, train_fake.unsqueeze(0)], dim=0)
        torch.save(prototype, best_path+relation_file_name)


def get_embeddings(custom_loader):

    z_real = torch.tensor([])
    z_fake = torch.tensor([])

    encoder_model.eval()
    with torch.no_grad():
        for i, (x_t, y_t) in enumerate(tqdm(custom_loader, desc='')):

            x_t = x_t.to(device)
            y_t = y_t.to(device)

            z_t = encoder_model(x_t)

            # classes = torch.unique(y_t)
            idx_real = y_t.eq(0).nonzero().squeeze()
            idx_fake = y_t.eq(1).nonzero().squeeze()

            z_real = torch.cat([z_real, z_t[idx_real].cpu()], dim=0)
            z_fake = torch.cat([z_fake, z_t[idx_fake].cpu()], dim=0)

    return z_real.mean(0), z_fake.mean(0)


def train_model_relation_net():
    best_loss = np.Inf
    early_stop = False
    counter = 0
    for epoch in range(1, args.epochs + 1):
        epoch_loss = train_relation_net(epoch)
        if epoch_loss <= best_loss:
            counter = 0
            best_loss = epoch_loss
            torch.save(encoder_model.state_dict(), best_path + 'encoder_' + model_name)
            torch.save(relation_model.state_dict(), best_path + 'relation_' + model_name)
            print("Best encoder_model and relation_model saved/updated..")
        else:
            counter += 1
            print("EarlyStopping counter: " + str(counter) + " out of " + str(train_patience))
            if counter >= train_patience:
                early_stop = True
        # If early stopping flag is true, then stop the training
        if early_stop:
            print("Early stopping")
            return True


if __name__ == "__main__":
    if train_mode == 'train':
        # train_model_relation_net()
        save_prototype_embeddings()
    elif train_mode == 'test':
        test_relationnet_after_training()

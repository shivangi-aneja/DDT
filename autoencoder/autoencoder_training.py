import os
import time
import torch
from os import makedirs
from torch.autograd import Variable
from common.logging.logger import rootLogger
from common.logging.tf_logger import Logger
from common.utils.image_utils import save_image
from sklearn.manifold import TSNE
from sklearn import manifold
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

width = 4000
height = 3000
max_dim = 100

category_to_label = {0: 'Orig', 1: 'F2F', 2: 'DF', 3: 'FS'}
category_to_color = np.array(['#ff80ed', '#065535', '#133337','#ffc0cb.'])
colormap = np.array(["#6d8dca", "#69de53", "#723bca", "#c3e14c", "#c84dc9", "#68af4e", "#6e6cd5", "#e3be38",
                     "#4e2d7c", "#5fdfa8", "#d34690", "#3f6d31", "#d44427", "#7fcdd8", "#cb4053", "#5e9981",
                     "#803a62", "#9b9e39", "#c88cca", "#e1c37b", "#34223b", "#bdd8a3", "#6e3326", "#cfbdce",
                     "#d07d3c", "#52697d", "#194196", "#d27c88", "#36422b", "#b68f79", "#00ffff", "#33ff33",
                     "#ffff99", "#99ff33", "#ff6666", "#666600", "#99004c", "#808080", "#a80a0a", "#a4924c",
                     "#4a8e92", "#92734a", "#7d4097", "#4b4097", "#c0c0c0", "#409794", "#1a709b", "#a7dcf6",
                     "#b1a7f6", "#eea7f6"])

class Autoencoder(object):

    def __init__(self, autoencoder, model_name, recon_loss, optim_kwargs, dataset, model_dataset=None, batch_size=64,
                 optim=None, epochs=10, tf_log_path=None, use_cuda=None, autoencoder_params=None):
        """
        :param autoencoder: Autoencoder Network
        :param model_name: Model Name
        :param recon_loss: Reconstruction Loss
        :param dataset: Dataset
        :param batch_size: Batch Size
        :param optim: Optimizer
        :param lr: Learning Rate
        :param epochs: Number of epochs
        :param tf_log_path: Tensorflow Log Path
        """
        self.autoencoder = autoencoder
        if autoencoder_params is None or not len(autoencoder_params):
            autoencoder_params = filter(lambda x: x.requires_grad, self.autoencoder.parameters())
            params = sum([np.prod(p.size()) for p in self.autoencoder.parameters()])
            rootLogger.info("Trainable Parameters : " + str(params))
        optim = optim or torch.optim.Adam
        optim_kwargs = optim_kwargs or {}
        optim_kwargs.setdefault('lr', 1e-3)
        self.optim = optim(autoencoder_params, **optim_kwargs)
        self.epochs = epochs
        self.recon_loss = recon_loss
        self.model_name = model_name
        self.dataset = dataset
        self.model_dataset = model_dataset
        self.batch_size = batch_size
        self.use_cuda = use_cuda
        if self.use_cuda is None:
            self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.autoencoder.cuda()
        if tf_log_path is not None:
            self.logger = Logger(model_name=self.model_name, data_name=self.dataset, log_path=tf_log_path)

    def train_autoencoder(self, train_loader, test_loader, model_path):
        """
        Function for training
        :param train_loader: Loader for training data
        :param test_loader: Loader for test data
        :param model_path: Path for saving the data
        """

        # results save folder
        model_current = model_path + '/autoencoders/' + self.dataset + '/current/' + self.model_name + '.pt'
        model_best = model_path + '/autoencoders/' + self.dataset + '/best/' + self.model_name + '.pt'
        try:
            rootLogger.info("Loading Saved Model")
            checkpoint = torch.load(model_current)
            self.autoencoder.load_state_dict(checkpoint)
            rootLogger.info("Saved Model successfully loaded")
        except:
            rootLogger.info("Model not found, Created a new one")
            # Make directory for Saving Models
            if not os.path.isdir(model_path + 'autoencoders/' + self.dataset + '/current/'):
                makedirs(model_path + 'autoencoders/' + self.dataset + '/current/')
            if not os.path.isdir(model_path + 'autoencoders/' + self.dataset + '/best/'):
                makedirs(model_path + 'autoencoders/' + self.dataset + '/best/')

        # training-loop
        np.random.seed(int(time.time()))
        rootLogger.info('Training Start!!!')
        best_loss = np.Inf

        for epoch in range(self.epochs):
            self.autoencoder.train()
            epoch_start_time = time.time()
            epoch_train_loss = 0.

            # Checkpoint after 5 epochs
            if epoch % 10 == 0:
                try:
                    rootLogger.info("Saving the model")
                    torch.save(self.autoencoder.state_dict(), model_current)
                    rootLogger.info("Model Saved")
                except:
                    rootLogger.info("Can't save the model")

            for epoch_iter, data in enumerate(train_loader):

                input_image, target = data
                # Move the images to the device first before computation
                if self.use_cuda:
                    input_image = input_image.cuda()
                    target = target.cuda()
                input_image = Variable(input_image)
                target = Variable(target)

                self.optim.zero_grad()  # clear gradients for this training step

                _, pred_image = self.autoencoder(input_image, target)
                loss = self.recon_loss(pred_image, input_image)
                epoch_train_loss += loss.item()

                loss.backward()  # backpropagation, compute gradients
                self.optim.step()  # apply gradients
            avg_loss = epoch_train_loss / len(train_loader)

            # Save the best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(self.autoencoder.state_dict(), model_best)
                rootLogger.info("Model Saved")

            # Log the training losses
            self.logger.log(mode="train", error=avg_loss, epoch=epoch + 1, n_batch=0, num_batches=1)

            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - epoch_start_time
            rootLogger.info(
                '[%d/%d] - ptime: %.2f Train loss: %.3f' % ((epoch + 1), self.epochs, per_epoch_ptime, avg_loss))

            # Validation
            self.evaluate_val_data_tf(test_loader=test_loader, model_name=model_current, epoch=epoch)

    def evaluate_val_data_tf(self, test_loader, model_name, epoch):
        """
        Function to evaluate the results on trained model
        :param test_loader: data loader on which clustering is evaluated
        :param model_name: name with which pre-trained model is saved
        :param epoch:
        :return: None
        """
        self.autoencoder.eval()
        # Load the parameters of pretrained model
        # checkpoint = torch.load(model_name)
        # self.autoencoder.load_state_dict(checkpoint)
        # Evaluate the results on
        epoch_train_loss = 0
        with torch.no_grad():
            for epoch_iter, data in enumerate(test_loader):
                input_image, target = data
                if self.use_cuda:
                    input_image = input_image.cuda()
                    target = target.cuda()

                _, pred_image = self.autoencoder(input_image,target)

                loss = self.recon_loss(pred_image, input_image)
                epoch_train_loss += loss.item()

                if epoch_iter == len(test_loader) - 1:
                    self.logger.log_images(mode='predicted', images=pred_image, num_images=len(pred_image),
                                           epoch=epoch, n_batch=1, num_batches=len(test_loader), normalize=True)

                    self.logger.log_images(mode='ground_truth', images=input_image, num_images=len(input_image),
                                           epoch=epoch, n_batch=1, num_batches=len(test_loader), normalize=True)

            avg_loss = epoch_train_loss / len(test_loader)

            rootLogger.info("Val loss= [%.3f]" % avg_loss)
            self.logger.log(mode="val", error=avg_loss, epoch=epoch, n_batch=0, num_batches=1)

    def validate_results(self, val_loader, model_path, image_path_pred, image_path_gt):
        """
        Function to evaluate the result on test data
        :param val_loader:
        :param model_path:
        :param image_path_pred:
        :param image_path_gt:
        :return:
        """

        try:
            rootLogger.info("Loading Saved Model")
            checkpoint = torch.load(model_path + '/autoencoders/' + self.model_dataset + '/best/' + self.model_name + '.pt')
            self.autoencoder.load_state_dict(checkpoint)
            rootLogger.info("Saved Model successfully loaded")
        except:
            rootLogger.info("Model not found.")

        with torch.no_grad():
            for epoch_iter, data in enumerate(val_loader):
                input_image, target = data
                if self.use_cuda:
                    input_image = input_image.cuda()
                    target = target.cuda()
                _, pred_image = self.autoencoder(input_image, target)
                ctr = epoch_iter * self.batch_size
                save_image(pred_image, image_path=image_path_pred, file_num=ctr, mode='pred')
                #save_image(input_image, image_path=image_path_gt, file_num=ctr, mode='gt')

    def visualize_latent_tsne(self, f2f_loader, orig_loader, df_loader, fs_loader, file_name, model_path,
                              num_clusters=4):
        """
        Function to generate and save the latent space
        :param f2f_loader:
        :param orig_loader:
        :param file_name:
        :param model_path:
        :return:
        """

        try:
            rootLogger.info("Loading F2F Model")
            checkpoint = torch.load(model_path + '/autoencoders/f2f/best/' + self.model_name + '.pt')
            self.autoencoder.load_state_dict(checkpoint)
            rootLogger.info("Saved Model successfully loaded")
        except:
            rootLogger.info("Model not found.")
        latent_all = None
        target_all = None

        # Latent for F2F
        with torch.no_grad():
            for epoch_iter, data in enumerate(f2f_loader):
                input_image, _ = data
                if self.use_cuda:
                    input_image = input_image.cuda()

                latent, _ = self.autoencoder(input_image)

                if latent_all is None:
                    latent_all = latent
                    target_all = torch.ones(self.batch_size, dtype=torch.int64)
                else:
                    latent_all = torch.cat([latent_all, latent], dim=0)
                    target_all = torch.cat([target_all, torch.ones(self.batch_size, dtype=torch.int64)], dim=0)

        # Latent for DeepFakes
        try:
            rootLogger.info("Loading DF Model")
            checkpoint = torch.load(model_path + '/autoencoders/df/best/' + self.model_name + '.pt')
            self.autoencoder.load_state_dict(checkpoint)
            rootLogger.info("Saved Model successfully loaded")
        except:
            rootLogger.info("Model not found.")
        with torch.no_grad():
            for epoch_iter, data in enumerate(df_loader):
                input_image, _ = data
                if self.use_cuda:
                    input_image = input_image.cuda()

                latent, _ = self.autoencoder(input_image)
                latent_all = torch.cat([latent_all, latent], dim=0)
                targets = torch.zeros(self.batch_size, dtype=torch.int64)
                targets = targets.fill_(2)
                target_all = torch.cat([target_all, targets], dim=0)

        # Latent for FS
        with torch.no_grad():
            for epoch_iter, data in enumerate(fs_loader):
                input_image, _ = data
                if self.use_cuda:
                    input_image = input_image.cuda()

                latent, _ = self.autoencoder(input_image)

                latent_all = torch.cat([latent_all, latent], dim=0)
                targets = torch.zeros(self.batch_size, dtype=torch.int64)
                targets = targets.fill_(3)
                target_all = torch.cat([target_all, targets], dim=0)
        # Latent for Original
        try:
            rootLogger.info("Loading orig Model")
            checkpoint = torch.load(model_path + '/autoencoders/orig/best/' + self.model_name + '.pt')
            self.autoencoder.load_state_dict(checkpoint)
            rootLogger.info("Saved Model successfully loaded")
        except:
            rootLogger.info("Model not found.")
        with torch.no_grad():
            for epoch_iter, data in enumerate(orig_loader):
                input_image, _ = data
                if self.use_cuda:
                    input_image = input_image.cuda()

                latent, _ = self.autoencoder(input_image)
                latent_all = torch.cat([latent_all, latent], dim=0)
                target_all = torch.cat([target_all, torch.zeros(self.batch_size, dtype=torch.int64)], dim=0)

        latent_all = latent_all.cpu().numpy()
        target_all = target_all.cpu().numpy()
        target_all = target_all.astype(int)
        rootLogger.info("T-SNE visualization started")
        tsne = manifold.TSNE(n_components=3, init='pca', random_state=0)
        x_tsne = tsne.fit_transform(latent_all)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for label_id in np.unique(target_all):
            ax.scatter3D(x_tsne[np.where(target_all == label_id), 0], x_tsne[np.where(target_all == label_id), 1],
                          x_tsne[np.where(target_all == label_id), 2],
                        marker='o', color=category_to_color[num_clusters], linewidth=1,
                        alpha=0.8,label=category_to_label[label_id])
        plt.legend(loc='best')
        #plt.scatter(x_tsne[:, 0], x_tsne[:, 1], s=2, c=target_all, cmap=plt.cm.get_cmap("jet", num_clusters))
        #plt.colorbar(ticks=range(num_clusters))
        fig.savefig(file_name + ".png", dpi=fig.dpi)

    def visualize_tsne(self, val_loader, model_path, file_name):
        """
        Function to evaluate the result on test data
        :param val_loader:
        :param model_path:target_all
        :param image_path_pred:
        :param image_path_gt:
        :return:
        """

        try:
            rootLogger.info("Loading Saved Model")
            checkpoint = torch.load(model_path + '/autoencoders/' + self.model_dataset + '/best/' + self.model_name + '.pt')
            self.autoencoder.load_state_dict(checkpoint)
            rootLogger.info("Saved Model successfully loaded")
        except:
            rootLogger.info("Model not found.")

        latent_all, target_all = None, None
        with torch.no_grad():
            for epoch_iter, data in enumerate(val_loader):
                input_image, target = data
                if self.use_cuda:
                    input_image = input_image.cuda()
                    target = target.cuda()
                latent, _ = self.autoencoder(input_image, target)
                if latent_all is None:
                    latent_all = latent
                    target_all = target
                    #pred_images = input_image
                else:
                    latent_all = torch.cat([latent_all, latent], dim=0)
                    target_all = torch.cat([target_all, target], dim=0)
                    #pred_images = torch.cat([pred_images, input_image], dim=0)

        latent_all = latent_all.cpu().numpy()
        target_all = target_all.cpu().numpy()
        #pred_images = pred_images.cpu().numpy()
        # tsne = TSNE(n_components=2, init='pca', random_state=0).fit_transform(latent_all)
        #
        # tx, ty = tsne[:, 0], tsne[:, 1]
        # tx = (tx - np.min(tx)) / (np.max(tx) - np.min(tx))
        # ty = (ty - np.min(ty)) / (np.max(ty) - np.min(ty))
        #
        # full_image = Image.new('RGB', (width, height))
        # for idx, x in enumerate(pred_images):
        #     tile = Image.fromarray(np.uint8(np.transpose(x, [1, 2, 0]) * 255))
        #     # Image._show(tile)
        #     rs = max(1, tile.width / max_dim, tile.height / max_dim)
        #     tile = tile.resize((int(tile.width / rs),
        #                         int(tile.height / rs)),
        #                        Image.ANTIALIAS)
        #     full_image.paste(tile, (int((width - max_dim) * tx[idx]),
        #                             int((height - max_dim) * ty[idx])))
        #
        # plt.figure(figsize=(16, 12))
        # plt.imsave(fname=file_name + ".png", arr=full_image)

        rootLogger.info("T-SNE visualization started")
        tsne = manifold.TSNE(n_components=3, init='pca', random_state=0)
        x_tsne = tsne.fit_transform(latent_all)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        print(np.unique(target_all))
        for label_id in np.unique(target_all):
            ax.scatter3D(x_tsne[np.where(target_all == label_id), 0], x_tsne[np.where(target_all == label_id), 1],
                         x_tsne[np.where(target_all == label_id), 2],
                         marker='o', color=plt.cm.Set1(label_id), linewidth=1,
                         alpha=0.8, label=category_to_label[label_id])
        plt.legend(loc='best')
        fig.savefig(file_name + ".png", dpi=fig.dpi)

    def generate_latent_space(self, data_loader, model_path, file_name, mode):
        """
        Function to generate and save the latent space
        :param data_loader:
        :param model_path:
        :param path:
        :return:
        """

        try:
            rootLogger.info("Loading Saved Model")
            checkpoint = torch.load(
                model_path + '/autoencoders/' + self.model_dataset + '/best/' + self.model_name + '.pt')
            self.autoencoder.load_state_dict(checkpoint)
            rootLogger.info("Saved Model successfully loaded")
        except:
            rootLogger.info("Model not found.")
        latent_all = None
        target_all = None
        with torch.no_grad():
            for epoch_iter, data in enumerate(data_loader):
                input_image, target = data
                if self.use_cuda:
                    input_image = input_image.cuda()

                latent, _ = self.autoencoder(input_image, target)

                if latent_all is None:
                    latent_all = latent
                    if mode == 'ff':
                        target_all = target
                    elif mode == 'f2f' or mode == 'df' or mode == 'fs':
                        target_all = torch.ones(self.batch_size, dtype=torch.int64)
                    elif mode == 'orig':
                        target_all = torch.zeros(self.batch_size, dtype=torch.int64)
                else:
                    latent_all = torch.cat([latent_all, latent], dim=0)
                    if mode == 'ff':
                        target_all = torch.cat([target_all, target], dim=0)
                    elif mode == 'f2f' or mode == 'df' or mode == 'fs':
                        target_all = torch.cat([target_all, torch.ones(self.batch_size, dtype=torch.int64)], dim=0)
                    elif mode == 'orig':
                        target_all = torch.cat([target_all, torch.zeros(self.batch_size, dtype=torch.int64)], dim=0)

        latent_all = latent_all.cpu().numpy()
        target_all = target_all.cpu().numpy()

        # if "train" in file_name:
        #     latent_all = latent_all[:100]
        #     target_all = target_all[:100]

        print(latent_all.shape)
        print(target_all.shape)
        print(np.unique(target_all))

        np.save(file_name + '_X.npy', latent_all)
        np.save(file_name + '_Y.npy', target_all)
        rootLogger.info(file_name + " Saved !!")

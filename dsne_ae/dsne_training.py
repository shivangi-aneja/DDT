import os
import time
import torch
from os import makedirs
from torch.autograd import Variable
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from common.logging.logger import rootLogger
from common.logging.tf_logger import Logger
from common.losses.dSNE_loss import dSNELoss
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary
from mpl_toolkits.mplot3d import Axes3D
from sklearn import manifold

# Train : recon=2, center_loss=0.5, act_loss=1
# Finetune : recon=2, center_loss=0.5, act_loss=2
embed_size = 128
margin=1
alpha_recon_loss = 0.3
alpha_dsne_loss = 0.2
alpha_class_loss = 0.5
category_to_label = {0: 'Orig', 1: 'F2F', 2: 'DF', 3: 'FS'}
category_to_color = np.array(['#ff80ed', '#065535', '#133337', '#ffc0cb.'])


class Autoencoder(object):

    def __init__(self, autoencoder, model_name, recon_loss, optim_kwargs, dataset, num_classes, batch_size=64,
                 optim=None, epochs=10, tf_log_path=None, use_cuda=None, selection=False,
                 patience=50, counter=0, early_stop=False, model_dir='dsne_ae'):
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
        autoencoder_params = filter(lambda x: x.requires_grad, self.autoencoder.parameters())
        params = sum([np.prod(p.size()) for p in self.autoencoder.parameters()])
        rootLogger.info("Trainable Parameters : " + str(params))

        optim = optim or torch.optim.Adam
        optim_kwargs = optim_kwargs or {}
        optim_kwargs.setdefault('lr', 1e-3)
        self.optim = optim(autoencoder_params, **optim_kwargs)
        self.epochs = epochs
        self.num_classes = num_classes
        self.classification_loss = nn.CrossEntropyLoss(reduction='none')
        self.recon_loss = recon_loss
        self.model_name = model_name
        self.dataset = dataset
        self.batch_size = batch_size
        self.selection = selection
        self.use_cuda = use_cuda
        self.patience = patience
        self.counter = counter
        self.early_stop = early_stop
        self.model_dir = model_dir
        self.scheduler = StepLR(self.optim, step_size=10, gamma=0.9)
        if self.use_cuda is None:
            self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.autoencoder.cuda()
        if tf_log_path is not None:
            self.logger = Logger(model_name=self.model_name, data_name=self.dataset, log_path=tf_log_path)

    def train_autoencoder(self, train_loader, val_loader, model_path):
        """
        Function for training
        :param train_loader: Loader for training data
        :param test_loader: Loader for test data
        :param model_path: Path for saving the data
        """

        # results save folder
        model_current = model_path + '/dsne_models/' + self.model_dir + '/' + self.dataset + '/current/' + self.model_name + '.pt'
        model_best = model_path + '/dsne_models/' + self.model_dir + '/' + self.dataset + '/best/' + self.model_name + '.pt'

        try:
            rootLogger.info("Loading Saved Model")
            checkpoint = torch.load(model_current)
            self.autoencoder.load_state_dict(checkpoint)
            rootLogger.info("Saved Model successfully loaded")
        except:
            rootLogger.info("Model not found, Created a new one")
            # Make directory for Saving Models
            if not os.path.isdir(model_path + '/dsne_models/' + self.model_dir + '/' + self.dataset + '/current/'):
                makedirs(model_path + '/dsne_models/' + self.model_dir + '/' + self.dataset + '/current/')
            if not os.path.isdir(model_path + '/dsne_models/' + self.model_dir + '/' + self.dataset + '/best/'):
                makedirs(model_path + '/dsne_models/' + self.model_dir + '/' + self.dataset + '/best/')

        # training-loop
        np.random.seed(int(time.time()))
        rootLogger.info('Training Start!!!')
        best_loss = np.Inf

        for epoch in range(self.epochs):
            self.autoencoder.train()
            epoch_start_time = time.time()

            epoch_train_loss_s = 0.
            epoch_train_recon_loss_s = 0.
            epoch_train_class_loss_s = 0.
            epoch_train_dsne_loss_s = 0.

            epoch_train_loss_t = 0.
            epoch_train_recon_loss_t = 0.
            epoch_train_class_loss_t = 0.
            epoch_train_dsne_loss_t = 0.
            correct_s = 0
            total_s = 0
            correct_t = 0
            total_t = 0
            # Checkpoint after 5 epochs
            if (epoch + 1) % 5 == 0:
                try:
                    rootLogger.info("Saving the model after 5 epochs")
                    torch.save(self.autoencoder.state_dict(), model_current)
                    rootLogger.info("Model Saved")
                except:
                    rootLogger.info("Can't save the model")

            for epoch_iter, data in enumerate(train_loader):

                Xs = data[0]
                Ys = data[1]
                Xt = data[2]
                Yt = data[3]
                # Move the images to the device first before computation
                if self.use_cuda:
                    Xs = Xs.cuda()
                    Ys = Ys.cuda()
                    Xt = Xt.cuda()
                    Yt = Yt.cuda()

                Xs = Variable(Xs)
                Ys = Variable(Ys)
                Xt = Variable(Xt)
                Yt = Variable(Yt)

                embed_s, ys_hat, _, _, classifier_loss_src, dsne_loss_src, recon_loss_src, loss_src, _ = self.update_weights(Xs=Xs, Ys=Ys,
                                                                    Xt=Xt, Yt=Yt, optimize=True)
                epoch_train_loss_s += loss_src
                epoch_train_recon_loss_s += recon_loss_src
                epoch_train_class_loss_s += classifier_loss_src
                epoch_train_dsne_loss_s += dsne_loss_src

                embed_t, yt_hat, _, _, classifier_loss_tar, dsne_loss_tar, recon_loss_tar, loss_tar, _ = self.update_weights(Xs=Xt, Ys=Yt,
                                                                                                          Xt=Xs, Yt=Ys, optimize=True)
                epoch_train_loss_t += loss_tar
                epoch_train_recon_loss_t += recon_loss_tar
                epoch_train_class_loss_t += classifier_loss_tar
                epoch_train_dsne_loss_t += dsne_loss_tar

                _, ys_hat = torch.max(ys_hat.data, 1)
                total_s += ys_hat.size(0)
                _, yt_hat = torch.max(yt_hat.data, 1)
                total_t += yt_hat.size(0)
                correct_s += (ys_hat == Ys).sum().item()
                correct_t += (yt_hat == Yt).sum().item()

            avg_loss_s = epoch_train_loss_s / len(train_loader)
            avg_loss_class_s = epoch_train_class_loss_s / len(train_loader)
            avg_loss_dsne_s = epoch_train_dsne_loss_s / len(train_loader)
            avg_loss_recon_s = epoch_train_recon_loss_s / len(train_loader)
            avg_loss_t = epoch_train_loss_t / len(train_loader)
            avg_loss_class_t = epoch_train_class_loss_t / len(train_loader)
            avg_loss_dsne_t = epoch_train_dsne_loss_t / len(train_loader)
            avg_loss_recon_t = epoch_train_recon_loss_t / len(train_loader)

            # Calculate accuracy for current epoch
            accuracy_s = 100 * correct_s / total_s
            accuracy_t = 100 * correct_t / total_t

            # Log the training losses and accuracy
            self.logger.log(mode="train_src", error=avg_loss_s, epoch=epoch + 1, n_batch=0, num_batches=1,
                            scalar='avg_loss')
            self.logger.log(mode="train_src", error=avg_loss_class_s, epoch=epoch + 1, n_batch=0, num_batches=1,
                            scalar='class_loss')
            self.logger.log(mode="train_src", error=avg_loss_dsne_s, epoch=epoch + 1, n_batch=0, num_batches=1,
                            scalar='dsne_loss')
            self.logger.log(mode="train_src", error=avg_loss_recon_s, epoch=epoch + 1, n_batch=0, num_batches=1,
                            scalar='recon_loss')
            self.logger.log(mode="train_src", error=accuracy_s, epoch=epoch + 1, n_batch=0, num_batches=1,
                            scalar='accuracy')

            self.logger.log(mode="train_tar", error=avg_loss_t, epoch=epoch + 1, n_batch=0, num_batches=1,
                            scalar='avg_loss')
            self.logger.log(mode="train_tar", error=avg_loss_class_t, epoch=epoch + 1, n_batch=0, num_batches=1,
                            scalar='class_loss')
            self.logger.log(mode="train_tar", error=avg_loss_dsne_t, epoch=epoch + 1, n_batch=0, num_batches=1,
                            scalar='dsne_loss')
            self.logger.log(mode="train_tar", error=avg_loss_recon_t, epoch=epoch + 1, n_batch=0, num_batches=1,
                            scalar='recon_loss')
            self.logger.log(mode="train_tar", error=accuracy_t, epoch=epoch + 1, n_batch=0, num_batches=1,
                            scalar='accuracy')
            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - epoch_start_time
            rootLogger.info(
                '[%d/%d] - ptime: %.2f Train loss_s: %.3f loss_class_s: %.3f loss_dsne_s: %.3f  acc_s: %.2f  loss_t: %.3f loss_class_t: %.3f loss_dsne_t: %.3f acc_t: %.2f  ' %
                (
                    (epoch + 1), self.epochs, per_epoch_ptime, avg_loss_s, avg_loss_class_s, avg_loss_dsne_s,
                    accuracy_s, avg_loss_t, avg_loss_class_t, avg_loss_dsne_t, accuracy_t))

            # Validation
            val_loss, val_acc = self.evaluate_val_data_tf(val_loader=val_loader, epoch=epoch)

            # Save the best model
            if val_loss <= best_loss:
                self.counter = 0
                self.autoencoder.train()
                best_loss = val_loss
                torch.save(self.autoencoder.state_dict(), model_best)
                rootLogger.info("Best model saved/updated..")
            else:
                self.counter += 1
                rootLogger.info("EarlyStopping counter: " + str(self.counter) + " out of " + str(self.patience))
                if self.counter >= self.patience:
                    self.early_stop = True
            # If early stopping flag is true, then stop the training
            if self.early_stop:
                rootLogger.info("Early stopping")
                break
            self.scheduler.step()

    def train_autoencoder_source_only(self, train_loader, val_loader, model_path):
        """
        Function for training
        :param train_loader: Loader for training data
        :param test_loader: Loader for test data
        :param model_path: Path for saving the data
        """

        # results save folder
        model_current = model_path + '/dsne_models/' + self.model_dir + '/' + self.dataset + '/current/' + self.model_name + '.pt'
        model_best = model_path + '/dsne_models/' + self.model_dir + '/' + self.dataset + '/best/' + self.model_name + '.pt'

        try:
            rootLogger.info("Loading Saved Model")
            checkpoint = torch.load(model_current)
            self.autoencoder.load_state_dict(checkpoint)
            rootLogger.info("Saved Model successfully loaded")
        except:
            rootLogger.info("Model not found, Created a new one")
            # Make directory for Saving Models
            if not os.path.isdir(model_path + '/dsne_models/' + self.model_dir + '/' + self.dataset + '/current/'):
                makedirs(model_path + '/dsne_models/' + self.model_dir + '/' + self.dataset + '/current/')
            if not os.path.isdir(model_path + '/dsne_models/' + self.model_dir + '/' + self.dataset + '/best/'):
                makedirs(model_path + '/dsne_models/' + self.model_dir + '/' + self.dataset + '/best/')

        # training-loop
        np.random.seed(int(time.time()))
        rootLogger.info('Training Start!!!')
        best_loss = np.Inf

        for epoch in range(self.epochs):
            self.autoencoder.train()
            epoch_start_time = time.time()

            epoch_train_loss_s = 0.
            epoch_train_recon_loss_s = 0.
            epoch_train_class_loss_s = 0.

            correct_s = 0
            total_s = 0

            # Checkpoint after 5 epochs
            if (epoch + 1) % 5 == 0:
                try:
                    rootLogger.info("Saving the model after 5 epochs")
                    torch.save(self.autoencoder.state_dict(), model_current)
                    rootLogger.info("Model Saved")
                except:
                    rootLogger.info("Can't save the model")

            for epoch_iter, data in enumerate(train_loader):

                Xs = data[0]
                Ys = data[1]
                # Move the images to the device first before computation
                if self.use_cuda:
                    Xs = Xs.cuda()
                    Ys = Ys.cuda()

                Xs = Variable(Xs)
                Ys = Variable(Ys)

                # Train for source
                self.optim.zero_grad()  # clear gradients for this training step
                embed_s, ys_hat, xs_recon = self.autoencoder(Xs)
                classifier_loss = self.classification_loss(ys_hat, Ys)
                recon_loss = torch.mean(self.recon_loss(xs_recon, Xs), dim=[1, 2, 3])
                loss = alpha_class_loss * classifier_loss + alpha_recon_loss * recon_loss

                for i, l in enumerate(loss):
                    if i == self.batch_size - 1:
                        l.backward()
                    else:
                        l.backward(retain_graph=True)  # backpropagation, compute gradients
                    self.optim.step()  # apply gradients

                epoch_train_loss_s += torch.mean(loss)
                epoch_train_recon_loss_s += torch.mean(recon_loss)
                epoch_train_class_loss_s += torch.mean(classifier_loss)

                _, ys_hat = torch.max(ys_hat.data, 1)
                total_s += ys_hat.size(0)
                correct_s += (ys_hat == Ys).sum().item()

            avg_loss_s = epoch_train_loss_s / len(train_loader)
            avg_loss_class_s = epoch_train_class_loss_s / len(train_loader)
            avg_loss_recon_s = epoch_train_recon_loss_s / len(train_loader)

            # Calculate accuracy for current epoch
            accuracy_s = 100 * correct_s / total_s

            # Log the training losses and accuracy
            self.logger.log(mode="train_src", error=avg_loss_s, epoch=epoch + 1, n_batch=0, num_batches=1,
                            scalar='avg_loss')
            self.logger.log(mode="train_src", error=avg_loss_class_s, epoch=epoch + 1, n_batch=0, num_batches=1,
                            scalar='class_loss')
            self.logger.log(mode="train_src", error=avg_loss_recon_s, epoch=epoch + 1, n_batch=0, num_batches=1,
                            scalar='recon_loss')
            self.logger.log(mode="train_src", error=accuracy_s, epoch=epoch + 1, n_batch=0, num_batches=1,
                            scalar='accuracy')


            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - epoch_start_time
            rootLogger.info(
                '[%d/%d] - ptime: %.2f Train loss_s: %.3f loss_class_s: %.3f loss_recon: %.3f  acc_s: %.2f  ' %
                ((epoch + 1), self.epochs, per_epoch_ptime, avg_loss_s, avg_loss_class_s, avg_loss_recon_s, accuracy_s))

            # Validation
            val_loss, val_acc = self.evaluate_val_data_tf_source(val_loader=val_loader, epoch=epoch)

            # Save the best model
            if val_loss <= best_loss:
                self.counter = 0
                self.autoencoder.train()
                best_loss = val_loss
                torch.save(self.autoencoder.state_dict(), model_best)
                rootLogger.info("Best model saved/updated..")
            else:
                self.counter += 1
                rootLogger.info("EarlyStopping counter: " + str(self.counter) + " out of " + str(self.patience))
                if self.counter >= self.patience:
                    self.early_stop = True
            # If early stopping flag is true, then stop the training
            if self.early_stop:
                rootLogger.info("Early stopping")
                break
            self.scheduler.step()

    # For evaluation during training
    def evaluate_val_data_tf(self, val_loader, epoch):
        """
        Function to evaluate the results on trained model
        :param val_loader: data loader on which clustering is evaluated
        :param epoch:
        :return: None
        """
        self.autoencoder.eval()
        # Load the parameters of pretrained model
        # checkpoint = torch.load(model_name)
        # Evaluate the results on current model
        epoch_loss_src = 0.
        epoch_recon_loss_s = 0.
        epoch_class_loss_s = 0.
        epoch_dsne_loss_s = 0.

        epoch_loss_tar = 0.
        epoch_recon_loss_t = 0.
        epoch_class_loss_t = 0.
        epoch_dsne_loss_t = 0.
        correct_src = 0
        correct_tar = 0
        total_src = 0
        total_tar = 0

        with torch.no_grad():
            for epoch_iter, data in enumerate(val_loader):
                Xs = data[0]
                Ys = data[1]
                Xt = data[2]
                Yt = data[3]
                # Move the images to the device first before computation
                if self.use_cuda:
                    Xs = Xs.cuda()
                    Ys = Ys.cuda()
                    Xt = Xt.cuda()
                    Yt = Yt.cuda()
                Xs = Variable(Xs)
                Ys = Variable(Ys)
                Xt = Variable(Xt)
                Yt = Variable(Yt)

                embed_s, ys_hat, _, _, classifier_loss_src, dsne_loss_src, recon_loss_src, loss_src, xs_recon = self.update_weights(Xs=Xs, Ys=Ys, Xt=Xt, Yt=Yt)
                epoch_loss_src += loss_src
                epoch_recon_loss_s += recon_loss_src
                epoch_class_loss_s += classifier_loss_src
                epoch_dsne_loss_s += dsne_loss_src

                embed_t, yt_hat, _, _, classifier_loss_tar, dsne_loss_tar, recon_loss_tar, loss_tar, xt_recon = self.update_weights(Xs=Xt, Ys=Yt, Xt=Xs, Yt=Ys)
                epoch_loss_tar += loss_tar
                epoch_recon_loss_t += recon_loss_tar
                epoch_class_loss_t += classifier_loss_tar
                epoch_dsne_loss_t += dsne_loss_tar

                _, ys_hat = torch.max(ys_hat.data, 1)
                total_src += ys_hat.size(0)
                _, yt_hat = torch.max(yt_hat.data, 1)
                total_tar += yt_hat.size(0)
                correct_src += (ys_hat == Ys).sum().item()
                correct_tar += (yt_hat == Yt).sum().item()

                if epoch_iter == len(val_loader) - 1:
                    self.logger.log_images(mode='pred_src', images=xs_recon, num_images=len(xs_recon),
                                           epoch=epoch, n_batch=1, num_batches=len(val_loader), normalize=True)

                    self.logger.log_images(mode='gt_src', images=Xs, num_images=len(Xs),
                                           epoch=epoch, n_batch=1, num_batches=len(val_loader), normalize=True)

                    self.logger.log_images(mode='pred_tar', images=xt_recon, num_images=len(xt_recon),
                                           epoch=epoch, n_batch=1, num_batches=len(val_loader), normalize=True)

                    self.logger.log_images(mode='gt_tar', images=Xt, num_images=len(Xt),
                                           epoch=epoch, n_batch=1, num_batches=len(val_loader), normalize=True)


            avg_loss_s = epoch_loss_src / len(val_loader)
            avg_loss_recon_s = epoch_recon_loss_s / len(val_loader)
            avg_loss_class_s = epoch_class_loss_s / len(val_loader)
            avg_loss_dsne_s = epoch_dsne_loss_s / len(val_loader)
            avg_loss_t = epoch_loss_tar / len(val_loader)
            avg_loss_recon_t = epoch_recon_loss_t / len(val_loader)
            avg_loss_class_t = epoch_class_loss_t / len(val_loader)
            avg_loss_dsne_t = epoch_dsne_loss_t / len(val_loader)

            # Calculate accuracy for current epoch
            accuracy_s = 100 * correct_src / total_src
            accuracy_t = 100 * correct_tar / total_tar

            # Log the training losses and accuracy
            self.logger.log(mode="val_src", error=avg_loss_s, epoch=epoch + 1, n_batch=0, num_batches=1,
                            scalar='avg_loss')
            self.logger.log(mode="val_src", error=avg_loss_class_s, epoch=epoch + 1, n_batch=0, num_batches=1,
                            scalar='class_loss')
            self.logger.log(mode="val_src", error=avg_loss_dsne_s, epoch=epoch + 1, n_batch=0, num_batches=1,
                            scalar='dsne_loss')
            self.logger.log(mode="val_src", error=avg_loss_recon_s, epoch=epoch + 1, n_batch=0, num_batches=1,
                            scalar='recon_loss')
            self.logger.log(mode="val_src", error=accuracy_s, epoch=epoch + 1, n_batch=0, num_batches=1,
                            scalar='accuracy')

            self.logger.log(mode="val_tar", error=avg_loss_t, epoch=epoch + 1, n_batch=0, num_batches=1,
                            scalar='avg_loss')
            self.logger.log(mode="val_tar", error=avg_loss_class_t, epoch=epoch + 1, n_batch=0, num_batches=1,
                            scalar='class_loss')
            self.logger.log(mode="val_tar", error=avg_loss_dsne_t, epoch=epoch + 1, n_batch=0, num_batches=1,
                            scalar='dsne_loss')
            self.logger.log(mode="val_tar", error=avg_loss_recon_t, epoch=epoch + 1, n_batch=0, num_batches=1,
                            scalar='recon_loss')
            self.logger.log(mode="val_tar", error=accuracy_t, epoch=epoch + 1, n_batch=0, num_batches=1,
                            scalar='accuracy')
            rootLogger.info('[%d/%d] - Val loss_s: %.3f loss_class_s: %.3f loss_dsne_s: %.3f  acc_s: %.2f  loss_t: %.3f loss_class_t: %.3f loss_dsne_t: %.3f acc_t: %.2f  ' %
                            (
            (epoch + 1), self.epochs, avg_loss_s, avg_loss_class_s, avg_loss_dsne_s, accuracy_s, avg_loss_t, avg_loss_class_t, avg_loss_dsne_t, accuracy_t))

        return avg_loss_s, accuracy_s

    def evaluate_val_data_tf_source(self, val_loader, epoch):
        """
        Function to evaluate the results on trained model
        :param val_loader: data loader on which clustering is evaluated
        :param epoch:
        :return: None
        """
        self.autoencoder.eval()
        # Load the parameters of pretrained model
        # checkpoint = torch.load(model_name)
        # Evaluate the results on current model
        epoch_loss = 0.
        epoch_recon_loss = 0.
        epoch_class_loss = 0.

        correct_src = 0
        total_src = 0

        with torch.no_grad():
            for epoch_iter, data in enumerate(val_loader):
                Xs = data[0]
                Ys = data[1]
                # Move the images to the device first before computation
                if self.use_cuda:
                    Xs = Xs.cuda()
                    Ys = Ys.cuda()
                Xs = Variable(Xs)
                Ys = Variable(Ys)

                # Train for source
                self.optim.zero_grad()  # clear gradients for this training step
                embed_s, ys_hat, xs_recon = self.autoencoder(Xs)
                classifier_loss = self.classification_loss(ys_hat, Ys)
                recon_loss = torch.mean(self.recon_loss(xs_recon, Xs), dim=[1, 2, 3])
                loss = alpha_class_loss * classifier_loss + alpha_recon_loss * recon_loss

                epoch_loss += torch.mean(loss)
                epoch_recon_loss += torch.mean(recon_loss)
                epoch_class_loss += torch.mean(classifier_loss)

                _, ys_hat = torch.max(ys_hat.data, 1)
                total_src += ys_hat.size(0)
                correct_src += (ys_hat == Ys).sum().item()

                if epoch_iter == len(val_loader) - 1:
                    self.logger.log_images(mode='pred_src', images=xs_recon, num_images=len(xs_recon),
                                           epoch=epoch, n_batch=1, num_batches=len(val_loader), normalize=True)

                    self.logger.log_images(mode='gt_src', images=Xs, num_images=len(Xs),
                                           epoch=epoch, n_batch=1, num_batches=len(val_loader), normalize=True)

            avg_loss_s = epoch_loss / len(val_loader)
            avg_loss_recon_s = epoch_recon_loss / len(val_loader)
            avg_loss_class_s = epoch_class_loss / len(val_loader)

            # Calculate accuracy for current epoch
            accuracy_s = 100 * correct_src / total_src

            # Log the training losses and accuracy
            self.logger.log(mode="val_src", error=avg_loss_s, epoch=epoch + 1, n_batch=0, num_batches=1,
                            scalar='avg_loss')
            self.logger.log(mode="val_src", error=avg_loss_class_s, epoch=epoch + 1, n_batch=0, num_batches=1,
                            scalar='class_loss')
            self.logger.log(mode="val_src", error=avg_loss_recon_s, epoch=epoch + 1, n_batch=0, num_batches=1,
                            scalar='recon_loss')
            self.logger.log(mode="val_src", error=accuracy_s, epoch=epoch + 1, n_batch=0, num_batches=1,
                            scalar='accuracy')
            rootLogger.info('[%d/%d] - Val loss_s: %.3f loss_class_s: %.3f loss_recon: %.3f  acc_s: %.2f ' %
                            (
            (epoch + 1), self.epochs, avg_loss_s, avg_loss_class_s, avg_loss_recon_s, accuracy_s))

        return avg_loss_s, accuracy_s
    
    def update_weights(self, Xs, Ys, Xt, Yt, optimize=False):

        # Train for source
        self.optim.zero_grad()  # clear gradients for this training step
        dsne_loss_fn = dSNELoss(Xs.shape[0], Xt.shape[0], embed_size, margin, True)
        embed_s, ys_hat, xs_recon = self.autoencoder(Xs)
        embed_t, yt_hat, xt_recon = self.autoencoder(Xt)
        dsne_loss = dsne_loss_fn(fts=embed_s, ys=Ys, ftt=embed_t, yt=Yt)
        classifier_loss = self.classification_loss(ys_hat, Ys)
        recon_loss = torch.mean(self.recon_loss(xs_recon, Xs), dim=[1,2,3])
        loss = alpha_class_loss * classifier_loss + alpha_dsne_loss * dsne_loss + alpha_recon_loss * recon_loss
        if optimize:
            for i, l in enumerate(loss):
                if i == self.batch_size-1:
                    l.backward()
                else:
                    l.backward(retain_graph=True)  # backpropagation, compute gradients
                self.optim.step()  # apply gradients
        return embed_s, ys_hat, embed_t, yt_hat, torch.mean(classifier_loss), torch.mean(dsne_loss), torch.mean(recon_loss), torch.mean(loss), xs_recon
    
    # For the 2-class (F2F + Orig) case, and trained classes for 3-class (F2F + DF + Orig) case
    def validate_results(self, val_loader, model_path):
        """
        Function to evaluate the results on trained model
        :param val_loader: data loader on which clustering is evaluated
        :param model_path: name with which pre-trained model is saved
        :return: None
        """

        try:
            rootLogger.info("Loading Saved Model")
            checkpoint = torch.load(model_path + '/dsne_models/' + self.model_dir + '/' + self.dataset + '/best/' + self.model_name + '.pt')
            self.autoencoder.load_state_dict(checkpoint)
            rootLogger.info("Saved Model successfully loaded")
        except:
            rootLogger.info("Model not found.")
            exit()

        epoch_loss = 0.
        epoch_recon_loss = 0.
        epoch_class_loss = 0.
        epoch_dsne_loss = 0.
        correct = 0
        total = 0

        classes = ('orig', 'f2f')

        class_correct = list(0. for i in range(2))
        class_total = list(0. for i in range(2))
        self.autoencoder.eval()
        with torch.no_grad():
            for epoch_iter, data in enumerate(val_loader):
                Xs = data[0]
                Ys = data[1]
                # Move the images to the device first before computation
                if self.use_cuda:
                    Xs = Xs.cuda()
                    Ys = Ys.cuda()
                Xs = Variable(Xs)
                Ys = Variable(Ys)

                embed_s, ys_hat, _, _, classifier_loss, dsne_loss, recon_loss, loss, _ = self.update_weights(Xs=Xs, Ys=Ys, Xt=Xs, Yt=Ys)
                epoch_loss += loss
                epoch_recon_loss += recon_loss
                epoch_class_loss += classifier_loss
                epoch_dsne_loss += dsne_loss

                _, y_hat = torch.max(ys_hat.data, 1)
                total += y_hat.size(0)
                correct += (y_hat == Ys).sum().item()

            avg_loss = epoch_loss / len(val_loader)
            avg_recon_loss = epoch_recon_loss / len(val_loader)
            avg_loss_class = epoch_class_loss / len(val_loader)
            avg_loss_dsne = epoch_dsne_loss / len(val_loader)

            # Calculate accuracy for current epoch
            accuracy = 100 * correct / total

            rootLogger.info(
                ' Val loss: %.3f loss_class: %.3f loss_dsne: %.3f  loss_recon: %.3f accuracy: %.2f  ' % (avg_loss, avg_loss_class, avg_loss_dsne, avg_recon_loss, accuracy))

            # for i in range(2):
            #     print('Accuracy of %5s : %2d %%' % (
            #         classes[i], 100 * class_correct[i] / class_total[i]))
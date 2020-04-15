import os
import time
from os import makedirs
import numpy as np
import torch
import torch.nn as nn
from torchvision.models.vgg import VGG, model_urls, load_state_dict_from_url
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from common.logging.logger import rootLogger
from common.logging.tf_logger import Logger
from common.losses.dSNE_loss import dSNELoss
embed_size = 128
margin=1
alpha = 0.25


class Classifier(object):

    def __init__(self, classifier, model_name, classification_loss, optim_kwargs, dataset, batch_size,
                 optim=None, epochs=10, tf_log_path=None, use_cuda=None, patience=50, counter=0, early_stop=False,
                 classifier_params=None, num_classes=2, model_dir=None):
        """
        :param dsne: dsne Network
        :param model_name: Model Name
        :param recon_loss: Reconstruction Loss
        :param dataset: Dataset
        :param batch_size: Batch Size
        :param optim: Optimizer
        :param lr: Learning Rate
        :param epochs: Number of epochs
        :param tf_log_path: Tensorflow Log Path
        """
        self.classifier = classifier
        if classifier_params is None or not len(classifier_params):
            classifier_params = filter(lambda x: x.requires_grad, self.classifier.parameters())
            params = sum([np.prod(p.size()) for p in self.classifier.parameters()])
            rootLogger.info("Trainable Parameters : " + str(params))
        optim = optim or torch.optim.Adam
        optim_kwargs = optim_kwargs or {}
        optim_kwargs.setdefault('lr', 1e-3)
        self.optim = optim(classifier_params, **optim_kwargs)
        self.epochs = epochs
        self.classification_loss = classification_loss
        self.model_name = model_name
        self.dataset = dataset
        self.batch_size = batch_size
        self.use_cuda = use_cuda
        self.patience = patience
        self.counter = counter
        self.early_stop = early_stop
        self.num_classes = num_classes
        self.model_dir = model_dir
        self.scheduler = StepLR(self.optim, step_size=10, gamma=0.5)
        if self.use_cuda is None:
            self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.classifier.cuda()
        if tf_log_path is not None:
            self.logger = Logger(model_name=self.model_name, data_name=self.dataset, log_path=tf_log_path)

    def update_weights(self, Xs, Ys, Xt, Yt, optimize=False):

        # Train for source
        self.optim.zero_grad()  # clear gradients for this training step
        dsne_loss_fn = dSNELoss(Xs.shape[0], Xt.shape[0], embed_size, margin, True)
        embed_s, ys_hat = self.classifier(Xs)
        embed_t, yt_hat = self.classifier(Xt)
        dsne_loss = dsne_loss_fn(fts=embed_s, ys=Ys, ftt=embed_t, yt=Yt)
        classifier_loss = self.classification_loss(ys_hat, Ys)
        loss = (1 - alpha) * classifier_loss + alpha * dsne_loss
        if optimize:
            for i, l in enumerate(loss):
                if i == self.batch_size-1:
                    l.backward()
                else:
                    l.backward(retain_graph=True)  # backpropagation, compute gradients
                self.optim.step()  # apply gradients
        return embed_s, ys_hat, embed_t, yt_hat, torch.mean(classifier_loss), torch.mean(dsne_loss), torch.mean(loss)

    def update_weights_vgg(self, Xs, Ys, Xt, Yt, optimize=False):

        # Train for source
        self.optim.zero_grad()  # clear gradients for this training step
        dsne_loss_fn = dSNELoss(Xs.shape[0], Xt.shape[0], embed_size, margin, True)
        ys_hat = self.classifier(Xs)
        out = self.classifier.avgpool(self.classifier.features(Xs))
        out = out.view(out.size(0), -1)
        embed_s = self.classifier.classifier[:4](out)
        yt_hat = self.classifier(Xt)
        out = self.classifier.avgpool(self.classifier.features(Xt))
        out = out.view(out.size(0), -1)
        embed_t = self.classifier.classifier[:4](out)
        dsne_loss = dsne_loss_fn(fts=embed_s, ys=Ys, ftt=embed_t, yt=Yt)
        classifier_loss = self.classification_loss(ys_hat, Ys)
        loss = (1 - alpha) * classifier_loss + alpha * dsne_loss
        if optimize:
            for i, l in enumerate(loss):
                if i == self.batch_size-1:
                    l.backward()
                else:
                    l.backward(retain_graph=True)  # backpropagation, compute gradients
                self.optim.step()  # apply gradients
        return embed_s, ys_hat, embed_t, yt_hat, torch.mean(classifier_loss), torch.mean(dsne_loss), torch.mean(loss)

    def train_classifier(self, train_loader, val_loader, model_path, train_src):
        """

        :param train_loader:
        :param val_loader:
        :param model_path:
        :param train_src:
        :return:
        """
        # results save folder
        model_current = model_path + '/dsne_models/arch1/' + self.model_dir + '/' + self.dataset +'/current/' + self.model_name + '.pt'
        model_best = model_path + '/dsne_models/arch1/' + self.model_dir + '/' + self.dataset +'/best/' + self.model_name + '.pt'

        try:
            rootLogger.info("Loading Saved Model")
            checkpoint = torch.load(model_current)
            self.classifier.load_state_dict(checkpoint)
            rootLogger.info("Saved Model successfully loaded")
        except:
            rootLogger.info("Model not found, Created a new one")
            # Make directory for Saving Models
            if not os.path.isdir(model_path + '/dsne_models/arch1/' + self.model_dir + '/' + self.dataset +'/current/'):
                makedirs(model_path + '/dsne_models/arch1/' + self.model_dir + '/' + self.dataset +'/current/')
            if not os.path.isdir(model_path + '/dsne_models/arch1/' + self.model_dir + '/' + self.dataset +'/best/'):
                makedirs(model_path + '/dsne_models/arch1/' + self.model_dir + '/' + self.dataset +'/best/')

        # training-loop
        np.random.seed(int(time.time()))
        rootLogger.info('Training Start!!!')
        best_loss = np.Inf

        for epoch in range(self.epochs):
            self.classifier.train()
            epoch_start_time = time.time()
            epoch_train_loss_s = 0.
            epoch_train_class_loss_s = 0.
            epoch_train_dsne_loss_s = 0.
            epoch_train_loss_t = 0.
            epoch_train_class_loss_t = 0.
            epoch_train_dsne_loss_t = 0.
            correct_s = 0
            total_s = 0
            correct_t = 0
            total_t = 0

            # Checkpoint after 5 epochs
            if epoch % 10 == 0:
                try:
                    rootLogger.info("Saving the current model")
                    torch.save(self.classifier.state_dict(), model_current)
                    rootLogger.info("Current model saved")
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

                if train_src:
                    embed_s, ys_hat, _, _, classifier_loss_src, dsne_loss_src, loss_src = self.update_weights(Xs=Xs, Ys=Ys, Xt=Xt, Yt=Yt, optimize=True)
                    epoch_train_loss_s += loss_src
                    epoch_train_class_loss_s += classifier_loss_src
                    epoch_train_dsne_loss_s += dsne_loss_src

                embed_t, yt_hat, _, _, classifier_loss_tar, dsne_loss_tar, loss_tar = self.update_weights(Xs=Xt, Ys=Yt, Xt=Xs, Yt=Ys, optimize=True)
                epoch_train_loss_t += loss_tar
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
            avg_loss_t = epoch_train_loss_t / len(train_loader)
            avg_loss_class_t = epoch_train_class_loss_t / len(train_loader)
            avg_loss_dsne_t = epoch_train_dsne_loss_t / len(train_loader)

            # Calculate accuracy for current epoch
            accuracy_s = 100 * correct_s / total_s
            accuracy_t = 100 * correct_t / total_t

            # Log the training losses and accuracy
            self.logger.log(mode="train_src", error=avg_loss_s, epoch=epoch + 1, n_batch=0, num_batches=1, scalar='avg_loss')
            self.logger.log(mode="train_src", error=avg_loss_class_s, epoch=epoch + 1, n_batch=0, num_batches=1,
                            scalar='class_loss')
            self.logger.log(mode="train_src", error=avg_loss_dsne_s, epoch=epoch + 1, n_batch=0, num_batches=1,
                            scalar='dsne_loss')
            self.logger.log(mode="train_src", error=accuracy_s, epoch=epoch + 1, n_batch=0, num_batches=1,
                            scalar='accuracy')

            self.logger.log(mode="train_tar", error=avg_loss_t, epoch=epoch + 1, n_batch=0, num_batches=1,
                            scalar='avg_loss')
            self.logger.log(mode="train_tar", error=avg_loss_class_t, epoch=epoch + 1, n_batch=0, num_batches=1,
                            scalar='class_loss')
            self.logger.log(mode="train_tar", error=avg_loss_dsne_t, epoch=epoch + 1, n_batch=0, num_batches=1,
                            scalar='dsne_loss')
            self.logger.log(mode="train_tar", error=accuracy_t, epoch=epoch + 1, n_batch=0, num_batches=1,
                            scalar='accuracy')
            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - epoch_start_time
            rootLogger.info('[%d/%d] - ptime: %.2f Train loss_s: %.3f loss_class_s: %.3f loss_dsne_s: %.3f  acc_s: %.2f  loss_t: %.3f loss_class_t: %.3f loss_dsne_t: %.3f acc_t: %.2f  ' % (
                (epoch + 1), self.epochs, per_epoch_ptime, avg_loss_s, avg_loss_class_s, avg_loss_dsne_s, accuracy_s, avg_loss_t, avg_loss_class_t, avg_loss_dsne_t, accuracy_t))

            # Validation
            val_loss, val_acc = self.evaluate_val_data_tf(val_loader=val_loader, epoch=epoch)

            # Save the best model
            if val_loss <= best_loss:
                self.counter = 0
                self.classifier.train()
                best_loss = val_loss
                torch.save(self.classifier.state_dict(), model_best)
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

    def train_classifier_on_pretrained_imagenet(self, train_loader, val_loader, model_path, train_src):
        """

        :param train_loader:
        :param val_loader:
        :param model_path:
        :param train_src:
        :return:
        """
        # results save folder
        model_current = model_path + '/dsne_models/arch1/' + self.model_dir + '/' + self.dataset +'/current/' + self.model_name + '.pt'
        model_best = model_path + '/dsne_models/arch1/' + self.model_dir + '/' + self.dataset +'/best/' + self.model_name + '.pt'

        if not os.path.isdir(model_path + '/dsne_models/arch1/' + self.model_dir + '/' + self.dataset +'/current/'):
            makedirs(model_path + '/dsne_models/arch1/' + self.model_dir + '/' + self.dataset +'/current/')
        if not os.path.isdir(model_path + '/dsne_models/arch1/' + self.model_dir + '/' + self.dataset +'/best/'):
            makedirs(model_path + '/dsne_models/arch1/' + self.model_dir + '/' + self.dataset +'/best/')

        try:
            rootLogger.info("Loading ImageNet Pretrained")
            checkpoint = load_state_dict_from_url(model_urls['vgg16_bn'], progress=True)
            self.classifier.load_state_dict(checkpoint)
            num_ftrs = self.classifier.classifier[6].in_features
            self.classifier.classifier[6] = nn.Linear(num_ftrs, 2)  # Change the last layer output from 1000 to 2
            self.classifier.cuda()
            rootLogger.info("Imagenet successfully loaded")
        except:
            rootLogger.info("Model not found")
            exit()
            # Make directory for Saving Models

        # training-loop
        np.random.seed(int(time.time()))
        rootLogger.info('Training Start!!!')
        best_loss = np.Inf

        for epoch in range(self.epochs):
            self.classifier.train()
            epoch_start_time = time.time()
            epoch_train_loss_s = 0.
            epoch_train_class_loss_s = 0.
            epoch_train_dsne_loss_s = 0.
            epoch_train_loss_t = 0.
            epoch_train_class_loss_t = 0.
            epoch_train_dsne_loss_t = 0.
            correct_s = 0
            total_s = 0
            correct_t = 0
            total_t = 0

            # Checkpoint after 5 epochs
            if epoch % 10 == 0:
                try:
                    rootLogger.info("Saving the current model")
                    torch.save(self.classifier.state_dict(), model_current)
                    rootLogger.info("Current model saved")
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

                if train_src:
                    embed_s, ys_hat, _, _, classifier_loss_src, dsne_loss_src, loss_src = self.update_weights_vgg(Xs=Xs, Ys=Ys, Xt=Xt, Yt=Yt, optimize=True)
                    epoch_train_loss_s += loss_src
                    epoch_train_class_loss_s += classifier_loss_src
                    epoch_train_dsne_loss_s += dsne_loss_src

                embed_t, yt_hat, _, _, classifier_loss_tar, dsne_loss_tar, loss_tar = self.update_weights_vgg(Xs=Xt, Ys=Yt, Xt=Xs, Yt=Ys, optimize=True)
                epoch_train_loss_t += loss_tar
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
            avg_loss_t = epoch_train_loss_t / len(train_loader)
            avg_loss_class_t = epoch_train_class_loss_t / len(train_loader)
            avg_loss_dsne_t = epoch_train_dsne_loss_t / len(train_loader)

            # Calculate accuracy for current epoch
            accuracy_s = 100 * correct_s / total_s
            accuracy_t = 100 * correct_t / total_t

            # Log the training losses and accuracy
            self.logger.log(mode="train_src", error=avg_loss_s, epoch=epoch + 1, n_batch=0, num_batches=1, scalar='avg_loss')
            self.logger.log(mode="train_src", error=avg_loss_class_s, epoch=epoch + 1, n_batch=0, num_batches=1,
                            scalar='class_loss')
            self.logger.log(mode="train_src", error=avg_loss_dsne_s, epoch=epoch + 1, n_batch=0, num_batches=1,
                            scalar='dsne_loss')
            self.logger.log(mode="train_src", error=accuracy_s, epoch=epoch + 1, n_batch=0, num_batches=1,
                            scalar='accuracy')

            self.logger.log(mode="train_tar", error=avg_loss_t, epoch=epoch + 1, n_batch=0, num_batches=1,
                            scalar='avg_loss')
            self.logger.log(mode="train_tar", error=avg_loss_class_t, epoch=epoch + 1, n_batch=0, num_batches=1,
                            scalar='class_loss')
            self.logger.log(mode="train_tar", error=avg_loss_dsne_t, epoch=epoch + 1, n_batch=0, num_batches=1,
                            scalar='dsne_loss')
            self.logger.log(mode="train_tar", error=accuracy_t, epoch=epoch + 1, n_batch=0, num_batches=1,
                            scalar='accuracy')
            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - epoch_start_time
            rootLogger.info('[%d/%d] - ptime: %.2f Train loss_s: %.3f loss_class_s: %.3f loss_dsne_s: %.3f  acc_s: %.2f  loss_t: %.3f loss_class_t: %.3f loss_dsne_t: %.3f acc_t: %.2f  ' % (
                (epoch + 1), self.epochs, per_epoch_ptime, avg_loss_s, avg_loss_class_s, avg_loss_dsne_s, accuracy_s, avg_loss_t, avg_loss_class_t, avg_loss_dsne_t, accuracy_t))

            # Validation
            val_loss, val_acc = self.evaluate_val_data_tf_vgg(val_loader=val_loader, epoch=epoch)

            # Save the best model
            if val_loss <= best_loss:
                self.counter = 0
                self.classifier.train()
                best_loss = val_loss
                torch.save(self.classifier.state_dict(), model_best)
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

    def evaluate_val_data_tf(self, val_loader, epoch):
        """
        Function to evaluate the results on trained model
        :param val_loader: data loader on which clustering is evaluated
        :param epoch:
        :return: None
        """
        self.classifier.eval()
        # Load the parameters of pretrained model
        # checkpoint = torch.load(model_name)
        # Evaluate the results on current model
        epoch_loss_src = 0.
        epoch_class_loss_s = 0.
        epoch_dsne_loss_s = 0.
        epoch_loss_tar = 0.
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

                embed_s, ys_hat, _, _, classifier_loss_src, dsne_loss_src, loss_src = self.update_weights(Xs=Xs, Ys=Ys, Xt=Xt, Yt=Yt)
                epoch_loss_src += loss_src
                epoch_class_loss_s += classifier_loss_src
                epoch_dsne_loss_s += dsne_loss_src

                embed_t, yt_hat, _, _, classifier_loss_tar, dsne_loss_tar, loss_tar = self.update_weights(Xs=Xt, Ys=Yt, Xt=Xs, Yt=Ys)
                epoch_loss_tar += loss_tar
                epoch_class_loss_t += classifier_loss_tar
                epoch_dsne_loss_t += dsne_loss_tar

                _, ys_hat = torch.max(ys_hat.data, 1)
                total_src += ys_hat.size(0)
                _, yt_hat = torch.max(yt_hat.data, 1)
                total_tar += yt_hat.size(0)
                correct_src += (ys_hat == Ys).sum().item()
                correct_tar += (yt_hat == Yt).sum().item()

            avg_loss_s = epoch_loss_src / len(val_loader)
            avg_loss_class_s = epoch_class_loss_s / len(val_loader)
            avg_loss_dsne_s = epoch_dsne_loss_s / len(val_loader)
            avg_loss_t = epoch_loss_tar / len(val_loader)
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
            self.logger.log(mode="val_src", error=accuracy_s, epoch=epoch + 1, n_batch=0, num_batches=1,
                            scalar='accuracy')

            self.logger.log(mode="val_tar", error=avg_loss_t, epoch=epoch + 1, n_batch=0, num_batches=1,
                            scalar='avg_loss')
            self.logger.log(mode="val_tar", error=avg_loss_class_t, epoch=epoch + 1, n_batch=0, num_batches=1,
                            scalar='class_loss')
            self.logger.log(mode="val_tar", error=avg_loss_dsne_t, epoch=epoch + 1, n_batch=0, num_batches=1,
                            scalar='dsne_loss')
            self.logger.log(mode="val_tar", error=accuracy_t, epoch=epoch + 1, n_batch=0, num_batches=1,
                            scalar='accuracy')
            rootLogger.info('[%d/%d] - Val loss_s: %.3f loss_class_s: %.3f loss_dsne_s: %.3f  acc_s: %.2f  loss_t: %.3f loss_class_t: %.3f loss_dsne_t: %.3f acc_t: %.2f  ' % (
            (epoch + 1), self.epochs, avg_loss_s, avg_loss_class_s, avg_loss_dsne_s, accuracy_s, avg_loss_t, avg_loss_class_t, avg_loss_dsne_t, accuracy_t))

        return avg_loss_t, accuracy_t

    def evaluate_val_data_tf_vgg(self, val_loader, epoch):
        """
        Function to evaluate the results on trained model
        :param val_loader: data loader on which clustering is evaluated
        :param epoch:
        :return: None
        """
        self.classifier.eval()
        # Load the parameters of pretrained model
        # checkpoint = torch.load(model_name)
        # Evaluate the results on current model
        epoch_loss_src = 0.
        epoch_class_loss_s = 0.
        epoch_dsne_loss_s = 0.
        epoch_loss_tar = 0.
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

                embed_s, ys_hat, _, _, classifier_loss_src, dsne_loss_src, loss_src = self.update_weights_vgg(Xs=Xs, Ys=Ys, Xt=Xt, Yt=Yt)
                epoch_loss_src += loss_src
                epoch_class_loss_s += classifier_loss_src
                epoch_dsne_loss_s += dsne_loss_src

                embed_t, yt_hat, _, _, classifier_loss_tar, dsne_loss_tar, loss_tar = self.update_weights_vgg(Xs=Xt, Ys=Yt, Xt=Xs, Yt=Ys)
                epoch_loss_tar += loss_tar
                epoch_class_loss_t += classifier_loss_tar
                epoch_dsne_loss_t += dsne_loss_tar

                _, ys_hat = torch.max(ys_hat.data, 1)
                total_src += ys_hat.size(0)
                _, yt_hat = torch.max(yt_hat.data, 1)
                total_tar += yt_hat.size(0)
                correct_src += (ys_hat == Ys).sum().item()
                correct_tar += (yt_hat == Yt).sum().item()

            avg_loss_s = epoch_loss_src / len(val_loader)
            avg_loss_class_s = epoch_class_loss_s / len(val_loader)
            avg_loss_dsne_s = epoch_dsne_loss_s / len(val_loader)
            avg_loss_t = epoch_loss_tar / len(val_loader)
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
            self.logger.log(mode="val_src", error=accuracy_s, epoch=epoch + 1, n_batch=0, num_batches=1,
                            scalar='accuracy')

            self.logger.log(mode="val_tar", error=avg_loss_t, epoch=epoch + 1, n_batch=0, num_batches=1,
                            scalar='avg_loss')
            self.logger.log(mode="val_tar", error=avg_loss_class_t, epoch=epoch + 1, n_batch=0, num_batches=1,
                            scalar='class_loss')
            self.logger.log(mode="val_tar", error=avg_loss_dsne_t, epoch=epoch + 1, n_batch=0, num_batches=1,
                            scalar='dsne_loss')
            self.logger.log(mode="val_tar", error=accuracy_t, epoch=epoch + 1, n_batch=0, num_batches=1,
                            scalar='accuracy')
            rootLogger.info('[%d/%d] - Val loss_s: %.3f loss_class_s: %.3f loss_dsne_s: %.3f  acc_s: %.2f  loss_t: %.3f loss_class_t: %.3f loss_dsne_t: %.3f acc_t: %.2f  ' % (
            (epoch + 1), self.epochs, avg_loss_s, avg_loss_class_s, avg_loss_dsne_s, accuracy_s, avg_loss_t, avg_loss_class_t, avg_loss_dsne_t, accuracy_t))

        return avg_loss_t, accuracy_t

    def evaluate_val_data(self, val_loader, model_path):
        """
        Function to evaluate the results on trained model
        :param val_loader: data loader on which clustering is evaluated
        :param model_path: name with which pre-trained model is saved
        :return: None
        """

        try:
            rootLogger.info("Loading Saved Model")
            checkpoint = torch.load(model_path + '/dsne_models/arch1/' + self.model_dir + '/' + self.dataset + '/best/' + self.model_name + '.pt')
            self.classifier.load_state_dict(checkpoint)
            rootLogger.info("Saved Model successfully loaded")
        except:
            rootLogger.info("Model not found.")

        epoch_loss = 0.
        epoch_class_loss = 0.
        epoch_dsne_loss = 0.
        correct = 0
        total = 0

        classes = ('orig', 'f2f')

        class_correct = list(0. for i in range(2))
        class_total = list(0. for i in range(2))
        self.classifier.eval()
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

                embed_s, ys_hat, _, _, classifier_loss, dsne_loss, loss = self.update_weights(Xs=Xs, Ys=Ys, Xt=Xs, Yt=Ys)
                epoch_loss += loss
                epoch_class_loss += classifier_loss
                epoch_dsne_loss += dsne_loss

                _, y_hat = torch.max(ys_hat.data, 1)
                total += y_hat.size(0)
                correct += (y_hat == Ys).sum().item()

            avg_loss = epoch_loss / len(val_loader)
            avg_loss_class = epoch_class_loss / len(val_loader)
            avg_loss_dsne = epoch_dsne_loss / len(val_loader)

            # Calculate accuracy for current epoch
            accuracy = 100 * correct / total

            rootLogger.info(
                ' Val loss: %.3f loss_class: %.3f loss_dsne: %.3f  accuracy: %.2f  ' % (avg_loss, avg_loss_class, avg_loss_dsne, accuracy))

            for i in range(2):
                print('Accuracy of %5s : %2d %%' % (
                    classes[i], 100 * class_correct[i] / class_total[i]))

    def evaluate_val_data_vgg(self, val_loader, model_path):
        """
        Function to evaluate the results on trained model
        :param val_loader: data loader on which clustering is evaluated
        :param model_path: name with which pre-trained model is saved
        :return: None
        """

        try:
            rootLogger.info("Loading Saved Model")
            checkpoint = torch.load(model_path + '/dsne_models/arch1/' + self.model_dir + '/' + self.dataset + '/best/' + self.model_name + '.pt')
            num_ftrs = self.classifier.classifier[6].in_features
            self.classifier.classifier[6] = nn.Linear(num_ftrs, 2)  # Change the last layer output from 1000 to 2
            self.classifier.cuda()
            self.classifier.load_state_dict(checkpoint)
            rootLogger.info("Saved Model successfully loaded")
        except:
            rootLogger.info("Model not found.")

        epoch_loss = 0.
        epoch_class_loss = 0.
        epoch_dsne_loss = 0.
        correct = 0
        total = 0

        classes = ('orig', 'f2f')

        class_correct = list(0. for i in range(2))
        class_total = list(0. for i in range(2))
        self.classifier.eval()
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

                embed_s, ys_hat, _, _, classifier_loss, dsne_loss, loss = self.update_weights_vgg(Xs=Xs, Ys=Ys, Xt=Xs, Yt=Ys)
                epoch_loss += loss
                epoch_class_loss += classifier_loss
                epoch_dsne_loss += dsne_loss

                _, y_hat = torch.max(ys_hat.data, 1)
                total += y_hat.size(0)
                correct += (y_hat == Ys).sum().item()

            avg_loss = epoch_loss / len(val_loader)
            avg_loss_class = epoch_class_loss / len(val_loader)
            avg_loss_dsne = epoch_dsne_loss / len(val_loader)

            # Calculate accuracy for current epoch
            accuracy = 100 * correct / total

            rootLogger.info(
                ' Val loss: %.3f loss_class: %.3f loss_dsne: %.3f  accuracy: %.2f  ' % (avg_loss, avg_loss_class, avg_loss_dsne, accuracy))

            for i in range(2):
                print('Accuracy of %5s : %2d %%' % (
                    classes[i], 100 * class_correct[i] / class_total[i]))

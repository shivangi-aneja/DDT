import os
import time
import torch
from os import makedirs
from common.losses.center_loss import CenterLoss
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from common.logging.logger import rootLogger
from common.logging.tf_logger import Logger
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary
from mpl_toolkits.mplot3d import Axes3D
from sklearn import manifold

alpha_siamese_loss = 1.0
alpha_classification = 0.1
category_to_label = {0: 'Orig', 1: 'F2F', 2: 'DF', 3: 'FS'}
category_to_color = np.array(['#ff80ed', '#065535', '#133337', '#ffc0cb.'])


class Siamese(object):

    def __init__(self, siamese, model_name, siamese_loss, classification_loss, optim_kwargs, dataset, batch_size, mode='train',
                 patience=50, optim=None, epochs=10, tf_log_path=None, use_cuda=None, siamese_params=None, model_dir='siamese'):
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
        self.siamese = siamese
        self.mode = mode
        if siamese_params is None or not len(siamese_params):
            siamese_params = filter(lambda x: x.requires_grad, self.siamese.parameters())
            params = sum([np.prod(p.size()) for p in self.siamese.parameters()])
            rootLogger.info("Trainable Parameters Siamese: " + str(params))

        l2_param_list = [self.siamese.network[25], self.siamese.network[26], self.siamese.network[27],self.siamese.network[28],
                        self.siamese.network[29], self.siamese.network[30]]
        for module_name, network_module in self.siamese.named_children():
            for name, module in network_module.named_children():
                if module not in l2_param_list:
                    for param in module.parameters():
                        param.requires_grad = False
        total_params = sum(p.numel() for p in self.siamese.parameters())
        trainable_params = sum(p.numel() for p in self.siamese.parameters() if p.requires_grad)
        l2_params = (p for p in self.siamese.parameters() if p.requires_grad)
        rootLogger.info("Trainable Parameters L2: " + str(trainable_params))

        optim = optim or torch.optim.Adam
        optim_kwargs = optim_kwargs or {}
        optim_kwargs.setdefault('lr', 1e-3)
        self.center_loss = CenterLoss(num_classes=2, feat_dim=128, use_gpu=True)
        self.optim_siamese = optim(siamese_params, **optim_kwargs)
        self.optim_l2 = optim(l2_params, **optim_kwargs)
        self.epochs = epochs
        self.counter = 0
        self.patience = patience
        self.early_stop = False
        self.siamese_loss = siamese_loss
        self.classification_loss = classification_loss
        self.model_name = model_name
        self.dataset = dataset
        self.batch_size = batch_size
        self.use_cuda = use_cuda
        self.model_dir = model_dir
        self.scheduler_siamese = StepLR(self.optim_siamese, step_size=10, gamma=0.5)
        if self.use_cuda is None:
            self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.siamese.cuda()
        self.logger = Logger(model_name=self.model_name, data_name=self.dataset, log_path=tf_log_path)

    def train_siamese(self, train_loader, test_loader, model_path):
        """
        Function for training
        :param train_loader: Loader for training data
        :param test_loader: Loader for test data
        :param model_path: Path for saving the data
        """

        # results save folder
        model_current = model_path + '/siamese_models/' + self.model_dir + '/' + self.dataset + '/current/' + self.model_name + '.pt'
        model_best = model_path + '/siamese_models/' + self.model_dir + '/' + self.dataset + '/best/' + self.model_name + '.pt'

        try:
            rootLogger.info("Loading Saved Model")
            checkpoint = torch.load(model_current)
            self.siamese.load_state_dict(checkpoint)
            rootLogger.info("Saved Model successfully loaded")
        except:
            rootLogger.info("Model not found, Created a new one")
            # Make directory for Saving Models
            if not os.path.isdir(model_path + '/siamese_models/' + self.model_dir + '/' + self.dataset + '/current/'):
                makedirs(model_path + '/siamese_models/' + self.model_dir + '/' + self.dataset + '/current/')
            if not os.path.isdir(model_path + '/siamese_models/' + self.model_dir + '/' + self.dataset + '/best/'):
                makedirs(model_path + '/siamese_models/' + self.model_dir + '/' + self.dataset + '/best/')

        # training-loop
        np.random.seed(int(time.time()))
        rootLogger.info('Training Start!!!')
        best_loss = np.Inf

        for epoch in range(self.epochs):
            self.siamese.train()
            epoch_start_time = time.time()
            epoch_train_loss = 0.
            epoch_class_loss = 0.
            epoch_siamese_loss = 0.
            correct = 0
            total = 0

            # Checkpoint after 5 epochs
            if epoch % 10 == 0:
                try:
                    rootLogger.info("Saving the model")
                    torch.save(self.siamese.state_dict(), model_current)
                    rootLogger.info("Model Saved")
                except:
                    rootLogger.info("Can't save the model")

            for epoch_iter, data in enumerate(train_loader):

                X0, y0, X1, y1, X2, y2 = data
                # Move the images to the device first before computation
                if self.use_cuda:
                    X0, y0, X1, y1, X2, y2 = X0.cuda(), y0.cuda(), X1.cuda(), y1.cuda(), X2.cuda(), y2.cuda()
                X0, y0, X1, y1, X2, y2 = Variable(X0), Variable(y0), Variable(X1), Variable(y1), Variable(X2), Variable(y2)

                self.optim_siamese.zero_grad()  # clear gradients for this training step
                y_hat_0, embed_0, y_hat_1, embed_1, y_hat_2, embed_2 = self.siamese(X0, X1, X2)
                y_hat = torch.cat((y_hat_0, y_hat_1, y_hat_2), dim=0)
                y = torch.cat((y0, y1, y2), dim=0)
                embed = torch.cat((embed_0, embed_1, embed_2), dim=0)
                loss = alpha_siamese_loss * self.siamese_loss(embed_0, embed_1, embed_2) + alpha_classification * self.classification_loss(y_hat, y)
                epoch_class_loss += self.classification_loss(y_hat, y)
                epoch_siamese_loss += self.siamese_loss(embed_0, embed_1, embed_2)
                epoch_train_loss += loss.item()
                loss.backward()  # backpropagation, compute gradients
                self.optim_siamese.step()  # apply gradients
                _, predicted = torch.max(y_hat.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

            avg_loss = epoch_train_loss / len(train_loader)
            avg_class_loss = epoch_class_loss / len(train_loader)
            avg_sim_loss = epoch_siamese_loss / len(train_loader)

            # Calculate accuracy for current epoch
            accuracy = 100 * correct / total

            # Log the training losses and accuracy
            self.logger.log(mode="train", error=avg_loss, epoch=epoch + 1, n_batch=0, num_batches=1)
            self.logger.log(mode="train", error=avg_class_loss, epoch=epoch + 1, n_batch=0, num_batches=1,
                            scalar='class_loss')
            self.logger.log(mode="train", error=avg_sim_loss, epoch=epoch + 1, n_batch=0, num_batches=1,
                            scalar='sim_loss')
            self.logger.log(mode="train", error=accuracy, epoch=epoch + 1, n_batch=0, num_batches=1, scalar='acc')

            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - epoch_start_time
            rootLogger.info('[%d/%d] - ptime: %.2f Train: loss: %.3f  Class_loss: %.3f  Sim_loss: %.3f  Acc: %.2f' % (
                (epoch + 1), self.epochs, per_epoch_ptime, avg_loss, avg_class_loss, avg_sim_loss, accuracy))

            val_loss, val_acc = self.evaluate_val_data_tf(test_loader=test_loader, epoch=epoch)

            # Save the best model
            if val_loss <= best_loss:
                self.counter = 0
                self.siamese.train()
                best_loss = val_loss
                torch.save(self.siamese.state_dict(), model_best)
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
            self.scheduler_siamese.step()

    def evaluate_val_data_tf(self, test_loader, epoch):
        """
        Function to evaluate the results on trained model
        :param test_loader: data loader on which clustering is evaluated
        :param model_name: name with which pre-trained model is saved
        :param epoch:
        :return: None
        """
        self.siamese.eval()
        epoch_train_loss = 0.
        epoch_class_loss = 0.
        epoch_siamese_loss = 0.
        correct = 0
        total = 0
        with torch.no_grad():
            for epoch_iter, data in enumerate(test_loader):
                X0, y0, X1, y1, X2, y2 = data
                # Move the images to the device first before computation
                if self.use_cuda:
                    X0, y0, X1, y1, X2, y2 = X0.cuda(), y0.cuda(), X1.cuda(), y1.cuda(), X2.cuda(), y2.cuda()
                    X0, y0, X1, y1, X2, y2 = Variable(X0), Variable(y0), Variable(X1), Variable(y1), Variable(X2), Variable(y2)

                y_hat_0, embed_0, y_hat_1, embed_1, y_hat_2, embed_2 = self.siamese(X0, X1, X2)
                y_hat = torch.cat((y_hat_0, y_hat_1, y_hat_2), dim=0)
                y = torch.cat((y0, y1, y2), dim=0)

                loss = alpha_siamese_loss * self.siamese_loss(embed_0, embed_1,
                                                              embed_2) + alpha_classification * self.classification_loss(
                    y_hat, y)
                epoch_class_loss += self.classification_loss(y_hat, y)
                epoch_siamese_loss += self.siamese_loss(embed_0, embed_1, embed_2)
                epoch_train_loss += loss.item()
                _, predicted = torch.max(y_hat.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

            avg_loss = epoch_train_loss / len(test_loader)
            avg_class_loss = epoch_class_loss / len(test_loader)
            avg_sim_loss = epoch_siamese_loss / len(test_loader)

            # Calculate accuracy for current epoch
            accuracy = 100 * correct / total

            # Log the training losses and accuracy
            self.logger.log(mode="val", error=avg_loss, epoch=epoch + 1, n_batch=0, num_batches=1)
            self.logger.log(mode="val", error=avg_class_loss, epoch=epoch + 1, n_batch=0, num_batches=1,
                            scalar='class_loss')
            self.logger.log(mode="val", error=avg_sim_loss, epoch=epoch + 1, n_batch=0, num_batches=1,
                            scalar='sim_loss')
            self.logger.log(mode="val", error=accuracy, epoch=epoch + 1, n_batch=0, num_batches=1, scalar='acc')

            rootLogger.info('[%d/%d] Val loss: %.3f  Class_loss: %.3f  Sim_loss: %.3f  Acc: %.2f' %
                            ((epoch + 1), self.epochs, avg_loss, avg_class_loss, avg_sim_loss, accuracy))
            return avg_loss, accuracy

    # Validate the results on test loader
    def validate_results(self, test_loader, model_path):
        """
        Function to evaluate the result on test data
        :param val_loader:
        :param model_path:
        :return:
        """

        rootLogger.info("Validating ......")
        try:
            rootLogger.info("Loading Saved Model")
            checkpoint = torch.load(
                model_path + '/siamese_models/' + self.model_dir + '/' + self.dataset + '/best/' + self.model_name + '.pt')
            self.siamese.load_state_dict(checkpoint)
            rootLogger.info("Saved Model successfully loaded")
        except:
            rootLogger.info("Model not found.")
            exit()
        classes = ('orig', 'fake')

        class_correct = list(0. for i in range(2))
        class_total = list(0. for i in range(2))
        self.siamese.eval()
        epoch_loss = 0.
        epoch_class_loss = 0.
        epoch_siamese_loss = 0.
        correct = 0
        total = 0
        with torch.no_grad():
            for epoch_iter, data in enumerate(test_loader):
                X, y = data
                # Move the images to the device first before computation
                if self.use_cuda:
                    X, y = X.cuda(), y.cuda()
                X, y = Variable(X), Variable(y)

                y_hat, embed, _, _, _, _ = self.siamese(X, X, X)

                loss = alpha_siamese_loss * self.siamese_loss(embed, embed,
                                                              embed) + alpha_classification * self.classification_loss(y_hat, y)
                epoch_class_loss += self.classification_loss(y_hat, y)
                epoch_siamese_loss += self.siamese_loss(embed, embed, embed)
                epoch_loss += loss.item()
                _, predicted = torch.max(y_hat.data, 1)

                c = (predicted == y).squeeze()
                for i in range(y.shape[0]):
                    label = y[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

                total += y.size(0)
                correct += (predicted == y).sum().item()


            avg_loss = epoch_loss / len(test_loader)
            avg_class_loss = epoch_class_loss / len(test_loader)
            avg_sim_loss = epoch_siamese_loss / len(test_loader)

            # Calculate accuracy for current epoch
            accuracy = 100 * correct / total

            f = open("results.txt", "a+")
            f.write("Validation loss: [%.3f] Class_loss: [%.3f] Siamese_loss: [%.3f] acc: [%.3f]" % (avg_loss, avg_class_loss, avg_sim_loss, accuracy))
            f.write("\n")
            f.close()
            rootLogger.info("Validation loss: [%.3f] Class_loss: [%.3f] Siamese_loss: [%.3f] acc: [%.3f]" % (avg_loss, avg_class_loss, avg_sim_loss, accuracy))
            for i in range(2):
                print('Accuracy of %5s : %2d %%' % (
                    classes[i], 100 * class_correct[i] / class_total[i]))

    def validate_results_current(self, test_loader, model_path):
        """
        Function to evaluate the result on test data
        :param val_loader:
        :param model_path:
        :return:
        """

        rootLogger.info("Validating ......")
        try:
            rootLogger.info("Loading Saved Model")
            checkpoint = torch.load(
                model_path + '/siamese_models/' + self.model_dir + '/' + self.dataset + '/current/' + self.model_name + '.pt')
            self.siamese.load_state_dict(checkpoint)
            rootLogger.info("Saved Model successfully loaded")
        except:
            rootLogger.info("Model not found.")
            exit()
        classes = ('orig', 'fake')

        class_correct = list(0. for i in range(2))
        class_total = list(0. for i in range(2))
        self.siamese.eval()
        epoch_loss = 0.
        epoch_class_loss = 0.
        epoch_siamese_loss = 0.
        correct = 0
        total = 0
        with torch.no_grad():
            for epoch_iter, data in enumerate(test_loader):
                X, y = data
                # Move the images to the device first before computation
                if self.use_cuda:
                    X, y = X.cuda(), y.cuda()
                X, y = Variable(X), Variable(y)

                y_hat, embed, _, _, _, _ = self.siamese(X, X, X)

                loss = alpha_siamese_loss * self.siamese_loss(embed, embed,
                                                              embed) + alpha_classification * self.classification_loss(y_hat, y)
                epoch_class_loss += self.classification_loss(y_hat, y)
                epoch_siamese_loss += self.siamese_loss(embed, embed, embed)
                epoch_loss += loss.item()
                _, predicted = torch.max(y_hat.data, 1)

                c = (predicted == y).squeeze()
                for i in range(y.shape[0]):
                    label = y[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

                total += y.size(0)
                correct += (predicted == y).sum().item()


            avg_loss = epoch_loss / len(test_loader)
            avg_class_loss = epoch_class_loss / len(test_loader)
            avg_sim_loss = epoch_siamese_loss / len(test_loader)

            # Calculate accuracy for current epoch
            accuracy = 100 * correct / total

            f = open("results.txt", "a+")
            f.write("Validation loss: [%.3f] Class_loss: [%.3f] Siamese_loss: [%.3f] acc: [%.3f]" % (avg_loss, avg_class_loss, avg_sim_loss, accuracy))
            f.write("\n")
            f.close()
            rootLogger.info("Validation loss: [%.3f] Class_loss: [%.3f] Siamese_loss: [%.3f] acc: [%.3f]" % (avg_loss, avg_class_loss, avg_sim_loss, accuracy))
            for i in range(2):
                print('Accuracy of %5s : %2d %%' % (
                    classes[i], 100 * class_correct[i] / class_total[i]))

    def fine_tune_siamese(self, train_l2_loader, val_l2_loader, train_siamese_loader, val_siamese_loader, model_path):
        """
        Function for training
        :param train_loader: Loader for training data
        :param test_loader: Loader for test data
        :param model_path: Path for saving the data
        """

        # results save folder
        model_current = model_path + '/siamese_models/' + self.model_dir + '/' + self.dataset + '/current/' + self.model_name + '.pt'
        model_best = model_path + '/siamese_models/' + self.model_dir + '/' + self.dataset + '/best/' + self.model_name + '.pt'

        try:
            rootLogger.info("Loading Saved Model")
            checkpoint = torch.load(model_best)
            self.siamese.load_state_dict(checkpoint)
            rootLogger.info("Saved Model successfully loaded")
        except:
            rootLogger.info("Model not found")
            exit()

        # training-loop
        np.random.seed(int(time.time()))
        rootLogger.info('Training Start!!!')
        best_loss = np.Inf

        for epoch in range(self.epochs):

            self.siamese.train()
            epoch_start_time = time.time()

            # Checkpoint after 5 epochs
            if epoch % 10 == 0:
                try:
                    rootLogger.info("Saving the model")
                    torch.save(self.siamese.state_dict(), model_current)
                    rootLogger.info("Model Saved")
                except:
                    rootLogger.info("Can't save the model")

            # Train
            train_embed_loss = self.update_embeddings(l2_loader=train_l2_loader, mode='train')

            if epoch % 50 == 0:
                avg_loss, avg_class_loss, avg_sim_loss, train_accuracy = self.update_siamese(data_loader=train_siamese_loader, mode='train')
                epoch_end_time = time.time()
                per_epoch_ptime = epoch_end_time - epoch_start_time
                # Log the training losses and accuracy
                self.logger.log(mode="train", error=avg_loss, epoch=epoch + 1, n_batch=0, num_batches=1)
                self.logger.log(mode="train", error=avg_class_loss, epoch=epoch + 1, n_batch=0, num_batches=1,
                                scalar='class_loss')
                self.logger.log(mode="train", error=avg_sim_loss, epoch=epoch + 1, n_batch=0, num_batches=1,
                                scalar='sim_loss')
                self.logger.log(mode="train", error=train_accuracy, epoch=epoch + 1, n_batch=0, num_batches=1, scalar='acc')

                rootLogger.info('[%d/%d] - ptime: %.2f Train: loss: %.3f  Class_loss: %.3f  Sim_loss: %.3f Embed_loss: %.3f  Acc: %.2f' % (
                (epoch + 1), self.epochs, per_epoch_ptime, avg_loss, avg_class_loss, avg_sim_loss, train_embed_loss, train_accuracy))

            self.logger.log(mode="train", error=train_embed_loss, epoch=epoch + 1, n_batch=0, num_batches=1,
                            scalar='embed_loss')

            rootLogger.info('[%d/%d] - ptime: %.2f Train:  Embed_loss: %.3f ' % ((epoch + 1), self.epochs, per_epoch_ptime, train_embed_loss))

            self.siamese.eval()
            val_embed_loss = self.update_embeddings(l2_loader=val_l2_loader, mode='val')

            if epoch % 50 == 0:
                val_avg_loss, val_avg_class_loss, val_avg_sim_loss, val_accuracy = self.update_siamese(data_loader=val_siamese_loader, mode='val')
                rootLogger.info(' Val: loss: %.3f  Class_loss: %.3f  Sim_loss: %.3f Embed_loss: %.3f  Acc: %.2f' % (
                val_avg_loss, val_avg_class_loss, val_avg_sim_loss, val_embed_loss, val_accuracy))

                # Log the validation losses and accuracy
                self.logger.log(mode="val", error=val_avg_loss, epoch=epoch + 1, n_batch=0, num_batches=1)
                self.logger.log(mode="val", error=val_avg_class_loss, epoch=epoch + 1, n_batch=0, num_batches=1,
                                scalar='class_loss')
                self.logger.log(mode="val", error=val_avg_sim_loss, epoch=epoch + 1, n_batch=0, num_batches=1,
                                scalar='sim_loss')
                self.logger.log(mode="val", error=val_embed_loss, epoch=epoch + 1, n_batch=0, num_batches=1,
                                scalar='embed_loss')
                self.logger.log(mode="val", error=val_accuracy, epoch=epoch + 1, n_batch=0, num_batches=1, scalar='acc')

            self.logger.log(mode="val", error=val_embed_loss, epoch=epoch + 1, n_batch=0, num_batches=1, scalar='embed_loss')
            rootLogger.info('[%d/%d] - ptime: %.2f Val:  Embed_loss: %.3f ' % ((epoch + 1), self.epochs, per_epoch_ptime, val_embed_loss))

            # Save the best model
            if val_embed_loss <= best_loss:
                self.counter = 0
                self.siamese.train()
                best_loss = val_embed_loss
                torch.save(self.siamese.state_dict(), model_best)
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
            # self.scheduler_siamese.step()

    def update_embeddings(self, l2_loader, mode):

        epoch_loss = 0.
        for epoch_iter, data in enumerate(l2_loader):

            df_X, fs_X = data
            # Move the images to the device first before computation
            if self.use_cuda:
                df_X, fs_X = df_X.cuda(), fs_X.cuda()
            df_X, fs_X = Variable(df_X), Variable(fs_X)

            if mode=='train':
                self.optim_l2.zero_grad()  # clear gradients for this training step

            y_hat_df, embed_df, y_hat_fs, embed_fs, _, _ = self.siamese(df_X, fs_X, fs_X)
            loss = (embed_df - embed_fs).pow(2).sum(1).mean()
            epoch_loss += loss.item()

            if mode == 'train':
                loss.backward()  # backpropagation, compute gradients
                self.optim_l2.step()  # apply gradients

        avg_loss = epoch_loss / len(l2_loader)
        return avg_loss

    def update_siamese(self, data_loader, mode):

        epoch_loss = 0.
        epoch_class_loss = 0.
        epoch_siamese_loss = 0.
        correct = 0
        total = 0

        for epoch_iter, data in enumerate(data_loader):

            X0, y0, X1, y1, X2, y2 = data
            # Move the images to the device first before computation
            if self.use_cuda:
                X0, y0, X1, y1, X2, y2 = X0.cuda(), y0.cuda(), X1.cuda(), y1.cuda(), X2.cuda(), y2.cuda()
            X0, y0, X1, y1, X2, y2 = Variable(X0), Variable(y0), Variable(X1), Variable(y1), Variable(X2), Variable(y2)

            if mode == 'train':
                self.optim_siamese.zero_grad()  # clear gradients for this training step

            y_hat_0, embed_0, y_hat_1, embed_1, y_hat_2, embed_2 = self.siamese(X0, X1, X2)
            y_hat = torch.cat((y_hat_0, y_hat_1, y_hat_2), dim=0)
            y = torch.cat((y0, y1, y2), dim=0)
            loss = alpha_siamese_loss * self.siamese_loss(embed_0, embed_1, embed_2)
            epoch_class_loss += self.classification_loss(y_hat, y)
            epoch_siamese_loss += self.siamese_loss(embed_0, embed_1, embed_2)
            epoch_loss += loss.item()

            if mode == 'train':
                loss.backward()  # backpropagation, compute gradients
                self.optim_siamese.step()  # apply gradients
            _, predicted = torch.max(y_hat.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

        avg_loss = epoch_loss / len(data_loader)
        avg_class_loss = epoch_class_loss / len(data_loader)
        avg_sim_loss = epoch_siamese_loss / len(data_loader)

        # Calculate accuracy for current epoch
        accuracy = 100 * correct / total

        return avg_loss, avg_class_loss, avg_sim_loss, accuracy

    def visualize_latent_tsne(self, val_loader, model_path, file_name):
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
            print(model_path + '/siamese_models/' + self.model_dir + '/' + self.dataset + '/best/' + self.model_name + '.pt')
            checkpoint = torch.load(
                model_path + '/siamese_models/' + self.model_dir + '/' + self.dataset + '/best/' + self.model_name + '.pt')
            self.siamese.load_state_dict(checkpoint)
            rootLogger.info("Saved Model successfully loaded")
        except:
            rootLogger.info("Model not found.")
            exit()
        self.siamese.eval()
        latent_all, target_all = None, None
        with open(os.getcwd()+'/tsne_embeddings/' + file_name + '_target.tsv', 'w') as f:
            index = 0
            f.write("Index\tLabel\tClassLabel\n")
            with torch.no_grad():
                for epoch_iter, data in enumerate(val_loader):
                    X, y = data
                    # Move the images to the device first before computation
                    if self.use_cuda:
                        X, y = X.cuda(), y.cuda()
                    X, y = Variable(X), Variable(y)
                    y_hat, embed, _, _, _, _ = self.siamese(X, X, X)

                    if latent_all is None:
                        latent_all = embed
                        c_labels_all = y
                    else:
                        latent_all = torch.cat([latent_all, embed], dim=0)
                        c_labels_all = torch.cat([c_labels_all, y], dim=0)
                    for _, label in enumerate(y):
                        f.write("%d\t%d\t%d\n" % (index, label, label))
                        index += 1

            latent_all = latent_all.cpu().numpy()
            target_all = c_labels_all.cpu().numpy()
            target_all = target_all.astype(int)

        np.save(os.getcwd()+'/tsne_embeddings/' + file_name + '_latent', latent_all)

        rootLogger.info("T-SNE visualization started")
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
        x_tsne = tsne.fit_transform(latent_all)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        print(np.unique(target_all))
        for label_id in np.unique(target_all):
            ax.scatter(x_tsne[np.where(target_all == label_id), 0], x_tsne[np.where(target_all == label_id), 1],
                       marker='o', color=plt.cm.Set1(label_id), linewidth=1,
                       alpha=0.8, label=category_to_label[label_id])
        plt.legend(loc='best')
        fig.savefig(file_name + ".png", dpi=fig.dpi)
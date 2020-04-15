import os
import time
import torch
from os import makedirs
import numpy as np
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from common.logging.logger import rootLogger
from common.logging.tf_logger import Logger
from common.losses.custom_losses import ActivationLoss
from common.utils.common_utils import calc_activation_vector
from torchsummary import summary
from mpl_toolkits.mplot3d import Axes3D
from sklearn import manifold

alpha_recon_loss = 0.1
# alpha_center_loss = 2.0
alpha_act_loss = 1.0
category_to_label = {0: 'Orig', 1: 'F2F', 2: 'DF', 3: 'FS'}
category_to_color = np.array(['#ff80ed', '#065535', '#133337', '#ffc0cb.'])
orig_weight_factor = 1

class Autoencoder(object):

    def __init__(self, autoencoder, model_name, recon_loss, mode, optim_kwargs, dataset, num_classes, batch_size=64,
                 optim=None, epochs=10, tf_log_path=None, use_cuda=None, selection=False,
                 patience=50, counter=0, early_stop=False, model_dir='forensic_transfer'):
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
        # self.center_loss = CenterLoss(num_classes=2, feat_dim=64 * num_classes, use_gpu=True)
        autoencoder_params = None
        if mode == 'train':
            autoencoder_params = filter(lambda x: x.requires_grad, self.autoencoder.parameters())
            params = sum([np.prod(p.size()) for p in self.autoencoder.parameters()])
            rootLogger.info("Trainable Parameters : " + str(params))
        # elif mode == 'fine_tune':
        #     fine_tune_param_list = [self.autoencoder.encoder[16], self.autoencoder.encoder[17], self.autoencoder.encoder[20],self.autoencoder.encoder[21],
        #                             self.autoencoder.encoder[25], self.autoencoder.encoder[26],
        #                             self.autoencoder.decoder[0], self.autoencoder.decoder[1], self.autoencoder.decoder[4],
        #                             self.autoencoder.decoder[5],self.autoencoder.decoder[6], self.autoencoder.decoder[8],
        #                             self.autoencoder.decoder[9],  self.autoencoder.decoder[10]]
        #     for module_name, network_module in self.autoencoder.named_children():
        #         for name, module in network_module.named_children():
        #             if module not in fine_tune_param_list:
        #                 for param in module.parameters():
        #                     param.requires_grad = False
        #     total_params = sum(p.numel() for p in self.autoencoder.parameters())
        #     trainable_params = sum(p.numel() for p in self.autoencoder.parameters() if p.requires_grad)
        #     autoencoder_params = (p for p in self.autoencoder.parameters() if p.requires_grad)
        #     rootLogger.info("Total Parameters : " + str(total_params))
        #     rootLogger.info("Trainable Parameters : " + str(trainable_params))
        optim = optim or torch.optim.Adam
        optim_kwargs = optim_kwargs or {}
        optim_kwargs.setdefault('lr', 1e-4)
        self.optim = optim(autoencoder_params, **optim_kwargs)
        # self.center_loss_optim = optim(list(self.center_loss.parameters()), **optim_kwargs)
        self.epochs = epochs
        self.num_classes = num_classes
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
        self.scheduler = ReduceLROnPlateau(self.optim, mode='min', factor=0.7, patience=10, verbose=True)
        # self.center_loss_scheduler = StepLR(self.center_loss_optim, step_size=10, gamma=0.9)
        if self.use_cuda is None:
            self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.autoencoder.cuda()
        if tf_log_path is not None:
            self.logger = Logger(model_name=self.model_name, data_name=self.dataset, log_path=tf_log_path)

    def loss_function(self, latent_dim, z, y):
        act_vector = calc_activation_vector(latent_dim, z)
        orig_indices = (y == 0).nonzero().squeeze(1)
        weight = torch.ones(y.size()[0])
        weight.put_(orig_indices.cpu(), torch.Tensor(len(orig_indices)).fill_(orig_weight_factor)).cuda()
        y_onehot = torch.FloatTensor(y.size()[0], 2).cuda()
        y_onehot.zero_()
        y_onehot.scatter_(1, y.view(-1, 1), 1)
        act_loss = ActivationLoss(act_vector, y_onehot, weight)
        return act_loss


    def train_autoencoder(self, train_loader, test_loader, model_path):
        """
        Function for training
        :param train_loader: Loader for training data
        :param test_loader: Loader for test data
        :param model_path: Path for saving the data
        """

        # results save folder
        model_current = model_path + '/ft_models/' + self.model_dir + '/' + self.dataset + '/current/' + self.model_name + '.pt'
        model_best = model_path + '/ft_models/' + self.model_dir + '/' + self.dataset + '/best/' + self.model_name + '.pt'
        # loss_params = model_path + '/ft_models/' + self.model_dir + '/' + self.dataset + '/best/' + self.model_name + '_loss_params.pt'
        try:
            rootLogger.info("Loading Saved Model")
            checkpoint = torch.load(model_current)
            self.autoencoder.load_state_dict(checkpoint)
            rootLogger.info("Saved Model successfully loaded")
        except:
            rootLogger.info("Model not found, Created a new one")
            # Make directory for Saving Models
            if not os.path.isdir(model_path + 'ft_models/' + self.model_dir + '/' + self.dataset + '/current/'):
                makedirs(model_path + 'ft_models/' + self.model_dir + '/' + self.dataset + '/current/')
            if not os.path.isdir(model_path + 'ft_models/' + self.model_dir + '/' + self.dataset + '/best/'):
                makedirs(model_path + 'ft_models/' + self.model_dir + '/' + self.dataset + '/best/')

        # training-loop
        np.random.seed(int(time.time()))
        rootLogger.info('Training Start!!!')
        best_loss = np.Inf

        for epoch in range(self.epochs):
            self.autoencoder.train()
            epoch_start_time = time.time()
            epoch_train_loss = 0.
            epoch_recon_loss = 0.
            epoch_act_loss = 0.
            # epoch_center_loss = 0.
            correct = 0
            total = 0
            # Checkpoint after 5 epochs
            if (epoch + 1) % 5 == 0:
                try:
                    rootLogger.info("Saving the model after 5 epochs")
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
                # self.center_loss_optim.zero_grad()
                latent_dim = 128
                with torch.set_grad_enabled(True):
                    z_unmasked, pred_image = self.autoencoder(input_image, target, self.selection)
                    act_loss = self.loss_function(latent_dim, z_unmasked, target)
                    act_vector = calc_activation_vector(latent_dim, z_unmasked)

                    # if self.num_classes == 2:
                    #     # Calculate Activation Loss
                    #     with torch.no_grad():
                    #         latent_length = z_unmasked.size()[-1]  # 128
                    #         latent_half = latent_length // 2  # 64
                    #
                    #     latent_first_half = torch.mean(torch.abs(z_unmasked[:, :latent_half]), dim=1)
                    #     latent_second_half = torch.mean(torch.abs(z_unmasked[:, latent_half:]), dim=1)
                    #     activation_vector = torch.stack((latent_first_half, latent_second_half), dim=1)
                    #     y_onehot = torch.FloatTensor(target.size()[0], 2).cuda()
                    # else:
                    #     # Calculate Activation Loss
                    #     with torch.no_grad():
                    #         latent_length = z_unmasked.size()[-1]  # 192
                    #         latent_each_class = latent_length // self.num_classes  # 64
                    #
                    #     latent_orig = torch.mean(torch.abs(z_unmasked[:, :latent_each_class]), dim=1)
                    #     latent_f2f = torch.mean(torch.abs(z_unmasked[:, latent_each_class:2 * latent_each_class]), dim=1)
                    #     latent_df = torch.mean(torch.abs(z_unmasked[:, 2 * latent_each_class:]), dim=1)
                    #     activation_vector = torch.stack((latent_orig, latent_f2f, latent_df), dim=1)
                    #     y_onehot = torch.FloatTensor(target.size()[0], 3).cuda()
                    #
                    # y_onehot.zero_()
                    # y_onehot.scatter_(1, target.view(-1, 1), 1)
                    # target_center = target.clone().detach()
                    # target_center[target_center == 2] = 1
                    loss = alpha_recon_loss * self.recon_loss(pred_image, input_image) + alpha_act_loss * act_loss
                           # + alpha_center_loss * self.center_loss(z_unmasked, target_center)
                    epoch_train_loss += loss.item()
                    epoch_recon_loss += self.recon_loss(pred_image, input_image).item()
                    epoch_act_loss += act_loss.item()
                    # epoch_center_loss += self.center_loss(z_unmasked, target_center).item()
                    loss.backward()  # backpropagation, compute gradients
                    self.optim.step()  # apply gradients
                    # self.center_loss_optim.step()

                # Calculate correct predictions
                total += target.size(0)
                _, predicted = torch.max(act_vector, 1)
                correct += (predicted == target).sum().item()

            avg_loss = epoch_train_loss / len(train_loader)
            avg_recon_loss = epoch_recon_loss / len(train_loader)
            avg_act_loss = epoch_act_loss / len(train_loader)
            # avg_center_loss = epoch_center_loss / len(train_loader)
            accuracy = 100. * correct / total

            # Log the training losses and accuracy
            self.logger.log(mode="train", error=avg_loss, epoch=epoch + 1, n_batch=0, num_batches=1)
            self.logger.log(mode="train", error=avg_recon_loss, epoch=epoch + 1, n_batch=0, num_batches=1,
                            scalar='recon_loss')
            self.logger.log(mode="train", error=avg_act_loss, epoch=epoch + 1, n_batch=0, num_batches=1,
                            scalar='act_loss')
            # self.logger.log(mode="train", error=avg_center_loss, epoch=epoch + 1, n_batch=0, num_batches=1,
            #                 scalar='center_loss')
            self.logger.log(mode="train", error=accuracy, epoch=epoch + 1, n_batch=0, num_batches=1, scalar='acc')

            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - epoch_start_time
            rootLogger.info('[%d/%d] - ptime: %.2f Train : Total loss: %.3f, Recon loss: %.3f, Act loss: %.3f,  acc: %.3f' % (
            (epoch + 1), self.epochs, per_epoch_ptime, avg_loss, avg_recon_loss, avg_act_loss, accuracy))

            # Validation
            val_loss, val_acc = self.evaluate_val_data_tf(test_loader=test_loader, epoch=epoch)

            # Save the best model
            if val_loss <= best_loss:
                self.counter = 0
                self.autoencoder.train()
                best_loss = val_loss
                torch.save(self.autoencoder.state_dict(), model_best)
                # torch.save(self.center_loss.centers, loss_params)
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
            self.scheduler.step(val_loss)
            # self.center_loss_scheduler.step()

    def fine_tune_autoencoder(self, train_loader, test_loader, model_path):
        """
        Function for training
        :param train_loader: Loader for training data
        :param test_loader: Loader for test data
        :param model_path: Path for saving the data
        """

        # results save folder
        model_current = model_path + '/ft_models/' + self.model_dir + '/' + self.dataset + '/current/' + self.model_name + '.pt'
        model_best = model_path + '/ft_models/' + self.model_dir + '/' + self.dataset + '/best/' + self.model_name + '.pt'
        loss_params = model_path + '/ft_models/' + self.model_dir + '/' + self.dataset + '/best/' + self.model_name + '_loss_params.pt'
        try:
            rootLogger.info("Loading Saved Model")
            checkpoint = torch.load(model_current)
            self.autoencoder.load_state_dict(checkpoint)
            rootLogger.info("Saved Model successfully loaded")
        except:
            rootLogger.info("Model not found, Created a new one")
            # Make directory for Saving Models
            if not os.path.isdir(model_path + 'ft_models/' + self.model_dir + '/' + self.dataset + '/current/'):
                makedirs(model_path + 'ft_models/' + self.model_dir + '/' + self.dataset + '/current/')
            if not os.path.isdir(model_path + 'ft_models/' + self.model_dir + '/' + self.dataset + '/best/'):
                makedirs(model_path + 'ft_models/' + self.model_dir + '/' + self.dataset + '/best/')

        # training-loop
        np.random.seed(int(time.time()))
        rootLogger.info('Training Start!!!')
        best_loss = np.Inf
        self.center_loss.centers = torch.load(loss_params)
        for epoch in range(self.epochs):
            self.autoencoder.train()
            epoch_start_time = time.time()
            epoch_train_loss = 0.
            epoch_recon_loss = 0.
            epoch_act_loss = 0.
            epoch_center_loss = 0.
            correct = 0
            total = 0
            # Checkpoint after 5 epochs
            if (epoch + 1) % 5 == 0:
                try:
                    rootLogger.info("Saving the model after 5 epochs")
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
                self.center_loss_optim.zero_grad()

                with torch.set_grad_enabled(True):
                    z_unmasked, pred_image = self.autoencoder(input_image, target, self.selection)

                    if self.num_classes == 2:
                        # Calculate Activation Loss
                        with torch.no_grad():
                            latent_length = z_unmasked.size()[-1]  # 128
                            latent_half = latent_length // 2  # 64

                        latent_first_half = torch.mean(torch.abs(z_unmasked[:, :latent_half]), dim=1)
                        latent_second_half = torch.mean(torch.abs(z_unmasked[:, latent_half:]), dim=1)
                        activation_vector = torch.stack((latent_first_half, latent_second_half), dim=1)
                        y_onehot = torch.FloatTensor(target.size()[0], 2).cuda()
                        y_onehot.zero_()
                        y_onehot.scatter_(1, target.view(-1, 1), 1)

                    else:
                        # Calculate Activation Loss
                        with torch.no_grad():
                            latent_length = z_unmasked.size()[-1]  # 192
                            latent_each_class = latent_length // self.num_classes  # 64

                        latent_orig = torch.mean(torch.abs(z_unmasked[:, :latent_each_class]), dim=1)
                        fake_max, _ = torch.topk(torch.abs(z_unmasked[:, latent_each_class:]), latent_each_class, dim=1)
                        latent_fake = torch.mean(fake_max, dim=1)
                        activation_vector = torch.stack((latent_orig, latent_fake), dim=1)
                        # 3 because we are dealing with 3 classes here
                        y_onehot = torch.FloatTensor(target.size()[0], 2).cuda()
                        y_onehot.zero_()
                        y_onehot.scatter_(1, target.view(-1, 1), 1)
                    loss = alpha_recon_loss * self.recon_loss(pred_image,
                                                              input_image) + alpha_act_loss * ActivationLoss(
                        activation_vector, y_onehot) + alpha_center_loss * self.center_loss(z_unmasked, target)
                    epoch_train_loss += loss.item()
                    epoch_recon_loss += self.recon_loss(pred_image, input_image).item()
                    epoch_act_loss += ActivationLoss(activation_vector, y_onehot).item()
                    epoch_center_loss += self.center_loss(z_unmasked, target).item()

                    loss.backward()  # backpropagation, compute gradients
                    self.optim.step()  # apply gradients
                    self.center_loss_optim.step()

                # Calculate correct predictions
                total += target.size(0)
                _, predicted = torch.max(activation_vector, 1)
                predicted[predicted == 2] = 1
                correct += (predicted == target).sum().item()

            avg_loss = epoch_train_loss / len(train_loader)
            avg_recon_loss = epoch_recon_loss / len(train_loader)
            avg_act_loss = epoch_act_loss / len(train_loader)
            avg_center_loss = epoch_center_loss / len(train_loader)
            accuracy = 100. * correct / total

            # Log the training losses and accuracy
            self.logger.log(mode="train", error=avg_loss, epoch=epoch + 1, n_batch=0, num_batches=1)
            self.logger.log(mode="train", error=avg_recon_loss, epoch=epoch + 1, n_batch=0, num_batches=1,
                            scalar='recon_loss')
            self.logger.log(mode="train", error=avg_act_loss, epoch=epoch + 1, n_batch=0, num_batches=1,
                            scalar='act_loss')
            self.logger.log(mode="train", error=avg_center_loss, epoch=epoch + 1, n_batch=0, num_batches=1,
                            scalar='center_loss')
            self.logger.log(mode="train", error=accuracy, epoch=epoch + 1, n_batch=0, num_batches=1, scalar='acc')

            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - epoch_start_time
            rootLogger.info(
                '[%d/%d] - ptime: %.2f Train : Total loss: %.3f, Recon loss: %.3f, Act loss: %.3f, Center loss: %.3f, acc: %.3f' % (
                    (epoch + 1), self.epochs, per_epoch_ptime, avg_loss, avg_recon_loss, avg_act_loss, avg_center_loss,
                    accuracy))

            # Validation
            val_loss, val_acc = self.evaluate_val_data_finetune(test_loader=test_loader, epoch=epoch)

            # Save the best model
            if val_loss <= best_loss:
                self.counter = 0
                self.autoencoder.train()
                best_loss = val_loss
                torch.save(self.autoencoder.state_dict(), model_best)
                torch.save(self.center_loss.centers, loss_params)
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
            self.center_loss_scheduler.step()

    # For evaluation during training
    def evaluate_val_data_tf(self, test_loader, epoch):
        """
        Function to evaluate the results on trained model
        :param test_loader: data loader on which clustering is evaluated
        :param model_name: name with which pre-trained model is saved
        :param epoch:
        :return: None
        """
        self.autoencoder.eval()
        epoch_val_loss = 0
        epoch_recon_loss = 0.
        epoch_act_loss = 0.
        # epoch_center_loss = 0.
        correct = 0
        total = 0
        latent_dim = 128
        with torch.no_grad():
            for epoch_iter, data in enumerate(test_loader):
                input_image, target = data
                if self.use_cuda:
                    input_image = input_image.cuda()
                    target = target.cuda()

                z_unmasked, pred_image = self.autoencoder(input_image, target, False)
                act_loss = self.loss_function(latent_dim, z_unmasked, target)
                act_vector = calc_activation_vector(latent_dim, z_unmasked)
                # if self.num_classes == 2:
                #     # Calculate Activation Loss
                #     with torch.no_grad():
                #         latent_length = z_unmasked.size()[-1]  # 128
                #         latent_half = latent_length // 2  # 64
                #
                #     latent_first_half = torch.mean(torch.abs(z_unmasked[:, :latent_half]), dim=1)
                #     latent_second_half = torch.mean(torch.abs(z_unmasked[:, latent_half:]), dim=1)
                #     activation_vector = torch.stack((latent_first_half, latent_second_half), dim=1)
                #     y_onehot = torch.FloatTensor(target.size()[0], 2).cuda()
                # else:
                #     # Calculate Activation Loss
                #     with torch.no_grad():
                #         latent_length = z_unmasked.size()[-1]  # 192
                #         latent_each_class = latent_length // self.num_classes  # 64
                #
                #     latent_orig = torch.mean(torch.abs(z_unmasked[:, :latent_each_class]), dim=1)
                #     latent_f2f = torch.mean(torch.abs(z_unmasked[:, latent_each_class:2 * latent_each_class]), dim=1)
                #     latent_df = torch.mean(torch.abs(z_unmasked[:, 2 * latent_each_class:]), dim=1)
                #     activation_vector = torch.stack((latent_orig, latent_f2f, latent_df), dim=1)
                #     y_onehot = torch.FloatTensor(target.size()[0], 3).cuda()
                # y_onehot.zero_()
                # y_onehot.scatter_(1, target.view(-1, 1), 1)
                # target_center = target.clone().detach()
                # target_center[target_center == 2] = 1
                loss = alpha_recon_loss * self.recon_loss(pred_image, input_image) + alpha_act_loss * act_loss
                epoch_val_loss += loss.item()
                epoch_recon_loss += self.recon_loss(pred_image, input_image).item()
                epoch_act_loss += act_loss.item()
                # epoch_center_loss += self.center_loss(z_unmasked, target_center)

                # Calculate correct predictions
                total += target.size(0)
                _, predicted = torch.max(act_vector, 1)
                correct += (predicted == target).sum().item()

                if epoch_iter == len(test_loader) - 1:
                    self.logger.log_images(mode='predicted', images=pred_image, num_images=len(pred_image),
                                           epoch=epoch, n_batch=1, num_batches=len(test_loader), normalize=True)

                    self.logger.log_images(mode='ground_truth', images=input_image, num_images=len(input_image),
                                           epoch=epoch, n_batch=1, num_batches=len(test_loader), normalize=True)

            avg_loss = epoch_val_loss / len(test_loader)
            avg_recon_loss = epoch_recon_loss / len(test_loader)
            avg_act_loss = epoch_act_loss / len(test_loader)
            # avg_center_loss = epoch_center_loss / len(test_loader)
            accuracy = 100. * correct / total

            rootLogger.info("Validation Total loss= [%.3f], Recon loss: %.3f, Act loss: %.3f, Val acc= [%.3f]" %
                            (avg_loss, avg_recon_loss, avg_act_loss, accuracy))
            self.logger.log(mode="val", error=avg_loss, epoch=epoch, n_batch=0, num_batches=1)
            self.logger.log(mode="val", error=avg_recon_loss, epoch=epoch + 1, n_batch=0, num_batches=1,
                            scalar='recon_loss')
            self.logger.log(mode="val", error=avg_act_loss, epoch=epoch + 1, n_batch=0, num_batches=1,
                            scalar='act_loss')
            # self.logger.log(mode="val", error=avg_center_loss, epoch=epoch + 1, n_batch=0, num_batches=1,
            #                 scalar='center_loss')
            self.logger.log(mode="val", error=accuracy, epoch=epoch, n_batch=0, num_batches=1, scalar='acc')
            return avg_loss, accuracy

    def evaluate_val_data_finetune(self, test_loader, epoch):
        """
        Function to evaluate the results on trained model
        :param test_loader: data loader on which clustering is evaluated
        :param model_name: name with which pre-trained model is saved
        :param epoch:
        :return: None
        """
        self.autoencoder.eval()
        epoch_val_loss = 0
        epoch_recon_loss = 0.
        epoch_act_loss = 0.
        epoch_center_loss = 0.
        correct = 0
        total = 0
        with torch.no_grad():
            for epoch_iter, data in enumerate(test_loader):
                input_image, target = data
                if self.use_cuda:
                    input_image = input_image.cuda()
                    target = target.cuda()

                z_unmasked, pred_image = self.autoencoder(input_image, target, False)

                if self.num_classes == 2:
                    # Calculate Activation Loss
                    with torch.no_grad():
                        latent_length = z_unmasked.size()[-1]  # 128
                        latent_half = latent_length // 2  # 64

                    latent_first_half = torch.mean(torch.abs(z_unmasked[:, :latent_half]), dim=1)
                    latent_second_half = torch.mean(torch.abs(z_unmasked[:, latent_half:]), dim=1)
                    activation_vector = torch.stack((latent_first_half, latent_second_half), dim=1)
                    y_onehot = torch.FloatTensor(target.size()[0], 2).cuda()
                    y_onehot.zero_()
                    y_onehot.scatter_(1, target.view(-1, 1), 1)


                else:
                    # Calculate Activation Loss
                    with torch.no_grad():
                        latent_length = z_unmasked.size()[-1]  # 192
                        latent_each_class = latent_length // self.num_classes  # 64

                    latent_orig = torch.mean(torch.abs(z_unmasked[:, :latent_each_class]), dim=1)
                    fake_max, _ = torch.topk(torch.abs(z_unmasked[:, latent_each_class:]), latent_each_class, dim=1)
                    latent_fake = torch.mean(fake_max, dim=1)
                    activation_vector = torch.stack((latent_orig, latent_fake), dim=1)
                    # 3 because we are dealing with 3 classes here
                    y_onehot = torch.FloatTensor(target.size()[0], 2).cuda()
                    y_onehot.zero_()
                    y_onehot.scatter_(1, target.view(-1, 1), 1)
                loss = alpha_recon_loss * self.recon_loss(pred_image, input_image) + alpha_act_loss * ActivationLoss(
                    activation_vector, y_onehot) + alpha_center_loss * self.center_loss(z_unmasked, target)
                epoch_val_loss += loss.item()
                epoch_recon_loss += self.recon_loss(pred_image, input_image).item()
                epoch_act_loss += ActivationLoss(activation_vector, y_onehot).item()
                epoch_center_loss += self.center_loss(z_unmasked, target).item()

                # Calculate correct predictions
                total += target.size(0)
                _, predicted = torch.max(activation_vector, 1)
                predicted[predicted == 2] = 1
                correct += (predicted == target).sum().item()

                if epoch_iter == len(test_loader) - 1:
                    self.logger.log_images(mode='predicted', images=pred_image, num_images=len(pred_image),
                                           epoch=epoch, n_batch=1, num_batches=len(test_loader), normalize=True)

                    self.logger.log_images(mode='ground_truth', images=input_image, num_images=len(input_image),
                                           epoch=epoch, n_batch=1, num_batches=len(test_loader), normalize=True)

            avg_loss = epoch_val_loss / len(test_loader)
            avg_recon_loss = epoch_recon_loss / len(test_loader)
            avg_act_loss = epoch_act_loss / len(test_loader)
            avg_center_loss = epoch_center_loss / len(test_loader)
            accuracy = 100. * correct / total

            rootLogger.info(
                "Validation Total loss= [%.3f], Recon loss: %.3f, Act loss: %.3f, Center loss: %.3f, Val acc= [%.3f]" %
                (avg_loss, avg_recon_loss, avg_act_loss, avg_center_loss, accuracy))
            self.logger.log(mode="val", error=avg_loss, epoch=epoch, n_batch=0, num_batches=1)
            self.logger.log(mode="val", error=avg_recon_loss, epoch=epoch + 1, n_batch=0, num_batches=1,
                            scalar='recon_loss')
            self.logger.log(mode="val", error=avg_act_loss, epoch=epoch + 1, n_batch=0, num_batches=1,
                            scalar='act_loss')
            self.logger.log(mode="val", error=avg_center_loss, epoch=epoch + 1, n_batch=0, num_batches=1,
                            scalar='center_loss')
            self.logger.log(mode="val", error=accuracy, epoch=epoch, n_batch=0, num_batches=1, scalar='acc')
            return avg_loss, accuracy

    # For the 2-class (F2F + Orig) case, and trained classes for 3-class (F2F + DF + Orig) case
    def validate_results(self, val_loader, model_path):
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
                model_path + '/ft_models/' + self.model_dir + '/' + self.dataset + '/best/' + self.model_name + '.pt')
            loss_params = model_path + '/ft_models/' + self.model_dir + '/' + self.dataset + '/best/' + self.model_name + '_loss_params.pt'
            self.autoencoder.load_state_dict(checkpoint)
            self.center_loss.centers = torch.load(loss_params)
            rootLogger.info("Saved Model successfully loaded")
        except:
            rootLogger.info("Model not found.")

        self.autoencoder.eval()
        val_loss = 0
        recon_loss = 0.
        act_loss = 0.
        center_loss = 0.
        correct = 0
        total = 0
        with torch.no_grad():
            for epoch_iter, data in enumerate(val_loader):
                input_image, target = data
                if self.use_cuda:
                    input_image = input_image.cuda()
                    target = target.cuda()

                z_unmasked, pred_image = self.autoencoder(input_image, target, False)
                if self.num_classes == 2:
                    # Calculate Activation Loss
                    with torch.no_grad():
                        latent_length = z_unmasked.size()[-1]  # 128
                        latent_half = latent_length // 2  # 64

                    latent_first_half = torch.mean(torch.abs(z_unmasked[:, :latent_half]), dim=1)
                    latent_second_half = torch.mean(torch.abs(z_unmasked[:, latent_half:]), dim=1)
                    activation_vector = torch.stack((latent_first_half, latent_second_half), dim=1)
                    y_onehot = torch.FloatTensor(target.size()[0], 2).cuda()
                    y_onehot.zero_()
                    y_onehot.scatter_(1, target.view(-1, 1), 1)

                else:
                    # Calculate Activation Loss
                    with torch.no_grad():
                        latent_length = z_unmasked.size()[-1]  # 192
                        latent_each_class = latent_length // self.num_classes  # 64

                    latent_orig = torch.mean(torch.abs(z_unmasked[:, :latent_each_class]), dim=1)
                    latent_f2f = torch.mean(torch.abs(z_unmasked[:, latent_each_class:2 * latent_each_class]),
                                            dim=1)
                    latent_df = torch.mean(torch.abs(z_unmasked[:, 2 * latent_each_class:]), dim=1)
                    activation_vector = torch.stack((latent_orig, latent_f2f, latent_df), dim=1)
                    # 3 because we are dealing with 3 classes here
                    y_onehot = torch.FloatTensor(target.size()[0], 3).cuda()
                    y_onehot.zero_()
                    y_onehot.scatter_(1, target.view(-1, 1), 1)
                loss = alpha_recon_loss * self.recon_loss(pred_image, input_image) + alpha_act_loss * ActivationLoss(activation_vector, y_onehot) \
                       + alpha_center_loss * self.center_loss(z_unmasked, target)
                val_loss += loss.item()
                recon_loss += self.recon_loss(pred_image, input_image).item()
                act_loss += ActivationLoss(activation_vector, y_onehot).item()
                center_loss += self.center_loss(z_unmasked, target).item()

                # Calculate correct predictions
                total += target.size(0)
                _, predicted = torch.max(activation_vector, 1)
                correct += (predicted == target).sum().item()

            avg_loss = val_loss / len(val_loader)
            avg_recon_loss = recon_loss / len(val_loader)
            avg_act_loss = act_loss / len(val_loader)
            avg_center_loss = center_loss / len(val_loader)
            accuracy = 100. * correct / total

            f = open("results.txt", "a+")
            f.write("Validation Total loss= [%.3f], Recon loss: %.3f, Act loss: %.3f, Center loss: %.3f, Val acc= [%.3f]" %
                    (avg_loss, avg_recon_loss, avg_act_loss, avg_center_loss, accuracy))
            f.write("\n")
            f.close()
            rootLogger.info("Validation Total loss= [%.3f], Recon loss: %.3f, Act loss: %.3f, Center loss: %.3f, Val acc= [%.3f]" %
                            (avg_loss, avg_recon_loss, avg_act_loss, avg_center_loss, accuracy))

    def validate_results_finetune(self, val_loader, model_path):
        """
        Function to evaluate the result on test data
        :param val_loader:
        :param model_path:
        :return:
        """

        rootLogger.info("Validating ......")
        loss_params = model_path + '/ft_models/' + self.model_dir + '/' + self.dataset + '/best/' + self.model_name + '_loss_params.pt'
        try:
            rootLogger.info("Loading Saved Model")
            checkpoint = torch.load(
                model_path + '/ft_models/' + self.model_dir + '/' + self.dataset + '/best/' + self.model_name + '.pt')
            self.center_loss.centers = torch.load(loss_params)
            self.autoencoder.load_state_dict(checkpoint)
            rootLogger.info("Saved Model successfully loaded")
        except:
            rootLogger.info("Model not found.")

        self.autoencoder.eval()
        val_loss = 0
        recon_loss = 0.
        act_loss = 0.
        center_loss = 0.
        correct = 0
        total = 0
        with torch.no_grad():
            for epoch_iter, data in enumerate(val_loader):
                input_image, target = data
                if self.use_cuda:
                    input_image = input_image.cuda()
                    target = target.cuda()
                    target[target == 3] = 1  # FS = 3, changed to fake(1)
                    target[target == 2] = 1
                z_unmasked, pred_image = self.autoencoder(input_image, target, False)

                # Calculate Activation Loss
                with torch.no_grad():
                    latent_length = z_unmasked.size()[-1]  # 192
                    latent_each_class = latent_length // self.num_classes  # 64

                latent_orig = torch.mean(torch.abs(z_unmasked[:, :latent_each_class]), dim=1)
                fake_max, _ = torch.topk(torch.abs(z_unmasked[:, latent_each_class:]), latent_each_class, dim=1)
                latent_fake = torch.mean(fake_max, dim=1)
                activation_vector = torch.stack((latent_orig, latent_fake), dim=1)
                # 3 because we are dealing with 3 classes here
                y_onehot = torch.FloatTensor(target.size()[0], 2).cuda()
                y_onehot.zero_()
                y_onehot.scatter_(1, target.view(-1, 1), 1)
                loss = alpha_recon_loss * self.recon_loss(pred_image,
                                                      input_image) + alpha_act_loss * ActivationLoss(
                activation_vector, y_onehot) + alpha_center_loss * self.center_loss(z_unmasked, target)
                val_loss += loss.item()
                recon_loss += self.recon_loss(pred_image, input_image).item()
                act_loss += ActivationLoss(activation_vector, y_onehot).item()
                center_loss += self.center_loss(z_unmasked, target).item()

                # Calculate correct predictions
                total += target.size(0)
                _, predicted = torch.max(activation_vector, 1)
                predicted[predicted == 2] = 1  # Prediction 1(F2F) and 2 (DF) should be counted as fake (1)
                correct += (predicted == target).sum().item()

                #correct += self.center_loss.classify(z_unmasked, target)
            avg_loss = val_loss / len(val_loader)
            avg_recon_loss = recon_loss / len(val_loader)
            avg_act_loss = act_loss / len(val_loader)
            avg_center_loss = center_loss / len(val_loader)
            accuracy = 100. * correct / total

            f = open("results.txt", "a+")
            f.write(
                "Validation Total loss= [%.3f], Recon loss: %.3f, Act loss: %.3f, Center loss: %.3f, Val acc= [%.3f]" %
                (avg_loss, avg_recon_loss, avg_act_loss, avg_center_loss, accuracy))
            f.write("\n")
            f.close()
            rootLogger.info(
                "Validation Total loss= [%.3f], Recon loss: %.3f, Act loss: %.3f, Center loss: %.3f, Val acc= [%.3f]" %
                (avg_loss, avg_recon_loss, avg_act_loss, avg_center_loss, accuracy))

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
            print(model_path + '/ft_models/' + self.model_dir + '/' + self.dataset + '/best/' + self.model_name + '.pt')
            checkpoint = torch.load(
                model_path + '/ft_models/' + self.model_dir + '/' + self.dataset + '/best/' + self.model_name + '.pt')
            self.autoencoder.load_state_dict(checkpoint)
            rootLogger.info("Saved Model successfully loaded")
        except:
            rootLogger.info("Model not found.")
        self.autoencoder.eval()
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
                else:
                    latent_all = torch.cat([latent_all, latent], dim=0)
                    target_all = torch.cat([target_all, target], dim=0)

        latent_all = latent_all.cpu().numpy()
        target_all = target_all.cpu().numpy()
        target_all = target_all.astype(int)
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

    def visualize_actiavtion_tsne(self, val_loader, model_path, file_name):
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
            print(model_path + '/ft_models/' + self.model_dir + '/' + self.dataset + '/best/' + self.model_name + '.pt')
            checkpoint = torch.load(
                model_path + '/ft_models/' + self.model_dir + '/' + self.dataset + '/best/' + self.model_name + '.pt')
            self.autoencoder.load_state_dict(checkpoint)
            rootLogger.info("Saved Model successfully loaded")
        except:
            rootLogger.info("Model not found.")
        self.autoencoder.eval()
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
                else:
                    latent_all = torch.cat([latent_all, latent], dim=0)
                    target_all = torch.cat([target_all, target], dim=0)

        latent_all = latent_all.cpu().numpy()
        target_all = target_all.cpu().numpy()
        target_all = target_all.astype(int)
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

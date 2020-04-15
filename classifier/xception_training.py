import os
import time
import torch
from os import makedirs
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from common.logging.logger import rootLogger
from common.logging.tf_logger import Logger
import numpy as np
import torch.nn as nn


class XceptionModel(object):

    def __init__(self, xception_model, model_name, loss, mode, optim_kwargs, dataset, batch_size=64,
                 optim=None, epochs=10, tf_log_path=None, use_cuda=None, xception_params=None,
                 patience=30, counter=0, early_stop=False, num_classes=2, model_dir='xception_base'):
        """

        :param xception_model:
        :param model_name:
        :param loss:
        :param optim_kwargs:
        :param dataset:
        :param batch_size:
        :param optim:
        :param epochs:
        :param tf_log_path:
        :param use_cuda:
        :param xception_params:
        :param patience:
        :param counter:
        :param early_stop:
        :param num_classes:
        """
        self.xception_model = xception_model
        model_params = None
        if mode == 'train':

            # For Conv 1, 2 and bn 1,2,3,4
            for module_name, network_module in self.xception_model.named_children():
                for param in network_module.parameters():
                    param.requires_grad = True

            # For Block 1, 2, ..... 12
            for module_name, network_module in self.xception_model.named_children():
                for name, module in network_module.named_children():
                    for param in module.parameters():
                        param.requires_grad = True
            total_params = sum(p.numel() for p in self.xception_model.parameters())
            trainable_params = sum(p.numel() for p in self.xception_model.parameters() if p.requires_grad)
            model_params = (p for p in self.xception_model.parameters() if p.requires_grad)
            rootLogger.info("Total Parameters : " + str(total_params))
            rootLogger.info("Trainable Parameters : " + str(trainable_params))

            # # model_params = filter(lambda x: x.requires_grad, xception_model.parameters())
            # model_params = (p for p in self.xception_model.parameters() if p.requires_grad)
            # params = sum([np.prod(p.size()) for p in model_params])

        elif mode == 'fine_tune':
            fine_tune_param_list = [self.xception_model.fc, self.xception_model.conv3, self.xception_model.bn3,
                                    self.xception_model.conv4, self.xception_model.bn4]

            # For Conv 1, 2 and bn 1,2,3,4
            for module_name, network_module in self.xception_model.named_children():
                if network_module not in fine_tune_param_list:
                    for param in network_module.parameters():
                        param.requires_grad = False

            # For Block 1, 2, ..... 12
            for module_name, network_module in self.xception_model.named_children():
                for name, module in network_module.named_children():
                    if module not in fine_tune_param_list:
                        for param in module.parameters():
                            param.requires_grad = False
            total_params = sum(p.numel() for p in self.xception_model.parameters())
            trainable_params = sum(p.numel() for p in self.xception_model.parameters() if p.requires_grad)
            model_params = (p for p in self.xception_model.parameters() if p.requires_grad)
            for name, param in self.xception_model.named_parameters():
                if param.requires_grad:
                    print(name)
            rootLogger.info("Total Parameters : " + str(total_params))
            rootLogger.info("Trainable Parameters : " + str(trainable_params))
        optim = optim or torch.optim.Adam
        optim_kwargs = optim_kwargs or {}
        optim_kwargs.setdefault('lr', 1e-3)
        self.optim = optim(model_params, **optim_kwargs)
        self.scheduler = StepLR(self.optim, step_size=10, gamma=1)
        self.epochs = epochs
        self.loss = loss
        self.model_name = model_name
        self.dataset = dataset
        self.batch_size = batch_size
        self.use_cuda = use_cuda
        self.patience = patience
        self.counter = counter
        self.num_classes = num_classes
        self.early_stop = early_stop
        self.model_dir = model_dir
        self.scheduler = StepLR(self.optim, step_size=10, gamma=1)
        if self.use_cuda is None:
            self.use_cuda = torch.cuda.is_available()
        if tf_log_path is not None:
            self.logger = Logger(model_name=self.model_name, data_name=self.dataset, log_path=tf_log_path)

    def train_model(self, train_loader, val_loader, model_path):

        # Show model settings
        # print(pretrainedmodels.pretrained_settings[self.model_name])
        #
        # model_ft = pretrainedmodels.__dict__[self.model_name](num_classes=1000, pretrained='imagenet')

        # results save folder
        model_current = model_path + '/xception_models/' + self.model_dir + '/' + self.dataset + '/current/' + self.model_name + '.pt'
        model_best = model_path + '/xception_models/' + self.model_dir + '/' + self.dataset + '/best/' + self.model_name + '.pt'
        if not os.path.isdir(model_path + '/xception_models/' + self.model_dir + '/' + self.dataset + '/current/'):
            makedirs(model_path + '/xception_models/' + self.model_dir + '/' + self.dataset + '/current/')
        if not os.path.isdir(model_path + '/xception_models/' + self.model_dir + '/' + self.dataset + '/best/'):
            makedirs(model_path + '/xception_models/' + self.model_dir + '/' + self.dataset + '/best/')

        try:
            # Load the pretrained imagenet model weights
            rootLogger.info("Loading ImageNet Pretrained Model")
            checkpoint = torch.load(
                '/home/shivangi/Desktop/Projects/pretrained_models/checkpoints/xception-43020ad28.pth')
            self.xception_model.load_state_dict(checkpoint)
            self.xception_model.last_linear = nn.Linear(2048, 2)  # Change the last layer output from 1000 to 2
            del self.xception_model.fc
            rootLogger.info("ImageNet Pretrained Model successfully loaded")
        except:
            rootLogger.info("Model not found/ Error loading model")

        # training-loop
        np.random.seed(int(time.time()))
        rootLogger.info('Training Start!!!')
        best_loss = np.Inf

        self.xception_model.cuda()
        for epoch in range(self.epochs):
            self.xception_model.train()
            epoch_start_time = time.time()
            epoch_train_loss = 0.
            correct = 0
            total = 0
            # Checkpoint after 5 epochs
            if (epoch + 1) % 5 == 0:
                try:
                    rootLogger.info("Saving the model after 5 epochs")
                    torch.save(self.xception_model.state_dict(), model_current)
                    rootLogger.info("Model Saved")
                except:
                    rootLogger.info("Can't save the model")

            for epoch_iter, data in enumerate(train_loader):

                input_image, label = data
                # Move the images to the device first before computation
                if torch.cuda.is_available():
                    input_image = input_image.cuda()
                    label = label.cuda()
                input_image = Variable(input_image)
                label = Variable(label)

                self.optim.zero_grad()  # clear gradients for this training step

                # 0 for real, 1 for fake
                output = self.xception_model(input_image)
                loss = self.loss(output, label)
                epoch_train_loss += loss.item()
                loss.backward()  # backprop, compute gradients
                self.optim.step()  # apply gradients

                # Calculate correct predictions
                total += label.size(0)
                _, predicted = torch.max(output, 1)
                correct += (label == predicted).sum().item()

            avg_loss = epoch_train_loss / len(train_loader)
            accuracy = 100. * correct / total

            # Log the training losses and accuracy
            self.logger.log(mode="train", error=avg_loss, epoch=epoch + 1, n_batch=0, num_batches=1)
            self.logger.log(mode="train", error=accuracy, epoch=epoch + 1, n_batch=0, num_batches=1, scalar='acc')

            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - epoch_start_time
            rootLogger.info(
                '[%d/%d] - ptime: %.2f Train : Total loss: %.3f, acc: %.3f' % (
                    (epoch + 1), self.epochs, per_epoch_ptime, avg_loss, accuracy))

            # Validation
            val_loss, val_acc = self.evaluate_val_data_tf(val_loader=val_loader, epoch=epoch)

            # Save the best model
            if val_loss <= best_loss:
                self.counter = 0
                self.xception_model.train()
                best_loss = val_loss
                torch.save(self.xception_model.state_dict(), model_best)
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

    def fine_tune_model(self, train_loader, val_loader, model_path):

        rootLogger.info("Fine tuning the model")
        # results save folder
        model_current = model_path + '/xception_models/' + self.model_dir + '/' + self.dataset + '/current/' + self.model_name + '.pt'
        model_best = model_path + '/xception_models/' + self.model_dir + '/' + self.dataset + '/best/' + self.model_name + '.pt'
        try:
            rootLogger.info("Loading Saved Model")
            checkpoint = torch.load(model_best)
            self.xception_model.last_linear = nn.Linear(2048, 2)  # Change the last layer output from 1000 to 2
            del self.xception_model.fc
            self.xception_model.load_state_dict(checkpoint)
            rootLogger.info("Saved Model successfully loaded")
        except:
            rootLogger.info("Model not found.")
            return

        # training-loop
        np.random.seed(int(time.time()))
        rootLogger.info('Training Start!!!')
        best_loss = np.Inf

        self.xception_model.cuda()
        for epoch in range(self.epochs):
            self.xception_model.train()
            epoch_start_time = time.time()
            epoch_train_loss = 0.
            correct = 0
            total = 0
            # Checkpoint after 5 epochs
            if (epoch + 1) % 5 == 0:
                try:
                    rootLogger.info("Saving the model after 5 epochs")
                    torch.save(self.xception_model.state_dict(), model_current)
                    rootLogger.info("Model Saved")
                except:
                    rootLogger.info("Can't save the model")

            for epoch_iter, data in enumerate(train_loader):

                input_image, label = data
                # Move the images to the device first before computation
                if torch.cuda.is_available():
                    input_image = input_image.cuda()
                    label = label.cuda()
                input_image = Variable(input_image)
                label = Variable(label)

                self.optim.zero_grad()  # clear gradients for this training step

                # 0 for real, 1 for fake
                output = self.xception_model(input_image)
                loss = self.loss(output, label)
                epoch_train_loss += loss.item()
                loss.backward()  # backprop, compute gradients
                self.optim.step()  # apply gradients

                # Calculate correct predictions
                total += label.size(0)
                _, predicted = torch.max(output, 1)
                correct += (label == predicted).sum().item()

            avg_loss = epoch_train_loss / len(train_loader)
            accuracy = 100. * correct / total

            # Log the training losses and accuracy
            self.logger.log(mode="train", error=avg_loss, epoch=epoch + 1, n_batch=0, num_batches=1)
            self.logger.log(mode="train", error=accuracy, epoch=epoch + 1, n_batch=0, num_batches=1, scalar='acc')

            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - epoch_start_time
            rootLogger.info(
                '[%d/%d] - ptime: %.2f Train : Total loss: %.3f, acc: %.3f' % (
                    (epoch + 1), self.epochs, per_epoch_ptime, avg_loss, accuracy))

            # Validation
            val_loss, val_acc = self.evaluate_val_data_tf(val_loader=val_loader, epoch=epoch)

            # Save the best model
            if val_loss <= best_loss:
                self.counter = 0
                self.xception_model.train()
                best_loss = val_loss
                torch.save(self.xception_model.state_dict(), model_best)
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
        self.xception_model.eval()
        epoch_val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for epoch_iter, data in enumerate(val_loader):
                input_image, label = data

                input_image = input_image.cuda()
                label = label.cuda()

                output = self.xception_model(input_image)
                loss = self.loss(output, label)
                epoch_val_loss += loss.item()

                # Calculate correct predictions
                total += label.size(0)
                _, predicted = torch.max(output, 1)
                correct += (predicted == label).sum().item()

            avg_loss = epoch_val_loss / len(val_loader)
            accuracy = 100. * correct / total

            rootLogger.info("Validation Total loss= [%.3f], Val acc= [%.3f]" %
                            (avg_loss, accuracy))
            self.logger.log(mode="val", error=avg_loss, epoch=epoch, n_batch=0, num_batches=1)
            self.logger.log(mode="val", error=accuracy, epoch=epoch, n_batch=0, num_batches=1, scalar='acc')
            return avg_loss, accuracy

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
                model_path + '/xception_models/' + self.model_dir + '/' + self.dataset + '/best/' + self.model_name + '.pt')
            if 'fc' in list(dict(self.xception_model.named_children()).keys()):
                self.xception_model.last_linear = nn.Linear(2048, 2)  # Change the last layer output from 1000 to 2
                del self.xception_model.fc
            self.xception_model.load_state_dict(checkpoint)
            rootLogger.info("Saved Model successfully loaded")
        except:
            rootLogger.info("Model not found.")

        self.xception_model.cuda().eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for epoch_iter, data in enumerate(val_loader):
                input_image, label = data
                if self.use_cuda:
                    input_image = input_image.cuda()
                    label = label.cuda()

                output = self.xception_model(input_image)
                loss = self.loss(output, label)
                val_loss += loss.item()

                # Calculate correct predictions
                total += label.size(0)
                _, predicted = torch.max(output, 1)
                correct += (predicted == label).sum().item()

            avg_loss = val_loss / len(val_loader)
            accuracy = 100. * correct / total

            f = open("results.txt", "a+")
            f.write("Validation Total loss= [%.3f], Val acc= [%.3f]" %
                    (avg_loss, accuracy))
            f.write("\n")
            f.close()
            rootLogger.info("Validation Total loss= [%.3f], Val acc= [%.3f]" %
                            (avg_loss, accuracy))

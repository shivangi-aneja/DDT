import os
import time
import torch
from torch.autograd import Variable
from common.logging.logger import rootLogger
from common.logging.tf_logger import Logger
from os import makedirs
from common.utils.image_utils import save_image
from sklearn.manifold import TSNE
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

width = 4000
height = 3000
max_dim = 100


class Classifier(object):

    def __init__(self, classifier, model_name, loss, optim_kwargs, dataset, batch_size, model_dataset,
                 optim=None, epochs=10, tf_log_path=None, use_cuda=None, classifier_params=None):
        """
        :param classifier: Classifier Network
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
        self.loss = loss
        self.model_name = model_name
        self.dataset = dataset
        self.model_dataset = model_dataset
        self.batch_size = batch_size
        self.use_cuda = use_cuda
        if self.use_cuda is None:
            self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.classifier.cuda()
        if tf_log_path is not None:
            self.logger = Logger(model_name=self.model_name, data_name=self.dataset, log_path=tf_log_path)

    def train_classifier(self, train_loader, val_loader, model_path):
        """
        Function for training
        :param train_loader: Loader for training data
        :param test_loader: Loader for test data
        :param model_path: Path for saving the data
        """

        # results save folder
        model_current = model_path + '/classifier/' + self.dataset + '/current/' + self.model_name + '.pt'
        model_best = model_path + '/classifier/' + self.dataset + '/best/' + self.model_name + '.pt'
        try:
            rootLogger.info("Loading Saved Model")
            checkpoint = torch.load(model_current)
            self.classifier.load_state_dict(checkpoint)
            rootLogger.info("Saved Model successfully loaded")
        except:
            rootLogger.info("Model not found, Created a new one")
            # Make directory for Saving Models
            if not os.path.isdir(model_path + 'classifier/' + self.dataset + '/current/'):
                makedirs(model_path + 'classifier/' + self.dataset + '/current/')
            if not os.path.isdir(model_path + 'classifier/' + self.dataset + '/best/'):
                makedirs(model_path + 'classifier/' + self.dataset + '/best/')

        # training-loop
        np.random.seed(int(time.time()))
        rootLogger.info('Training Start!!!')
        best_loss = np.Inf

        for epoch in range(self.epochs):
            self.classifier.train()
            epoch_start_time = time.time()
            epoch_train_loss = 0.
            correct = 0
            total = 0

            # Checkpoint after 5 epochs
            if epoch % 10 == 0:
                try:
                    rootLogger.info("Saving the model")
                    torch.save(self.classifier.state_dict(), model_current)
                    rootLogger.info("Model Saved")
                except:
                    rootLogger.info("Can't save the model")

            for epoch_iter, data in enumerate(train_loader):
                latent_space, labels = data
                # Move the images to the device first before computation
                if self.use_cuda:
                    latent_space = latent_space.cuda()
                    labels = labels.cuda()
                latent_space = Variable(latent_space)
                labels = Variable(labels)

                self.optim.zero_grad()  # clear gradients for this training step

                pred_labels = self.classifier(latent_space)
                loss = self.loss(pred_labels, labels)
                epoch_train_loss += loss.item()

                loss.backward()  # backpropagation, compute gradients
                self.optim.step()  # apply gradients

                _, predicted = torch.max(pred_labels.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            avg_loss = epoch_train_loss / len(train_loader)

            # Save the best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(self.classifier.state_dict(), model_best)
                rootLogger.info("Model Saved")

            accuracy = 100 * correct / total
            # Log the training losses and accuracy
            self.logger.log(mode="train", error=avg_loss, epoch=epoch + 1, n_batch=0, num_batches=1)
            self.logger.log_scores(mode="train", acc=accuracy, epoch=epoch + 1)

            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - epoch_start_time
            rootLogger.info('[%d/%d] - ptime: %.2f Train loss: %.3f  Train Acc: %.2f' % (
            (epoch + 1), self.epochs, per_epoch_ptime, avg_loss, accuracy))

            # Validation
            self.evaluate_val_data_tf(val_loader=val_loader, epoch=epoch)

    def evaluate_val_data_tf(self, val_loader, epoch):
        """
        Function to evaluate the results on trained model
        :param test_loader: data loader on which clustering is evaluated
        :param model_name: name with which pre-trained model is saved
        :param epoch:
        :return: None
        """
        self.classifier.eval()
        # Load the parameters of pretrained model
        # checkpoint = torch.load(model_name)
        # Evaluate the results on current model
        epoch_train_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for epoch_iter, data in enumerate(val_loader):
                latent_space, labels = data
                if self.use_cuda:
                    latent_space = latent_space.cuda()
                    labels = labels.cuda()
                latent_space = Variable(latent_space)
                labels = Variable(labels)

                pred_labels = self.classifier(latent_space)

                _, predicted = torch.max(pred_labels, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = self.loss(pred_labels, labels)
                epoch_train_loss += loss.item()

            avg_loss = epoch_train_loss / len(val_loader)
            accuracy = 100 * correct / total

            rootLogger.info("Val loss = [%.3f], Val Acc = [%.2f]" % (avg_loss, accuracy))
            self.logger.log(mode="val", error=avg_loss, epoch=epoch, n_batch=0, num_batches=1)
            self.logger.log_scores(mode="val", acc=accuracy, epoch=epoch + 1)

    def evaluate_test_data(self, val_loader, model_path):
        """
        Function to evaluate the results on trained model
        :param val_loader: data loader on which clustering is evaluated
        :param model_path: name with which pre-trained model is saved
        :return: None
        """
        try:
            rootLogger.info("Loading Saved Model")
            checkpoint = torch.load(
                model_path + '/classifier/' + self.model_dataset + '/best/' + self.model_name + '.pt')
            self.classifier.load_state_dict(checkpoint)
            rootLogger.info("Saved Model successfully loaded")
        except:
            rootLogger.info("Model not found.")
        self.classifier.eval()

        train_loss = 0
        correct = 0
        total = 0

        # classes = ('plane', 'car', 'bird', 'cat',
        #           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        # class_correct = list(0. for i in range(10))
        # class_total = list(0. for i in range(10))

        with torch.no_grad():
            for epoch_iter, data in enumerate(val_loader):
                latent_space, labels = data
                if self.use_cuda:
                    latent_space = latent_space.cuda()
                    labels = labels.cuda()
                latent_space = Variable(latent_space)
                labels = Variable(labels)

                pred_labels = self.classifier(latent_space)

                _, predicted = torch.max(pred_labels, 1)
                c = (predicted == labels).squeeze()

                # for i in range(labels.shape[0]):
                # label = labels[i]
                # class_correct[label] += c[i].item()
                # class_total[label] += 1

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = self.loss(pred_labels, labels)
                train_loss += loss.item()

            avg_loss = train_loss / len(val_loader)
            accuracy = 100.0 * correct / total

            rootLogger.info("Val loss = [%.3f], Val Acc = [%.2f]" % (avg_loss, accuracy))

            # for i in range(10):
            #    print('Accuracy of %5s : %2d %%' % (
            #    classes[i], 100 * class_correct[i] / class_total[i]))

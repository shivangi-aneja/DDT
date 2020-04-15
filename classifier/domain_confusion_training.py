import os
import time
from os import makedirs
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from common.logging.logger import rootLogger
from common.logging.tf_logger import Logger
from common.losses.mmd import mmd_linear
alpha = 0.5

class Classifier(object):

    def __init__(self, classifier, model_name, loss, optim_kwargs, dataset, batch_size,
                 optim=None, epochs=10, tf_log_path=None, use_cuda=None, patience=50, counter=0, early_stop=False,
                 classifier_params=None, num_classes=2):
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
        self.batch_size = batch_size
        self.use_cuda = use_cuda
        self.patience = patience
        self.counter = counter
        self.early_stop = early_stop
        self.num_classes = num_classes
        self.scheduler = StepLR(self.optim, step_size=10, gamma=0.5)
        if self.use_cuda is None:
            self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.classifier.cuda()
        if tf_log_path is not None:
            self.logger = Logger(model_name=self.model_name, data_name=self.dataset, log_path=tf_log_path)

    def train_classifier(self, source_train_loader, target_train_loader, source_val_loader, target_val_loader,
                         model_path):
        """
        Function for training
        :param train_loader: Loader for training data
        :param val_loader: Loader for test data
        :param model_path: Path for saving the data
        """

        # results save folder
        model_current = model_path + '/domain_confusion_models/arch1/' + self.dataset + '/current/' + self.model_name + '.pt'
        model_best = model_path + '/domain_confusion_models/arch1/' + self.dataset + '/best/' + self.model_name + '.pt'

        try:
            # Load the pretrained autoencoder
            rootLogger.info("Loading VGGFace2 Autoencoder Model")
            checkpoint = torch.load(model_path + '/autoencoders/vgg/best/autoencoder_model.pt')
            ae_model.load_state_dict(checkpoint)
            del ae_model.decoder

            ae_dict = ae_model.encoder.state_dict()
            classifier_dict = self.classifier.classifier.state_dict()
            # ae_dict = {k: v for k, v in ae_model.items() if k in classifier_dict}
            for k in ae_dict.keys():
                classifier_dict.update({k: ae_dict[k]})
            self.classifier.classifier.load_state_dict(classifier_dict)
            rootLogger.info("VGGFace2 Pretrained Encoder Model successfully loaded")
            # Make directory for Saving Models
            if not os.path.isdir(model_path + '/domain_confusion_models/arch1/' + self.dataset + '/current/'):
                makedirs(model_path + '/domain_confusion_models/arch1/' + self.dataset + '/current/')
            if not os.path.isdir(model_path + '/domain_confusion_models/arch1/' + self.dataset + '/best/'):
                makedirs(model_path + '/domain_confusion_models/arch1/' + self.dataset + '/best/')
        except:
            rootLogger.info("Error loading VGGFace2 Model")
            exit()

        # training-loop
        np.random.seed(int(time.time()))
        rootLogger.info('Training Start!!!')

        best_loss = np.Inf
        best_accuracy = 0.0

        len_source_loader = len(source_train_loader)
        len_target_loader = len(target_train_loader)

        for epoch in range(self.epochs):
            self.classifier.train()
            epoch_start_time = time.time()
            epoch_train_loss = 0.
            correct = 0
            total = 0

            # Checkpoint after 5 epochs
            if epoch % 10 == 0:
                try:
                    rootLogger.info("Saving the current model")
                    torch.save(self.classifier.state_dict(), model_current)
                    rootLogger.info("Current model saved")
                except:
                    rootLogger.info("Can't save the model")

            iter_source = iter(source_train_loader)
            iter_target = iter(target_train_loader)
            num_iter = len_source_loader
            for i in range(0, num_iter):
                data_source, label_source = iter_source.next()
                data_target, label_target = iter_target.next()
                if i % len_target_loader == 0:
                    iter_target = iter(target_train_loader)
                # Move the images to the device first before computation
                if self.use_cuda:
                    data_source = data_source.cuda()
                    label_source = label_source.cuda()
                    data_target = data_target.cuda()
                    label_target = label_target.cuda()

                data_source = Variable(data_source)
                label_source = Variable(label_source)
                data_target = Variable(data_target)
                label_target = Variable(label_target)

                self.optim.zero_grad()  # clear gradients for this training step

                latent_source, pred_source = self.classifier(data_source)
                latent_target, pred_target = self.classifier(data_target)
                mmd_loss = mmd_linear(latent_source, latent_target)
                loss = self.loss(pred_source, label_source) + self.loss(pred_target, label_target) + alpha*mmd_loss
                epoch_train_loss += loss.item()

                loss.backward()  # backpropagation, compute gradients
                self.optim.step()  # apply gradients

                _, predicted_source = torch.max(pred_source.data, 1)
                total += pred_source.size(0)
                _, predicted_target = torch.max(pred_target.data, 1)
                total += pred_target.size(0)
                correct += (predicted_source == label_source).sum().item()
                correct += (predicted_target == label_target).sum().item()

            avg_loss = epoch_train_loss / (len_source_loader+len_target_loader)

            # Calculate accuracy for current epoch
            accuracy = 100 * correct / total

            # Log the training losses and accuracy
            self.logger.log(mode="train", error=avg_loss, epoch=epoch + 1, n_batch=0, num_batches=1)
            self.logger.log_scores(mode="train", acc=accuracy, epoch=epoch + 1)

            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - epoch_start_time
            rootLogger.info('[%d/%d] - ptime: %.2f Train loss: %.3f  Train Acc: %.2f' % (
                (epoch + 1), self.epochs, per_epoch_ptime, avg_loss, accuracy))

            # Validation
            val_loss, val_acc = self.evaluate_val_data_tf(source_val_loader=source_val_loader, target_val_loader=target_val_loader, epoch=epoch)

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

    def evaluate_val_data_tf(self, source_val_loader, target_val_loader, epoch):
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
        epoch_train_loss = 0
        correct = 0
        total = 0

        len_source_loader = len(source_val_loader)
        len_target_loader = len(target_val_loader)

        with torch.no_grad():
            iter_source = iter(source_val_loader)
            iter_target = iter(target_val_loader)
            num_iter = len_source_loader
            for i in range(0, num_iter):
                data_source, label_source = iter_source.next()
                data_target, label_target = iter_target.next()
                if i % len_target_loader == 0:
                    iter_target = iter(target_val_loader)
                # Move the images to the device first before computation
                if self.use_cuda:
                    data_source = data_source.cuda()
                    label_source = label_source.cuda()
                    data_target = data_target.cuda()
                    label_target = label_target.cuda()

                data_source = Variable(data_source)
                label_source = Variable(label_source)
                data_target = Variable(data_target)
                label_target = Variable(label_target)

                latent_source, pred_source = self.classifier(data_source)
                latent_target, pred_target = self.classifier(data_target)
                mmd_loss = mmd_linear(latent_source, latent_target)
                loss = self.loss(pred_source, label_source) + self.loss(pred_target, label_target) + alpha * mmd_loss
                epoch_train_loss += loss.item()

                _, predicted_source = torch.max(pred_source.data, 1)
                total += pred_source.size(0)
                _, predicted_target = torch.max(pred_target.data, 1)
                total += pred_target.size(0)
                correct += (predicted_source == label_source).sum().item()
                correct += (predicted_target == label_target).sum().item()

            avg_loss = epoch_train_loss / (len_source_loader + len_target_loader)
            accuracy = 100 * correct / total

            rootLogger.info("Val loss = [%.3f], Val Acc = [%.2f]" % (avg_loss, accuracy))
            self.logger.log(mode="val", error=avg_loss, epoch=epoch, n_batch=0, num_batches=1)
            self.logger.log_scores(mode="val", acc=accuracy, epoch=epoch + 1)

        return avg_loss, accuracy

    def evaluate_val_data(self, val_loader, model_path):
        """
        Function to evaluate the results on trained model
        :param val_loader: data loader on which clustering is evaluated
        :param model_path: name with which pre-trained model is saved
        :return: None
        """

        try:
            rootLogger.info("Loading Saved Model")
            checkpoint = torch.load(
                model_path + '/domain_confusion_models/arch1/' + self.dataset + '/best/' + self.model_name + '.pt')
            self.classifier.load_state_dict(checkpoint)
            rootLogger.info("Saved Model successfully loaded")
        except:
            rootLogger.info("Model not found.")

        train_loss = 0
        correct = 0
        total = 0

        classes = ('orig', 'f2f')

        class_correct = list(0. for i in range(2))
        class_total = list(0. for i in range(2))
        self.classifier.eval()
        with torch.no_grad():
            for epoch_iter, data in enumerate(val_loader):
                input_image, labels = data
                if self.use_cuda:
                    input_image = input_image.cuda()
                    labels = labels.cuda()
                input_image = Variable(input_image)
                labels = Variable(labels)

                _, pred_labels = self.classifier(input_image)

                _, predicted = torch.max(pred_labels, 1)
                c = (predicted == labels).squeeze()

                for i in range(labels.shape[0]):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = self.loss(pred_labels, labels)
                train_loss += loss.item()

            avg_loss = train_loss / len(val_loader)
            accuracy = 100.0 * correct / total

            rootLogger.info("Val loss = [%.3f], Val Acc = [%.2f]" % (avg_loss, accuracy))

            for i in range(2):
                print('Accuracy of %5s : %2d %%' % (
                    classes[i], 100 * class_correct[i] / class_total[i]))

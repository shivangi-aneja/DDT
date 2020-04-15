import os
import time
from os import makedirs
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from common.logging.logger import rootLogger
from common.logging.tf_logger import Logger
import matplotlib.pyplot as plt
from torchsummary import summary
from mpl_toolkits.mplot3d import Axes3D
from sklearn import manifold

category_to_label = {0: 'Orig', 1: 'F2F', 2: 'DF', 3: 'FS'}
category_to_color = np.array(['#ff80ed', '#065535', '#133337', '#ffc0cb.'])


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

    def train_classifier(self, train_loader, val_loader, model_path):
        """
        Function for training
        :param train_loader: Loader for training data
        :param val_loader: Loader for test data
        :param model_path: Path for saving the data
        """

        # results save folder
        model_current = model_path + '/classifier_models/arch1/' + self.dataset + '/current/' + self.model_name + '.pt'
        model_best = model_path + '/classifier_models/arch1/' + self.dataset + '/best/' + self.model_name + '.pt'
        try:
            rootLogger.info("Loading Saved Model")
            checkpoint = torch.load(model_current)
            self.classifier.load_state_dict(checkpoint)
            rootLogger.info("Saved Model successfully loaded")
        except:
            rootLogger.info("Model not found, Created a new one")
            # Make directory for Saving Models
            if not os.path.isdir(model_path + '/classifier_models/arch1/' + self.dataset + '/current/'):
                makedirs(model_path + '/classifier_models/arch1/' + self.dataset + '/current/')
            if not os.path.isdir(model_path + '/classifier_models/arch1/' + self.dataset + '/best/'):
                makedirs(model_path + '/classifier_models/arch1/' + self.dataset + '/best/')

        # training-loop
        np.random.seed(int(time.time()))
        rootLogger.info('Training Start!!!')

        best_loss = np.Inf
        best_accuracy = 0.0

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

            for epoch_iter, data in enumerate(train_loader):
                input_image, labels = data
                # Move the images to the device first before computation
                if self.use_cuda:
                    input_image = input_image.cuda()
                    labels = labels.cuda()
                input_image = Variable(input_image)
                labels = Variable(labels)

                self.optim.zero_grad()  # clear gradients for this training step

                _, pred_labels = self.classifier(input_image)
                loss = self.loss(pred_labels, labels)
                epoch_train_loss += loss.item()

                loss.backward()  # backpropagation, compute gradients
                self.optim.step()  # apply gradients

                _, predicted = torch.max(pred_labels.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            avg_loss = epoch_train_loss / len(train_loader)

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
        epoch_train_loss = 0
        correct = 0
        total = 0

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
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = self.loss(pred_labels, labels)
                epoch_train_loss += loss.item()

            avg_loss = epoch_train_loss / len(val_loader)
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
                model_path + '/classifier_models/arch1/' + self.dataset + '/best/' + self.model_name + '.pt')
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

    def visualize_latent_tsne(self, val_loader, model_path, model_dir, file_name):
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
            print(
                model_path + '/classifier_models/' + model_dir + '/' + self.dataset + '/best/' + self.model_name + '.pt')
            checkpoint = torch.load(
                model_path + '/classifier_models/' + model_dir + '/' + self.dataset + '/best/' + self.model_name + '.pt')
            self.classifier.load_state_dict(checkpoint)
            rootLogger.info("Saved Model successfully loaded")
        except:
            rootLogger.info("Model not found.")
        self.classifier.eval()
        latent_all, target_all = None, None
        with torch.no_grad():
            for epoch_iter, data in enumerate(val_loader):
                imgs, c_labels = data
                # Move the images to the device first before computation
                if self.use_cuda:
                    imgs, c_labels = imgs.cuda(), c_labels.cuda()
                imgs, c_labels = Variable(imgs), Variable(c_labels)

                latent, _ = self.classifier(imgs)

                if latent_all is None:
                    latent_all = latent
                    c_labels_all = c_labels
                else:
                    latent_all = torch.cat([latent_all, latent], dim=0)
                    c_labels_all = torch.cat([c_labels_all, c_labels], dim=0)

        latent_all = latent_all.cpu().numpy()
        target_all = c_labels_all.cpu().numpy()
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

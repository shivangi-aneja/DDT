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
alpha_classification = 1.0
alpha_center_loss = 1.0
category_to_label = {0: 'Orig', 1: 'F2F', 2: 'DF', 3: 'FS'}
category_to_color = np.array(['#ff80ed', '#065535', '#133337', '#ffc0cb.'])

class Siamese(object):

    def __init__(self, siamese, model_name, siamese_loss, classification_loss, optim_kwargs, dataset, batch_size,
                 patience=50,
                 optim=None, epochs=10, tf_log_path=None, use_cuda=None, siamese_params=None, model_dir='siamese'):
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
        if siamese_params is None or not len(siamese_params):
            siamese_params = filter(lambda x: x.requires_grad, self.siamese.parameters())
            params = sum([np.prod(p.size()) for p in self.siamese.parameters()])
            rootLogger.info("Trainable Parameters : " + str(params))
        optim = optim or torch.optim.Adam
        optim_kwargs = optim_kwargs or {}
        optim_kwargs.setdefault('lr', 1e-3)
        self.center_loss = CenterLoss(num_classes=2, feat_dim=128, use_gpu=True)
        self.optim = optim(siamese_params, **optim_kwargs)
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
        self.scheduler = StepLR(self.optim, step_size=10, gamma=0.5)
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
        loss_params = model_path + '/siamese_models/' + self.model_dir + '/' + self.dataset + '/best/' + self.model_name + '_loss_params.pt'

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
            epoch_sim_loss = 0.
            epoch_center_loss = 0.
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

                img0, c_label_img0, img1, c_label_img1, label = data
                # Move the images to the device first before computation
                if self.use_cuda:
                    img0, c_label_img0, img1, c_label_img1, label = img0.cuda(), c_label_img0.cuda(), img1.cuda(), c_label_img1.cuda(), label.cuda()
                img0, c_label_img0, img1, c_label_img1, label = Variable(img0), Variable(c_label_img0), Variable(img1), Variable(c_label_img1), Variable(label)

                self.optim.zero_grad()  # clear gradients for this training step
                pred_img_label1, output1, pred_img_label2, output2 = self.siamese(img0, img1)
                pred_img_labels = torch.cat((pred_img_label1, pred_img_label2), dim=0)
                img_labels = torch.cat((c_label_img0, c_label_img1), dim=0)
                center_loss = torch.mean(torch.tensor([self.center_loss(output1, c_label_img0), self.center_loss(output2, c_label_img1)]))
                loss = alpha_siamese_loss * self.siamese_loss(output1, output2, label) + alpha_classification * self.classification_loss(
                    pred_img_labels, img_labels) + alpha_center_loss * center_loss
                epoch_class_loss += self.classification_loss(pred_img_labels, img_labels)
                epoch_sim_loss += self.siamese_loss(output1, output2, label)
                epoch_center_loss += center_loss
                epoch_train_loss += loss.item()
                loss.backward()  # backpropagation, compute gradients
                self.optim.step()  # apply gradients
                _, predicted = torch.max(pred_img_labels.data, 1)
                total += img_labels.size(0)
                correct += (predicted == img_labels).sum().item()

            avg_loss = epoch_train_loss / len(train_loader)
            avg_class_loss = epoch_class_loss / len(train_loader)
            avg_sim_loss = epoch_sim_loss / len(train_loader)
            avg_center_loss = epoch_center_loss / len(train_loader)

            # Calculate accuracy for current epoch
            accuracy = 100 * correct / total

            # Log the training losses and accuracy
            self.logger.log(mode="train", error=avg_loss, epoch=epoch + 1, n_batch=0, num_batches=1)
            self.logger.log(mode="train", error=avg_class_loss, epoch=epoch + 1, n_batch=0, num_batches=1,
                            scalar='class_loss')
            self.logger.log(mode="train", error=avg_sim_loss, epoch=epoch + 1, n_batch=0, num_batches=1,
                            scalar='sim_loss')
            self.logger.log(mode="train", error=avg_center_loss, epoch=epoch + 1, n_batch=0, num_batches=1,
                            scalar='center_loss')
            self.logger.log(mode="train", error=accuracy, epoch=epoch + 1, n_batch=0, num_batches=1, scalar='acc')

            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - epoch_start_time
            rootLogger.info('[%d/%d] - ptime: %.2f Train: loss: %.3f  Class_loss: %.3f  Sim_loss: %.3f  Center_loss: %.3f  Acc: %.2f' % (
                (epoch + 1), self.epochs, per_epoch_ptime, avg_loss, avg_class_loss, avg_sim_loss, avg_center_loss, accuracy))

            val_loss, val_acc = self.evaluate_val_data_tf(test_loader=test_loader, epoch=epoch)

            # Save the best model
            if val_loss <= best_loss:
                self.counter = 0
                self.siamese.train()
                best_loss = val_loss
                torch.save(self.siamese.state_dict(), model_best)
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
        epoch_sim_loss = 0.
        epoch_center_loss = 0.
        correct = 0
        total = 0
        with torch.no_grad():
            for epoch_iter, data in enumerate(test_loader):
                img0, c_label_img0, img1, c_label_img1, label = data
                # Move the images to the device first before computation
                if self.use_cuda:
                    img0, c_label_img0, img1, c_label_img1, label = img0.cuda(), c_label_img0.cuda(), img1.cuda(), c_label_img1.cuda(), label.cuda()
                img0, c_label_img0, img1, c_label_img1, label = Variable(img0), Variable(c_label_img0), Variable(
                    img1), Variable(c_label_img1), Variable(label)

                pred_img_label1, output1, pred_img_label2, output2 = self.siamese(img0, img1)
                pred_img_labels = torch.cat((pred_img_label1, pred_img_label2), dim=0)
                img_labels = torch.cat((c_label_img0, c_label_img1), dim=0)
                center_loss = torch.mean(
                    torch.tensor([self.center_loss(output1, c_label_img0), self.center_loss(output2, c_label_img1)]))
                loss = alpha_siamese_loss * self.siamese_loss(output1, output2, label) + alpha_classification * self.classification_loss(
                    pred_img_labels, img_labels) + alpha_center_loss * center_loss
                epoch_class_loss += self.classification_loss(pred_img_labels, img_labels)
                epoch_sim_loss += self.siamese_loss(output1, output2, label)
                epoch_center_loss += center_loss
                epoch_train_loss += loss.item()
                _, predicted = torch.max(pred_img_labels.data, 1)
                total += img_labels.size(0)
                correct += (predicted == img_labels).sum().item()

            avg_loss = epoch_train_loss / len(test_loader)
            avg_class_loss = epoch_class_loss / len(test_loader)
            avg_sim_loss = epoch_sim_loss / len(test_loader)
            avg_center_loss = epoch_center_loss/ len(test_loader)

            # Calculate accuracy for current epoch
            accuracy = 100 * correct / total

            # Log the training losses and accuracy
            self.logger.log(mode="val", error=avg_loss, epoch=epoch + 1, n_batch=0, num_batches=1)
            self.logger.log(mode="val", error=avg_class_loss, epoch=epoch + 1, n_batch=0, num_batches=1,
                            scalar='class_loss')
            self.logger.log(mode="val", error=avg_sim_loss, epoch=epoch + 1, n_batch=0, num_batches=1,
                            scalar='sim_loss')
            self.logger.log(mode="val", error=avg_center_loss, epoch=epoch + 1, n_batch=0, num_batches=1,
                            scalar='center_loss')
            self.logger.log(mode="val", error=accuracy, epoch=epoch + 1, n_batch=0, num_batches=1, scalar='acc')

            rootLogger.info('[%d/%d] Val loss: %.3f  Class_loss: %.3f  Sim_loss: %.3f  Center_loss: %.3f  Acc: %.2f' %
                            ((epoch + 1), self.epochs, avg_loss, avg_class_loss, avg_sim_loss, avg_center_loss, accuracy))
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
            loss_params = model_path + '/siamese_models/' + self.model_dir + '/' + self.dataset + '/best/' + self.model_name + '_loss_params.pt'
            checkpoint = torch.load(
                model_path + '/siamese_models/' + self.model_dir + '/' + self.dataset + '/best/' + self.model_name + '.pt')
            self.siamese.load_state_dict(checkpoint)
            self.center_loss.centers = torch.load(loss_params)
            rootLogger.info("Saved Model successfully loaded")
        except:
            rootLogger.info("Model not found.")

        self.siamese.eval()
        epoch_class_loss = 0.
        epoch_center_loss = 0.
        correct = 0
        total = 0
        with torch.no_grad():
            for epoch_iter, data in enumerate(test_loader):
                imgs, c_labels = data
                # Move the images to the device first before computation
                if self.use_cuda:
                    imgs, c_labels = imgs.cuda(), c_labels.cuda()
                imgs, c_labels = Variable(imgs), Variable(c_labels)

                pred_img_label1, output1, pred_img_label2, output2 = self.siamese(imgs, imgs)
                epoch_class_loss += self.classification_loss(pred_img_label1, c_labels)
                center_loss = self.center_loss(output1, c_labels)
                epoch_center_loss += center_loss
                _, predicted = torch.max(pred_img_label1.data, 1)
                total += c_labels.size(0)
                correct += (predicted == c_labels).sum().item()

            avg_class_loss = epoch_class_loss / len(test_loader)
            avg_center_loss = epoch_center_loss / len(test_loader)

            # Calculate accuracy for current epoch
            accuracy = 100 * correct / total

            f = open("results.txt", "a+")
            f.write("Validation Class_loss: [%.3f] Center_loss: [%.3f] acc: [%.3f]" % (avg_class_loss, avg_center_loss, accuracy))
            f.write("\n")
            f.close()
            rootLogger.info("Validation Class_loss: [%.3f] Center_loss: [%.3f] acc: [%.3f]" % (avg_class_loss, avg_center_loss, accuracy))

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
        self.siamese.eval()
        latent_all, target_all = None, None
        with torch.no_grad():
            for epoch_iter, data in enumerate(val_loader):
                imgs, c_labels = data
                # Move the images to the device first before computation
                if self.use_cuda:
                    imgs, c_labels = imgs.cuda(), c_labels.cuda()
                imgs, c_labels = Variable(imgs), Variable(c_labels)

                _, latent, _, _ = self.siamese(imgs, imgs)

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
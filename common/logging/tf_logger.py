import errno
import os

import matplotlib
import numpy as np
import torchvision.utils as vutils
from IPython import display
from tensorboardX import SummaryWriter

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import torch


class Logger:

    def __init__(self, model_name, data_name, log_path):
        self.model_name = model_name
        self.data_name = data_name

        self.comment = '{}_{}'.format(model_name, data_name)
        self.data_subdir = '{}/{}'.format(model_name, data_name)

        # TensorBoard
        self.train_writer = SummaryWriter(logdir=log_path+'/train/', comment=self.comment)
        self.val_writer = SummaryWriter(logdir=log_path+'/val/', comment=self.comment)

    def log(self, mode, error, epoch, n_batch, num_batches, scalar='error'):
        """
        Log errors
        :param mode: Train/Val
        :param d_error: Discriminator Error
        :param g_error: Generator Error
        :param epoch: Epoch Number
        :param n_batch: Batch Number
        :param num_batches: Number Of Batches
        :param scalar:
        :return: None
        """

        # var_class = torch.autograd.variable.Variable
        if isinstance(error, torch.autograd.Variable):
            error = error.data.cpu().numpy()

        step = Logger._step(epoch, n_batch, num_batches)
        if mode == 'train':
            self.train_writer.add_scalar(self.comment+'_' + scalar, error, step)
        if mode == 'test':
            self.val_writer.add_scalar(self.comment+'_' + scalar, error, step)


    def log_scores(self, mode, acc, epoch):
        """
        Logs scores on Validation Data
        :param mode: Train/Val
        :param mse: Mean Squared Error
        :param psnr: Peak Signal To Noise Ratio
        :param disc_acc: Discriminator Accuracy
        :param epoch: Epoch
        :return: None
        """

        # var_class = torch.autograd.variable.Variable
        if isinstance(acc, torch.autograd.Variable):
            acc = acc.data.cpu().numpy()

        step = Logger._step(epoch, n_batch=0, num_batches=1)
        self.val_writer.add_scalar(self.comment+'_acc'+'/', acc, step)

    def log_images(self, mode, images, num_images, epoch, n_batch, num_batches, normalize=True):
        """
        Logs Images (input images are expected in format (NCHW))
        :param mode: Train/Val
        :param images: Images
        :param num_images: Number Of Images
        :param epoch: Epoch
        :param n_batch: Batch Number
        :param num_batches: Number Of Batches
        :param normalize: Normalize the image or not
        :return: None
        """

        img_name = '{}_images{}'.format(mode + '_' + self.comment, '', epoch)

        # Make horizontal grid from image tensor
        horizontal_grid = vutils.make_grid(
            images, normalize=normalize, scale_each=True).cpu()

        # Add images to tensorboard
        self.val_writer.add_image(img_name, horizontal_grid, epoch)

    def save_torch_images(self, horizontal_grid, grid, epoch, n_batch, plot_horizontal=True):
        out_dir = './data/images/{}'.format(self.data_subdir)
        Logger._make_dir(out_dir)

        # Plot and save horizontal
        fig = plt.figure(figsize=(16, 16))
        plt.imshow(np.moveaxis(horizontal_grid.numpy(), 0, -1))
        plt.axis('off')
        if plot_horizontal:
            display.display(plt.gcf())
        self._save_images(fig, epoch, n_batch, 'hori')
        plt.close()

        # Save squared
        fig = plt.figure()
        plt.imshow(np.moveaxis(grid.numpy(), 0, -1))
        plt.axis('off')
        self._save_images(fig, epoch, n_batch)
        plt.close()

    def _save_images(self, fig, epoch, n_batch, comment=''):
        out_dir = './data/images/{}'.format(self.data_subdir)
        Logger._make_dir(out_dir)
        fig.savefig('{}/{}_epoch_{}_batch_{}.png'.format(out_dir,
                                                         comment, epoch, n_batch))

    def display_status(self, epoch, num_epochs, n_batch, num_batches, d_error, g_error, d_pred_real, d_pred_fake):

        # var_class = torch.autograd.variable.Variable
        if isinstance(d_error, torch.autograd.Variable):
            d_error = d_error.data.cpu().numpy()
        if isinstance(g_error, torch.autograd.Variable):
            g_error = g_error.data.cpu().numpy()
        if isinstance(d_pred_real, torch.autograd.Variable):
            d_pred_real = d_pred_real.data
        if isinstance(d_pred_fake, torch.autograd.Variable):
            d_pred_fake = d_pred_fake.data

        print('Epoch: [{}/{}], Batch Num: [{}/{}]'.format(
            epoch, num_epochs, n_batch, num_batches)
        )
        print('Discriminator Loss: {:.4f}, Generator Loss: {:.4f}'.format(d_error, g_error))
        print('D(x): {:.4f}, D(G(z)): {:.4f}'.format(d_pred_real.mean(), d_pred_fake.mean()))

    def save_models(self, generator, discriminator, epoch):
        out_dir = './data/models/{}'.format(self.data_subdir)
        Logger._make_dir(out_dir)
        torch.save(generator.state_dict(),
                   '{}/G_epoch_{}'.format(out_dir, epoch))
        torch.save(discriminator.state_dict(),
                   '{}/D_epoch_{}'.format(out_dir, epoch))

    def close(self):
        self.writer.close()

    # Private Functionality

    @staticmethod
    def _step(epoch, n_batch, num_batches):
        return epoch * num_batches + n_batch

    @staticmethod
    def _make_dir(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

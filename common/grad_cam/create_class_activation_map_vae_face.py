import cv2
import os
import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable
from common.models.resnet_subset_models import VariationalEncoder1 as Encoder
from common.losses.custom_losses import wasserstein_distance_vector, wasserstein_distance

MODEL_PATH = os.path.join(os.getcwd(), 'models/')
latent_dim = 16


def save_class_activation_on_image(org_img, activation_map, file_name):
    """
        Saves CAM_ft activation map and activation map on the original image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): activation map (grayscale) 0-255
        file_name (str): File name of the exported image
    """

    path = os.getcwd() + '/results/CAM_vae/'
    if not os.path.exists(path):
        os.makedirs(path)

    # Heatmap of activation map
    activation_heatmap = cv2.applyColorMap(activation_map, cv2.COLORMAP_JET)
    # Heatmap on picture
    org_img = cv2.resize(org_img, (256, 256))
    img_with_heatmap = np.float32(activation_heatmap) + np.float32(org_img)
    img_with_heatmap = img_with_heatmap / np.max(img_with_heatmap)
    path_to_file = os.path.join(path, file_name + '_Cam_On_Image.jpg')
    cv2.imwrite(path_to_file, np.uint8(255 * img_with_heatmap))


class CamExtractor:
    """
        Extracts cam features from the model
    """

    def __init__(self, model):
        self.model = model
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x.register_hook(self.save_gradient)
        x = self.model.layer3(x)

        conv_output = x  # Save the convolution output on that layer

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        mu = self.model.fc1(x)
        logvar = self.model.fc2(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return conv_output, mu, logvar

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, mu, logvar = self.forward_pass_on_convolutions(x)
        return conv_output, mu, logvar


class GradCam:
    """
        Produces class activation map
    """

    def __init__(self, model):
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model)

    def generate_cam(self, input_image, target):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, mu, logvar = self.extractor.forward_pass(input_image)

        # Real
        mean1 = torch.zeros(int(latent_dim)).cuda()
        mean1[:int(latent_dim / 2)] = 1
        mean1[int(latent_dim / 2):] = 0
        # # Fake
        mean2 = torch.zeros(int(latent_dim)).cuda()
        mean2[:int(latent_dim / 2)] = 0
        mean2[int(latent_dim / 2):] = 1

        dist_real = wasserstein_distance_vector(mu=mu, logvar=logvar, mean=mean1)
        dist_fake = wasserstein_distance_vector(mu=mu, logvar=logvar, mean=mean2)
        dist = torch.stack([dist_real, dist_fake], dim=1)
        _, pred_class = torch.min(dist, 1)
        # Target for backprop
        self.model.zero_grad()
        # _, predicted = torch.max(pred, 1)
        print(bool(pred_class == target))
        # pred.backward(gradient=one_hot_output, retain_graph=True)
        if target == torch.tensor([0]).cuda():
            print("Real")
            loss = wasserstein_distance(mu=mu, logvar=logvar, mean=mean1)
            loss.backward(retain_graph=True)
        else:
            print("Fake")
            loss = wasserstein_distance(mu=mu, logvar=logvar, mean=mean2)
            loss.backward(retain_graph=True)

        # Backward pass with specified target


        # Get hooked gradients
        guided_gradients = self.extractor.gradients.data.cpu().numpy()[0]
        # Get convolution outputs
        target = conv_output.data.cpu().numpy()[0]
        # Get weights from gradients
        weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient
        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # Multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        cam = cv2.resize(cam, (256, 256))
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize

        return cam


if __name__ == '__main__':
    # Get image
    classes = {'orig': 0, 'df': 1, 'fs': 2, 'f2f': 3, 'nt': 4}
    img_list = ['004', '012', '026', '042', '046', '064', '084', '091', '135', '138', '896']
    for img_num in img_list:
        cam_class = 'df'
        print(img_num)
        target_class = torch.tensor([classes.get(cam_class)]).cuda()
        target_class[target_class == 2] = 1
        target_class[target_class == 3] = 1
        target_class[target_class == 4] = 1
        input_img = cv2.imread(os.getcwd() + "/data/sample/face/" + cam_class + '/' + img_num + ".png")
        input_img = np.array(input_img, dtype='uint8')
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        to_tensor = transforms.ToTensor()
        im_as_ten = to_tensor(input_img)
        im_as_ten.unsqueeze_(0)
        im_as_ten = Variable(im_as_ten, requires_grad=True)
        im_as_ten = im_as_ten.cuda()
        ft_images = '100'
        run = '10'
        # best_path = MODEL_PATH + 'vae_finetune/face/2classes_' + ft_images + 'images/nt_to_df/' + run + '_run/'
        best_path = MODEL_PATH + 'vae/face/2classes/best/'
        encoder_model_name = 'vae_train_20k_val3k_mean1_std1_c23_latent16_3blocks_2classes_mixup_flip_normalize_nt.pt'
        # encoder_model_name = 'vae_train_src_c23_latent128_resnet18_unweighted_2classes_update_last_nt_to_df_' + ft_images + 'images.pt'
        # best_path_classifier = MODEL_PATH + 'vae_classifier/face/2classes/best/'
        # classifier_model_name = 'vae_train_10k_val3k_mean1_std1_abs_c23_latent128_resnet18_unweighted_2classes_nt.pt'
        device = torch.device("cuda")
        model = Encoder(latent_dim=latent_dim).to(device)
        # classifier = CLASSIFIER(latent_dim=latent_dim).to(device)

        try:
            print("Loading Saved VAE Model")
            checkpoint = torch.load(best_path + encoder_model_name)
            model.load_state_dict(checkpoint)
        except:
            print("Model not found.")
            exit()

        model.eval()

        # Grad cam
        grad_cam = GradCam(model)
        # Generate cam mask
        cam = grad_cam.generate_cam(input_image=im_as_ten, target=target_class)
        # Save mask
        save_class_activation_on_image(input_img, cam, str(img_num) + '_' + cam_class)

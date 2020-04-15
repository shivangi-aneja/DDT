import cv2
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.autograd import Variable
from classifier.xception_training import XceptionModel
from pretrainedmodels.models.xception import Xception
from common.logging.logger import rootLogger

MODEL_PATH = os.path.join(os.getcwd(), 'models/')
loss = nn.CrossEntropyLoss()


def save_class_activation_on_image(org_img, activation_map, file_name):
    """
        Saves CAM_ft activation map and activation map on the original image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): activation map (grayscale) 0-255
        file_name (str): File name of the exported image
    """

    if not os.path.exists(os.getcwd() + '/results/CAM_xception/'):
        os.makedirs(os.getcwd() + '/results/CAM_xception/')

    # Heatmap of activation map
    activation_heatmap = cv2.applyColorMap(activation_map, cv2.COLORMAP_JET)
    # path_to_file = os.path.join(os.getcwd() + '/results/CAM_xception/', file_name + '_Cam_Heatmap.jpg')
    # cv2.imwrite(path_to_file, activation_heatmap)
    # Heatmap on picture
    org_img = cv2.resize(org_img, (256, 256))
    img_with_heatmap = np.float32(activation_heatmap) + np.float32(org_img)
    img_with_heatmap = img_with_heatmap / np.max(img_with_heatmap)
    path_to_file = os.path.join(os.getcwd() + '/results/CAM_xception/', file_name + '_Cam_On_Image.jpg')
    cv2.imwrite(path_to_file, np.uint8(255 * img_with_heatmap))


class CamExtractor:
    """
        Extracts cam features from the model
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        for module_pos, module in self.model._modules.items():
            x = module(x)  # Forward
            if module_pos == self.target_layer:
                x.register_hook(self.save_gradient)
                conv_output = x  # Save the convolution output on that layer
                return conv_output, x

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)
        relu_inplace = nn.ReLU(inplace=True)
        x = relu_inplace(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.model.last_linear(x)
        return conv_output, x


class GradCam:
    """
        Produces class activation map
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model, target_layer)

    def generate_cam(self, input_image, target):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, model_output = self.extractor.forward_pass(input_image)
        print(model_output)
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target] = 1
        # Zero grads
        self.model.last_linear.zero_grad()
        # Backward pass with specified target
        model_output.backward(gradient=one_hot_output, retain_graph=True)
        # Get hooked gradients
        guided_gradients = self.extractor.gradients.data.numpy()[0]
        # Get convolution outputs
        target = conv_output.data.numpy()[0]
        # Get weights from gradients
        # Take averages for each gradient
        weights = np.mean(guided_gradients, axis=(1, 2))
        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float64)
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
    classes = {'orig': 0, 'f2f': 1, 'df': 1, 'fs': 1}
    # img_num = '50325'
    img_num = '52445'
    cam_class = 'fs'
    target_class = torch.tensor([classes.get(cam_class)])
    input_img = cv2.imread(os.getcwd() + "/data/ff/test/" + cam_class + '/' + img_num + ".png")
    input_img = np.array(input_img, dtype='uint8')
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    to_tensor = transforms.ToTensor()
    im_as_ten = to_tensor(input_img)
    im_as_ten.unsqueeze_(0)
    im_as_ten = Variable(im_as_ten, requires_grad=True)
    im_as_ten = im_as_ten

    model_name = 'xception'
    lr = 1e-4
    dataset = 'ff'
    optim = torch.optim.Adam
    epochs = 1
    num_classes = 2
    run = '3_run'
    ft_image_mode = 'ft_100'

    # Create the Xception model for training
    xception_model = Xception(num_classes=1000)
    xception_classifier = XceptionModel(xception_model=xception_model, model_name=model_name, loss=loss,
                                        optim_kwargs={'lr': lr}, dataset=dataset, batch_size=64,
                                        optim=optim, epochs=epochs, use_cuda=True, num_classes=num_classes,
                                        mode='train')
    # Load the model
    try:
        rootLogger.info("Loading Saved Model")
        checkpoint = torch.load(
            MODEL_PATH + '/xception_models/' + run + '/xception_' + ft_image_mode + '/' + dataset + '/best/' + model_name + '.pt')
        xception_classifier.xception_model.last_linear = nn.Linear(2048,
                                                                   2)  # Change the last layer output from 1000 to 2
        del xception_classifier.xception_model.fc
        xception_classifier.xception_model.load_state_dict(checkpoint)
        rootLogger.info("Saved Model successfully loaded")
    except:
        rootLogger.info("Model not found.")

    xception_classifier.xception_model.eval()
    # freeze layers
    for param in xception_classifier.xception_model.parameters():
        param.requires_grad = False

    # Grad cam
    grad_cam = GradCam(model=xception_classifier.xception_model, target_layer='bn4')
    # Generate cam mask
    cam = grad_cam.generate_cam(input_image=im_as_ten, target=target_class)
    # Save mask
    save_class_activation_on_image(input_img, cam, str(img_num) + '_' + run + '_' + ft_image_mode + '_' + cam_class)
    print('Grad cam completed')

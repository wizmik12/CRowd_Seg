import numpy as np
import torch
from utils.model_architecture import create_segmentation_backbone
import utils.globals


def double_conv(in_channels, out_channels, step, norm):
    # ===========================================
    # in_channels: dimension of input
    # out_channels: dimension of output
    # step: stride
    # ===========================================
    if norm == 'in':
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, 3, stride=step, padding=1, groups=1, bias=False),
            torch.nn.InstanceNorm2d(out_channels, affine=True),
            torch.nn.PReLU(),
            torch.nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, groups=1, bias=False),
            torch.nn.InstanceNorm2d(out_channels, affine=True),
            torch.nn.PReLU()
        )
    elif norm == 'bn':
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, 3, stride=step, padding=1, groups=1, bias=False),
            torch.nn.BatchNorm2d(out_channels, affine=True),
            torch.nn.PReLU(),
            torch.nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, groups=1, bias=False),
            torch.nn.BatchNorm2d(out_channels, affine=True),
            torch.nn.PReLU()
        )
    elif norm == 'ln':
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, 3, stride=step, padding=1, groups=1, bias=False),
            torch.nn.GroupNorm(out_channels, out_channels, affine=True),
            torch.nn.PReLU(),
            torch.nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, groups=1, bias=False),
            torch.nn.GroupNorm(out_channels, out_channels, affine=True),
            torch.nn.PReLU()
        )
    elif norm == 'gn':
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, 3, stride=step, padding=1, groups=1, bias=False),
            torch.nn.GroupNorm(out_channels // 8, out_channels, affine=True),
            torch.nn.PReLU(),
            torch.nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, groups=1, bias=False),
            torch.nn.GroupNorm(out_channels // 8, out_channels, affine=True),
            torch.nn.PReLU()
        )



class global_CM(torch.nn.Module):
    """ This defines the annotator network (CR Global)
    """

    def __init__(self, class_no, input_height, input_width, noisy_labels_no):
        super(global_CM, self).__init__()
        self.class_no = class_no
        self.noisy_labels_no = noisy_labels_no
        self.input_height = input_height
        self.input_width = input_width
        self.noisy_labels_no = noisy_labels_no
        self.dense_output = torch.nn.Linear(noisy_labels_no, class_no ** 2)
        self.act = torch.nn.Softplus()
        # self.relu = torch.nn.ReLU()

    def forward(self, A_id, x=None):
        output = self.act(self.dense_output(A_id))
        all_weights = output.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, 512, 512)
        y = all_weights.view(-1, self.class_no**2, self.input_height, self.input_width)



        return y


class conv_layers_image(torch.nn.Module):
    def __init__(self, in_channels):
        super(conv_layers_image, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_bn = torch.nn.BatchNorm2d(8)
        self.conv_bn2 = torch.nn.BatchNorm2d(4)
        self.fc_bn = torch.nn.BatchNorm1d(128)
        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(in_features=4096, out_features=128)
        self.fc2 = torch.nn.Linear(in_features=128, out_features=64)

    def forward(self, x):
        x = self.pool(self.relu(self.conv_bn(self.conv(x))))
        x = self.pool(self.relu(self.conv_bn2(self.conv2(x))))
        x = self.pool(self.relu(self.conv_bn2(self.conv3(x))))
        x = self.pool(self.relu(self.conv_bn2(self.conv3(x))))
        x = self.flatten(x)

        x = self.relu(self.fc_bn(self.fc1(x)))
        y = self.fc2(x)

        return y


class image_CM(torch.nn.Module):
    """ This defines the annotator network (CR Image)
    """

    def __init__(self, class_no, input_height, input_width, noisy_labels_no):
        super(image_CM, self).__init__()
        self.class_no = class_no
        self.noisy_labels_no = noisy_labels_no
        self.input_height = input_height
        self.input_width = input_width
        self.noisy_labels_no = noisy_labels_no
        self.conv_layers = conv_layers_image(16)
        self.dense_annotator = torch.nn.Linear(noisy_labels_no, 64)
        self.dense_output = torch.nn.Linear(128, class_no ** 2)
        self.norm = torch.nn.BatchNorm1d(class_no ** 2)
        self.act = torch.nn.Softplus()

    def forward(self, A_id, x):
        A_feat = self.dense_annotator(A_id)  # B, F_A
        x = self.conv_layers(x)
        output = self.dense_output(torch.hstack((A_feat, x)))
        output = self.norm(output)
        output = self.act(output.view(-1, self.class_no, self.class_no))
        all_weights = output.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, self.input_height, self.input_width)
        y = all_weights.view(-1, self.class_no**2, self.input_height, self.input_width)

        return y



class cm_layers(torch.nn.Module):
    """ This defines the annotator network (CR Pixel)
    """

    def __init__(self, in_channels, norm, class_no, noisy_labels_no):
        super(cm_layers, self).__init__()
        self.conv_1 = double_conv(in_channels=in_channels, out_channels=in_channels, norm=norm, step=1)
        self.conv_2 = double_conv(in_channels=in_channels, out_channels=in_channels, norm=norm, step=1)
        # self.conv_last = torch.nn.Conv2d(in_channels, class_no ** 2, 1, bias=True)
        self.class_no = class_no
        self.dense = torch.nn.Linear(80, 25)
        self.dense2 = torch.nn.Linear(25, 25)
        self.dense_annotator = torch.nn.Linear(noisy_labels_no, 64)
        # self.dense_classes = torch.nn.Linear(noisy_labels_no, 50)
        self.norm = torch.nn.BatchNorm2d(80, affine=True)
        self.relu = torch.nn.Softplus()
        self.act = torch.nn.Softmax(dim=3)

    def forward(self, A_id, x):
        y = self.conv_2(self.conv_1(x))
        A_id = self.relu(self.dense_annotator(A_id))  # B, F_A
        A_id = A_id.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 512, 512)

        y = torch.cat((A_id, y), dim=1)
        y = self.norm(y)
        y = y.permute(0, 2, 3, 1)
        y = self.relu((self.dense(y)))
        y = self.dense2(y)
        
        y = self.relu(y.view(-1, 512, 512, self.class_no, self.class_no))
        y = y.view(-1, 512, 512, self.class_no ** 2).permute(0,3,1,2)

        return y
    
class Crowd_segmentationModel(torch.nn.Module):
    """ This defines the architecture of the chosen CR method
    """
    def __init__(self, noisy_labels):
        super().__init__()
        config = utils.globals.config
        self.seg_model = create_segmentation_backbone()
        self.activation = torch.nn.Softmax(dim=1)
        self.noisy_labels_no = len(noisy_labels)
        print("Number of annotators (model): ", self.noisy_labels_no)
        self.class_no = config['data']['class_no']
        self.crowd_type = config['model']['crowd_type']
        if self.crowd_type == 'global':
            print("Global crowdsourcing")
            self.crowd_layers = global_CM(self.class_no, 512, 512, self.noisy_labels_no)

        elif self.crowd_type == 'image': 
            print("Image dependent crowdsourcing")
            self.crowd_layers = image_CM(self.class_no, 512, 512, self.noisy_labels_no)
        elif self.crowd_type == 'pixel':
            print("Pixel dependent crowdsourcing")
            self.crowd_layers = cm_layers(in_channels=16, norm='in',
                                                  class_no=config['data']['class_no'], noisy_labels_no=self.noisy_labels_no)  
        self.activation = torch.nn.Softmax(dim=1)

    def forward(self, x, A_id=None):
        cm = None
        x = self.seg_model.encoder(x)
        x = self.seg_model.decoder(*x)
        if A_id is not None:
            cm = self.crowd_layers(A_id, x)
        x = self.seg_model.segmentation_head(x)
        y = self.activation(x)
        return y, cm

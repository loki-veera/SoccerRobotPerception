import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

resnet_model = torch.hub.load("pytorch/vision:v0.6.0", "resnet18", pretrained=True)
resnet_model.eval()


class nimbrRoNet2(nn.Module):
    def __init__(self):
        resnet_model = torch.hub.load(
            "pytorch/vision:v0.6.0", "resnet18", pretrained=True
        )
        super(nimbrRoNet2, self).__init__()
        self.resnet_layer1 = nn.Sequential(
            resnet_model.conv1,
            resnet_model.bn1,
            resnet_model.relu,
            resnet_model.maxpool,
            resnet_model.layer1,
        )
        self.resnet_layer2 = resnet_model.layer2
        self.resnet_layer3 = resnet_model.layer3
        self.resnet_layer4 = resnet_model.layer4

        self.singleconv1 = nn.Conv2d(64, 128, 1, bias=False)
        self.singleconv2 = nn.Conv2d(128, 256, 1, bias=False)
        self.singleconv3 = nn.Conv2d(256, 256, 1, bias=False)

        self.convtrans1 = nn.ConvTranspose2d(512, 256, 2, 2, bias=False)
        self.convtrans2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.ConvTranspose2d(512, 256, 2, 2, bias=False),
        )
        self.convtrans3 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.ConvTranspose2d(512, 128, 2, 2, bias=False),
        )

        self.relubn = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
        )

        self.detection_head = nn.Conv2d(256, 3, 1, bias=False)
        self.segment_head = nn.Conv2d(256, 3, 1, bias=False)
        self.loc_bias = nn.Parameter(
            torch.randn((1, 3, 480 // 4, 640 // 4), requires_grad=True)
        )

    def forward(self, input):
        x1 = self.resnet_layer1(input)
        x2 = self.resnet_layer2(x1)
        x3 = self.resnet_layer3(x2)
        rnet_output = self.resnet_layer4(x3)

        i1 = self.singleconv1(x1)
        i2 = self.singleconv2(x2)
        i3 = self.singleconv3(x3)

        o1 = self.convtrans1(rnet_output)
        o1 = torch.cat((o1, i3), 1)
        o2 = self.convtrans2(o1)
        o2 = torch.cat((o2, i2), 1)
        o3 = self.convtrans3(o2)
        o3 = torch.cat((o3, i1), 1)
        o3 = self.relubn(o3)

        seg_output = self.segment_head(o3) + self.loc_bias
        dec_output = self.detection_head(o3) + self.loc_bias

        return seg_output, dec_output
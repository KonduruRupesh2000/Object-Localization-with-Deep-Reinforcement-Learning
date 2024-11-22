import torch.nn as nn
import torchvision


class FeatureExtractor(nn.Module):
    def __init__(self, network='vgg16'):
        super(FeatureExtractor, self).__init__()
        if network == 'vgg16':
            model = torchvision.models.vgg16(pretrained=True)
        elif network == 'resnet50':
            model = torchvision.models.resnet50(pretrained=True)
        else:
            model = torchvision.models.alexnet(pretrained=True)
        model.eval() # to not do dropout
        self.features = list(model.children())[0] 
        self.classifier = nn.Sequential(*list(model.classifier.children())[:-2])
    def forward(self, x):
        x = self.features(x)
        return x
    
class DQN(nn.Module):
    def __init__(self, h, w, outputs, history_length):
        super(DQN, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features= outputs * history_length + 25088, out_features=1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear( in_features= 1024, out_features=1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear( in_features= 1024, out_features=outputs)
        )
    def forward(self, x):
      return self.classifier(x)


import torch
import torch.nn as nn
import torchvision
from torchvision.models import vit_b_16, ViT_B_16_Weights

class FeatureExtractor(nn.Module):
    def __init__(self, network='vit'):
        super(FeatureExtractor, self).__init__()
        if network == 'vit':
            # Load pre-trained ViT model
            self.model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
            # Remove the classification head
            self.model.heads = nn.Identity()
        elif network == 'vgg16':
            model = torchvision.models.vgg16(pretrained=True)
            self.model = nn.Sequential(*list(model.features.children()))
        elif network == 'resnet50':
            model = torchvision.models.resnet50(pretrained=True)
            self.model = nn.Sequential(*list(model.children())[:-2])
        else:
            model = torchvision.models.alexnet(pretrained=True)
            self.model = nn.Sequential(*list(model.features.children()))
        
        self.model.eval()

    def forward(self, x):
        return self.model(x)

# class DQN(nn.Module):
#     def __init__(self, h, w, outputs, history_length):
#         super(DQN, self).__init__()
#         # For ViT, the output feature size is 768
#         vit_feature_size = 768
#         self.classifier = nn.Sequential(
#             nn.Linear(in_features=outputs * history_length + vit_feature_size, out_features=1024),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(in_features=1024, out_features=1024),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(in_features=1024, out_features=outputs)
#         )

#     def forward(self, x):
#         return self.classifier(x)
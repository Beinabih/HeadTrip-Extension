import torch
from torchvision.models import resnet50, ResNet50_Weights, vgg19, VGG19_Weights, densenet121, DenseNet121_Weights
import argparse
import os

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--path',type=str, default="",
                    help='the path to the webUI')

args = parser.parse_args()

if args.path:
    print('Save VGG19 to model folder')
    torch.save(vgg19(weights=VGG19_Weights.IMAGENET1K_V1), os.path.join(args.path, 'headtrip' ,'vgg19.pt'))

    print('Save DenseNet121 to model folder')
    torch.save(densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1), os.path.join(args.path, 'headtrip' ,'densenet.pt'))
    print('Done')
else:
    print('Please specify the path argument --path to the model folder')
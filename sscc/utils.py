import torch
from sscc.models.lda import LDA

from sscc.models.supervised import Supervised
from sscc.models.scm import SCM
from sscc.models.pseudolabel import PseudoLabel
from sscc.models.ccm import CCM

from sscc.architectures.lenet import LeNet
from sscc.architectures.vgg import VGG
from sscc.architectures.resnet import ResNet
from sscc.architectures.resnet18 import resnet18, ClusteringModel
from sscc.architectures.wideresnet import WideResNet
from sscc.architectures.resnet18_small import ResNet18

models = {'supervised': Supervised,
          'scm': SCM,
          'ccm': CCM,
          'pseudolabel': PseudoLabel,
          'unsupervised': LDA}

architectures = {'lenet': LeNet,
                 'vgg': VGG,
                 'resnet': ResNet,
                 'wideresnet': WideResNet, 
                 'resnet18_small': ResNet18}

def parse_architecture_config(config):
    arch_params = config.get('model_params').get('architecture')
    #TODO: integrate the gansbeke resnet18 more slick
    if arch_params['arch'] == 'resnet18':
        architecture = get_scan_resnet(dataset=config['exp_params']['dataset'], **arch_params)
    else:
        architecture = architectures[arch_params.get('arch')](**arch_params)

    return architecture

def parse_model_config(config):
    model_param = config.get('model_params')
    model = models[model_param['model']]

    # architecture = parse_architecture_config(config)
    # model_instance = model(architecture, model_param['loss'])
    model_instance = model()

    return model_instance

def update_config(config, args):
    for name, value in vars(args).items():
        if value is None:
            continue

        for key in config.keys():
            if config[key].__contains__(name):
                config[key][name] = value

    return config

def get_scan_resnet(num_classes: int=10, freeze: bool=False, pretrained: bool=False, dataset: str ='cifar10', **kwargs):
    #TODO: integrate model weight download via gdown from Wouter's gdrive here
    if dataset.startswith('cifar'): dataset = dataset.split('r')[0] + 'r-' + dataset.split('cifar')[1]
    # Get backbone
    backbone = resnet18()
    # Setup
    model = ClusteringModel(backbone=backbone, nclusters=num_classes, nheads=1)
    # Load pretrained weights    
    if pretrained:
        pretrain_path=f'./data/model_weights/selflabel_{dataset}.pth.tar'
        state_dict = torch.load(pretrain_path, map_location='cpu')
        model.load_state_dict(state_dict)
        # freeze all layers but the head for the linear evaluation protocol
        if freeze:
            for name, param in model.named_parameters():
                if not name.startswith('cluster_head'): param.requires_grad = False
    return model

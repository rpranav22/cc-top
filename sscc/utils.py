import torch
from sscc.models.bert_kmeans import BERTKmeans
from sscc.models.constrained_clustering import ConstrainedClustering
from sscc.models.lda import LDA
from sscc.models.supervised_plm import SupervisedPLM
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
from sscc.architectures.bertsequenceclassification import BertForClassification


models = {'supervised': Supervised,
          'scm': SCM,
          'ccm': CCM,
          'pseudolabel': PseudoLabel,
          'unsupervised': LDA,
          'supervised_plm': SupervisedPLM,
          'constrained_clustering': ConstrainedClustering,
          'topic_discovery': ConstrainedClustering,
          'bert_kmeans': BERTKmeans}

architectures = {'lenet': LeNet,
                 'vgg': VGG,
                 'resnet': ResNet,
                 'wideresnet': WideResNet, 
                 'resnet18_small': ResNet18,
                 'bert_classifier': BertForClassification}

def parse_architecture_config(config):
    arch_params = config.get('model_params').get('architecture')
    #TODO: integrate the gansbeke resnet18 more slick
    if arch_params['arch'] == 'bert_kmeans':
        return None
    architecture = architectures[arch_params.get('arch')](**arch_params)

    return architecture

def parse_model_config(config):
    model_param = config.get('model_params')
    exp_params = config.get('exp_params')
    model_param['architecture']['max_length'] = exp_params['max_length']
    model = models[model_param['model']]
    architecture = parse_architecture_config(config)
    model_instance = model(architecture, model_param['loss'], **model_param['architecture'])
    print(f"loss: {model_param['loss']}")
    # model_instance = model()

    return model_instance

def update_config(config, args):
    for name, value in vars(args).items():
        if value is None:
            continue

        for key in config.keys():
            for k in config[key].keys():
                if isinstance(config[key][k], dict):
                    print('config key', config[key][k])
                    if config[key][k].__contains__(name):
                        config[key][k][name] = value
                        continue

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

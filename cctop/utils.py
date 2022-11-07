from cctop.models.bert_kmeans import BERTKmeans
from cctop.models.constrained_clustering import ConstrainedClustering
from cctop.models.lda import LDA
from cctop.models.supervised_plm import SupervisedPLM
from cctop.architectures.bertsequenceclassification import BertForClassification


models = {'unsupervised': LDA,
          'supervised_plm': SupervisedPLM,
          'constrained_clustering': ConstrainedClustering,
          'topic_discovery': ConstrainedClustering,
          'bert_kmeans': BERTKmeans}

architectures = {'bert_classifier': BertForClassification}

def parse_architecture_config(config):
    arch_params = config.get('model_params').get('architecture')
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

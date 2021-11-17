import torch


# set learning rate for different group of parameter
def get_parameters(model, model_init_lr, multiplier):
    """
    :param model_init_lr: original learning rate
    :param multiplier: factor defines how fast should learing rate decrease
    :return: params: list of parameters with different learing rate
    """
    params = []
    lr = model_init_lr

    # parameters of classifier
    classifier_params = {
        'params': [p for n, p in model.named_parameters() if 'bert' not in n],
        'lr': model_init_lr
    }
    params.append(classifier_params)

    # parameters of encoder in roberta, decrease in centain rate
    for layer in range(12, -1, -1):
        layer_params = {
            'params': [p for n, p in model.named_parameters() if f'encoder.layer.{layer}.' in n],
            'lr': lr
        }
        params.append(layer_params)
        lr *= multiplier

    # parameter in embedding layer
    other_params = {
        'params': [p for n, p in model.named_parameters() if 'embeddings' in n],
        'lr': lr
    }
    params.append(other_params)

    return params

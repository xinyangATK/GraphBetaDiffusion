import torch
import random
import numpy as np



def load_seed(seed):
    # Random Seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return seed


def load_device():
    if torch.cuda.is_available():
        device = list(range(torch.cuda.device_count()))
    else:
        device = 'cpu'
    return device


def load_model_optimizer(params, config_train, device):
    model = load_model(params)
    if isinstance(device, list):
        if len(device) > 1:
            model = torch.nn.DataParallel(model, device_ids=device)
        model = model.to(f'cuda:{device[0]}')
    if config_train.optimizer=='adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=config_train.lr, amsgrad=True,
                                        weight_decay=config_train.weight_decay)
    else:
        raise NotImplementedError(f'Optimizer:{config_train.optimizer} not implemented.')
    scheduler = None
    if config_train.lr_schedule:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config_train.lr_decay)
    return model, optimizer, scheduler


def load_batch(batch, device):
    device_id = f'cuda:{device[0]}' if isinstance(device, list) else device
    x_b = batch[0].to(device_id)
    adj_b = batch[1].to(device_id)
    return x_b, adj_b



    return sampling_fn

def load_model_params(config):
    input_dims = {'X': config.data.max_feat_num, 'E': config.model.input_dims.E, 
                    'y': config.model.input_dims.y+1} # +1 for time feature
    output_dims = {'X': config.data.max_feat_num, 'E': 1, 'y': 0}
    params = {'model_type': config.model.type, 'n_layers': config.model.num_layers,  
                'hidden_mlp_dims': config.model.hidden_mlp_dims,
                'hidden_dims': config.model.hidden_dims, 
                'input_dims': input_dims, 'output_dims': output_dims}
    return params

def load_ckpt(config, device):
    device_id = f'cuda:{device[0]}' if isinstance(device, list) else device
    path = f'checkpoints/{config.data.data}/{config.ckpt}.pth'
    ckpt_dict = torch.load(path, map_location=device_id)
    print(f'{path} loaded')
    return ckpt_dict


def load_opt_from_ckpt(config_train, state_dict, model):
    if config_train.optimizer=='adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=config_train.lr, amsgrad=True,
                                        weight_decay=config_train.weight_decay)
    else:
        raise NotImplementedError(f'Optimizer:{config_train.optimizer} not implemented.')
    optimizer.load_state_dict(state_dict)
    return optimizer

def load_eval_settings(data, kernel='emd'):
    # Settings for general graph generation
    methods = ['degree', 'cluster', 'orbit', 'spectral', 'connected'] 
    kernels = {'degree': kernel, 
                'cluster': kernel, 
                'orbit': kernel,
                'spectral': kernel}
    if data == 'sbm':
        try:
            import graph_tool.all as gt
            methods.append('eval_sbm')
        except:
            pass
    elif data == 'planar':
        methods.append('eval_planar')
    return methods, kernels
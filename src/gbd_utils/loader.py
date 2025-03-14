import os, sys
import math
import torch
import pickle
from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint
from gdss_utils.utils.ema import ExponentialMovingAverage, LitEma


dataset_space = ['comm20', 'ego', 'qm9', 'zinc250k', 'planar', 'sbm']

def load_model(cfg, **model_kwargs):
    dataset = cfg.dataset.name
    
    if dataset in ['comm20', 'ego', 'planar', 'sbm']:
        from graph_beta_diffusion_general import GraphBetaDiffusion
    elif dataset in ['qm9', 'zinc250k']:
        from graph_beta_diffusion_molecule import GraphBetaDiffusion
    else:
        print(f'Invalid dataset')
        assert dataset in dataset_space

    model = GraphBetaDiffusion(cfg=cfg, **model_kwargs)

    return model

def load_model_from_ckpt(cfg, **model_kwargs):
    dataset = cfg.dataset.name
    resume = cfg.general.test_only

    if dataset in ['comm20', 'ego', 'planar', 'sbm']:
        from graph_beta_diffusion_general import GraphBetaDiffusion
    elif dataset in ['qm9', 'zinc250k']:
        from graph_beta_diffusion_molecule import GraphBetaDiffusion
    else:
        print(f'Invalid dataset')
        assert dataset in dataset_space

    model = GraphBetaDiffusion(cfg=cfg, **model_kwargs)

    model = GraphBetaDiffusion.load_from_checkpoint(resume, **model_kwargs, map_location='cpu')

    return model

def build_ema(model, decay=0.999):
    '''Build model_ema model'''
    ema = LitEma(model, decay=decay)
    # ema = ExponentialMovingAverage(model.parameters(), decay=decay)
    return ema

def load_ema(ema_ckpt_path):
    '''Load model_ema model'''
    model_ema = torch.load(ema_ckpt_path)
    return model_ema


def save_ema(model_ema, current_epoch, ckpt_dir):
    '''Save model_ema model'''
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    torch.save(model_ema, os.path.join(ckpt_dir, f'{current_epoch}_ema.pth'))


def load_general_graph_list(dataset):
        current_file = Path(__file__).resolve()
        raw_dir = f"{current_file.parents[2]}/data/"

        if dataset in ['comm20', 'ego']:
            pkl_name = 'community_small' if dataset == 'comm20' else 'ego'
            file_path = os.path.join(raw_dir, f'{pkl_name}.pkl')
            print(file_path)
            with open(file_path, 'rb') as f:
                graph_list = pickle.load(f)
            test_size = int(0.2 * len(graph_list))
            train_graph_list, test_graph_list = graph_list[test_size:], graph_list[:test_size]
        elif dataset in ['sbm', 'planar']:
            file_path = os.path.join(raw_dir, f'{pkl_name}.pkl')
            with open(file_path, 'rb') as f:
                train_graph_list, val_graph_list, test_graph_list = pickle.load(f)
            train_graph_list, test_graph_list = train_graph_list, test_graph_list
        else:
            raise ValueError('wrong dataset name while loading')
        
        return train_graph_list, test_graph_list

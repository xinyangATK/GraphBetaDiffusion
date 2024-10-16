import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from GDSS_utils.utils.ema import ExponentialMovingAverage, LitEma


dataset_space = ['gdss-comm20', 'gdss-ego', 'qm9', 'zinc250k', 'planar']
def load_model(cfg, **model_kwargs):
    dataset = cfg.dataset.name
    
    if dataset == 'gdss-comm20':
        from beta_diffusion_model_comm20_eta import BetaDiffusion
    elif dataset == 'gdss-ego':
        from beta_diffusion_model_ego_eta import BetaDiffusion
    elif dataset == 'qm9':
        from src.beta_diffusion_model_qm9_eta import BetaDiffusion
    elif dataset == 'zinc250k':
        from beta_diffusion_model_zinc250k_eta import BetaDiffusion
    elif dataset == 'planar':
        from beta_diffusion_model_planar_eta import BetaDiffusion
    elif dataset == 'sbm':
        from beta_diffusion_model_sbm_eta import BetaDiffusion
    else:
        print(f'Invalid dataset')
        assert dataset in dataset_space

    model = BetaDiffusion(cfg=cfg, **model_kwargs)

    return model

def load_model_from_ckpt(cfg, **model_kwargs):
    dataset = cfg.dataset.name

    resume = cfg.general.test_only

    if dataset == 'gdss-comm20':
        from beta_diffusion_model_comm20_eta import BetaDiffusion
    elif dataset == 'gdss-ego':
        from beta_diffusion_model_ego_eta import BetaDiffusion
    elif dataset == 'qm9':
        from src.beta_diffusion_model_test import BetaDiffusion
    elif dataset == 'zinc250k':
        from beta_diffusion_model_zinc250k_eta import BetaDiffusion
    elif dataset == 'planar':
        from beta_diffusion_model_planar_eta import BetaDiffusion
    elif dataset == 'sbm':
        from beta_diffusion_model_sbm_eta import BetaDiffusion
    else:
        print(f'Invalid dataset')
        assert dataset in dataset_space

    model = BetaDiffusion.load_from_checkpoint(resume, **model_kwargs, map_location='cpu')

    return model

def load_ema(model, decay=0.999):
    ema = LitEma(model, decay=decay)
    # ema = ExponentialMovingAverage(model.parameters(), decay=decay)
    return ema

def load_ema_from_ckpt(model, ema_state_dict, decay=0.999):
    ema = ExponentialMovingAverage(model.parameters(), decay=decay)
    ema.load_state_dict(ema_state_dict)
    return ema



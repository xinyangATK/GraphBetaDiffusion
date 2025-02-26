# import graph_tool as gt
import logging
import os
import pathlib
import warnings

import pytorch_lightning.loggers
import torch
torch.cuda.empty_cache()
import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.warnings import PossibleUserWarning

from src import utils
from metrics.abstract_metrics import TrainAbstractMetricsDiscrete, TrainAbstractMetrics

from diffusion_model import LiftedDenoisingDiffusion
from diffusion_model_discrete import DiscreteDenoisingDiffusion
from diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
from beta_diffusion_model_comm20_eta import BetaDiffusion
# from beta_diffusion_model_ego_eta import BetaDiffusion
import torch.distributed as dist
from src.evaluation.evaluate import EVAL_METRICS
#
# os.environ["RANK"] = "0"
# os.environ["WORLD_SIZE"] = "1"
# os.environ["MASTER_ADDR"] = "localhost"
# os.environ["MASTER_PORT"] = "5678"
# torch.distributed.init_process_group("nccl")

warnings.filterwarnings("ignore", category=PossibleUserWarning)


def get_resume(cfg, model_kwargs):
    """ Resumes a run. It loads previous config without allowing to update keys (used for testing). """
    saved_cfg = cfg.copy()
    name = cfg.general.name + '_resume'
    resume = cfg.general.test_only
    if cfg.model.type == 'discrete':
        model = BetaDiffusion.load_from_checkpoint(resume, **model_kwargs, map_location='cpu')
    else:
        model = LiftedDenoisingDiffusion.load_from_checkpoint(resume, **model_kwargs)

    new_cfg = model.cfg

    for category in cfg:
        for arg in cfg[category]:
            if arg not in new_cfg[category]:
                continue
            else:
                new_cfg[category][arg] = cfg[category][arg]

    cfg.general.test_only = resume
    new_cfg.general.name = new_cfg.general.name + '_resume'

    new_cfg = utils.update_config_with_new_keys(new_cfg, saved_cfg)

    return new_cfg, model


def get_resume_adaptive(cfg, model_kwargs):
    """ Resumes a run. It loads previous config but allows to make some changes (used for resuming training)."""
    saved_cfg = cfg.copy()
    # Fetch path to this file to get base path
    current_path = os.path.dirname(os.path.realpath(__file__))
    root_dir = current_path.split('outputs')[0]

    resume_path = os.path.join(root_dir, cfg.general.resume)

    if cfg.model.type == 'discrete':
        model = BetaDiffusion.load_from_checkpoint(resume_path, **model_kwargs)
    else:
        model = LiftedDenoisingDiffusion.load_from_checkpoint(resume_path, **model_kwargs)
    new_cfg = model.cfg

    for category in cfg:
        for arg in cfg[category]:
            new_cfg[category][arg] = cfg[category][arg]

    new_cfg.general.resume = resume_path
    new_cfg.general.name = new_cfg.general.name + '_resume'

    new_cfg = utils.update_config_with_new_keys(new_cfg, saved_cfg)
    return new_cfg, model


@hydra.main(version_base='1.2', config_path='../configs/experiment/', config_name='gdss_ego.yaml')
def main(cfg: DictConfig):
    # cfg['general']['test_only'] = '/mnt/887fa118-38b2-4d6c-8658-17e3842e9651/lxy/GraphGen/BetaGraphGen_eta/outputs/gdss-comm20/2024-03-18/09-55-51-graph-tf-model/checkpoints/graph-tf-model/epoch=599999-v1.ckpt'
    # cfg['general']['test_only'] = '/mnt/887fa118-38b2-4d6c-8658-17e3842e9651/lxy/GraphGen/BetaGraphGen_eta/outputs/gdss-ego/2024-03-21/20-43-04-graph-tf-model/checkpoints/graph-tf-model/epoch=299999-v1.ckpt'

    # Mol
    # cfg['general']['test_only'] = '/mnt/944dbd2c-4b04-4b73-9d04-b3ecfffb1a8e/GraphGen/BetaDiGress/outputs/2023-12-22/23-29-11-graph-tf-model/checkpoints/graph-tf-model/epoch=999.ckpt'
    # cfg['general']['test_only'] = '/mnt/d1f47cf6-2a89-4f9d-97ff-1b3fa416e887/lxy/BetaGraphGen/outputs/qm9new/2024-01-11/17-38-01-graph-tf-model/checkpoints/graph-tf-model/epoch=2799.ckpt'
    dataset_config = cfg["dataset"]
    if dataset_config["name"] in ['sbm', 'comm20', 'planar', 'gdss-comm20', 'gdss-ego']:
        from datasets.spectre_dataset import SpectreGraphDataModule, SpectreDatasetInfos
        from analysis.spectre_utils import PlanarSamplingMetrics, SBMSamplingMetrics, Comm20SamplingMetrics
        from analysis.visualization import NonMolecularVisualization
        from src.datasets.generic_dataset import GdssSpectreGraphDataModule, GdssSpectreDatasetInfos

        if dataset_config['name'] in ['gdss-comm20', 'gdss-ego']:
            datamodule = GdssSpectreGraphDataModule(cfg)
            dataset_infos = GdssSpectreDatasetInfos(datamodule, dataset_config)
        else:
            datamodule = SpectreGraphDataModule(cfg)
            dataset_infos = SpectreDatasetInfos(datamodule, dataset_config)

        if dataset_config['name'] == 'sbm':
            sampling_metrics = SBMSamplingMetrics(datamodule)
        elif dataset_config['name'] in ['comm20', 'gdss-comm20']:
            sampling_metrics = Comm20SamplingMetrics(datamodule)
        else:
            sampling_metrics = PlanarSamplingMetrics(datamodule)

        # dataset_infos = SpectreDatasetInfos(datamodule, dataset_config)
        visualization_tools = NonMolecularVisualization()

        extra_features = DummyExtraFeatures()
        domain_features = DummyExtraFeatures()

        dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features,
                                                    domain_features=domain_features, cfg=cfg)

        model_kwargs = {'dataset_infos': dataset_infos,
                        'sampling_metrics': sampling_metrics, 'visualization_tools': visualization_tools,
                        'extra_features': extra_features, 'domain_features': domain_features}
    else:
        raise NotImplementedError("Unknown dataset {}".format(cfg["dataset"]))

    if cfg.general.test_only:
        # When testing, previous configuration is fully loaded
        cfg, _ = get_resume(cfg, model_kwargs)
        os.chdir(cfg.general.test_only.split('checkpoints')[0])
    elif cfg.general.resume is not None:
        # When resuming, we can override some parts of previous configuration
        cfg, _ = get_resume_adaptive(cfg, model_kwargs)
        os.chdir(cfg.general.resume.split('checkpoints')[0])

    utils.create_folders(cfg)

    model = BetaDiffusion(cfg=cfg, **model_kwargs)


    callbacks = []
    if cfg.train.save_model:
        checkpoint_callback = ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}",
                                              filename='{epoch}',
                                              save_top_k=-1,
                                              every_n_epochs=100)
        last_ckpt_save = ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}", filename='{epoch}', every_n_epochs=1)
        callbacks.append(last_ckpt_save)
        callbacks.append(checkpoint_callback)

    if cfg.general.process_visualization:
        model.process_visualization(datamodule, given_t_split=50)
        return



    name = cfg.general.name
    if name == 'debug':
        print("[WARNING]: Run is called 'debug' -- it will run with fast_dev_run. ")

    # use_gpu = cfg.general.gpus > 0 and torch.cuda.is_available()
    trainer = Trainer(gradient_clip_val=cfg.train.clip_grad,
                      strategy="ddp_find_unused_parameters_true",  # Needed to load old checkpoints
                      accelerator='gpu',  # if use_gpu else 'cpu',
                      devices=cfg.general.gpus,  # if use_gpu else 1,
                      max_epochs=cfg.train.n_epochs,
                      check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
                      fast_dev_run=cfg.general.name == 'debug',
                      enable_progress_bar=False,
                      callbacks=callbacks,
                      log_every_n_steps=50 if name != 'debug' else 1,
                      logger = [])

    if not cfg.general.test_only:
        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.general.resume)
        if cfg.general.name not in ['debug', 'test']:
            trainer.test(model, datamodule=datamodule)
    else:
        # Start by evaluating test_only_path
        trainer.test(model, datamodule=datamodule, ckpt_path=cfg.general.test_only)
        if cfg.general.evaluate_all_checkpoints:
            directory = pathlib.Path(cfg.general.test_only).parents[0]
            print("Directory:", directory)
            files_list = os.listdir(directory)
            for file in files_list:
                if '.ckpt' in file:
                    ckpt_path = os.path.join(directory, file)
                    if ckpt_path == cfg.general.test_only:
                        continue
                    print("Loading checkpoint", ckpt_path)
                    trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == '__main__':
    main()

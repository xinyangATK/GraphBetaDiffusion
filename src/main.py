# import graph_tool as gt

import logging
import os, sys
import pathlib
import warnings
import pytorch_lightning.loggers
import torch
torch.cuda.empty_cache()
import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.warnings import PossibleUserWarning
from src import utils

from diffusion.extra_features import DummyExtraFeatures
from src.gbd_utils.loader import load_model, load_model_from_ckpt
from src.evaluation.evaluate import EVAL_METRICS


warnings.filterwarnings("ignore", category=PossibleUserWarning)


def get_resume(cfg, model_kwargs):
    """ Resumes a run. It loads previous config without allowing to update keys (used for testing). """
    saved_cfg = cfg.copy()
    name = cfg.general.name + '_resume'
    resume = cfg.general.test_only

    model = load_model_from_ckpt(cfg, **model_kwargs)

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


@hydra.main(version_base='1.3', config_path='../configs/experiment/', config_name='gdss-comm20.yaml')
def main(cfg: DictConfig):
    dataset_config = cfg["dataset"]
    if dataset_config["name"] in ['sbm', 'planar', 'gdss-comm20', 'gdss-ego']:
        from analysis.visualization import NonMolecularVisualization
        from src.datasets.generic_dataset import GenericGraphDataModule, GenericDatasetInfos

        datamodule = GenericGraphDataModule(cfg)
        dataset_infos = GenericDatasetInfos(datamodule, dataset_config)

        visualization_tools = NonMolecularVisualization()

        extra_features = DummyExtraFeatures()
        domain_features = DummyExtraFeatures()

        dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features,
                                                    domain_features=domain_features, cfg=cfg)

        model_kwargs = {'dataset_infos': dataset_infos, 'visualization_tools': visualization_tools,
                        'extra_features': extra_features, 'domain_features': domain_features}
    
    elif dataset_config["name"] in ['qm9', 'zinc250k']:
            from analysis.visualization import MolecularVisualization

            if dataset_config["name"] == 'qm9':
                from src.datasets import qm9_dataset
                datamodule = qm9_dataset.QM9DataModule(cfg)
                dataset_infos = qm9_dataset.QM9infos(datamodule=datamodule, cfg=cfg)

            elif dataset_config['name'] == 'zinc250k':
                from datasets import zinc250k_dataset
                datamodule = zinc250k_dataset.ZINC250KDataModule(cfg)
                dataset_infos = zinc250k_dataset.ZINC250Kinfos(datamodule=datamodule, cfg=cfg)
            else:
                raise ValueError("Dataset not implemented")

            extra_features = DummyExtraFeatures()
            domain_features = DummyExtraFeatures()

            dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features,
                                                        domain_features=domain_features, cfg=cfg)

            visualization_tools = MolecularVisualization(cfg.dataset.remove_h, dataset_infos=dataset_infos)
            eval_metrics = EVAL_METRICS(dataset='QM9')

            model_kwargs = {'dataset_infos': dataset_infos, 'visualization_tools': visualization_tools,
                            'extra_features': extra_features, 'domain_features': domain_features,
                            'eval_metrics': eval_metrics,
                            }
    else:
        raise NotImplementedError("Unknown dataset {}".format(cfg["dataset"]))

    if cfg.general.test_only:
        # When testing, previous configuration is fully loaded
        cfg, _ = get_resume(cfg, model_kwargs)
        os.chdir(cfg.general.test_only.split('checkpoints')[0])

    utils.create_folders(cfg)

    model = load_model(cfg=cfg, **model_kwargs)

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

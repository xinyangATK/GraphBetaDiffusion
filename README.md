# [Advancing Graph Generation through Beta Diffusion](https://arxiv.org/abs/2406.09357) (ICLR 2025)

By [Xinyang Liu](https://xinyangatk.github.io)\*<sup>1</sup>, [Yilin He](https://scholar.google.com/citations?user=oJu3WzsAAAAJ&hl=en)\*<sup>1</sup>, [Bo Chen](https://web.xidian.edu.cn/bchen/)<sup>2</sup>, [Mingyuan Zhou](https://mingyuanzhou.github.io)<sup>1</sup>
\
<sup>1</sup>The University of Texas at Austin, <sup>2</sup>Xidian University
\
\*Equal contribution

<a href='https://arxiv.org/pdf/2406.09357'><img alt="Static Badge" src="https://img.shields.io/badge/Paper-arXiv-red"></a>


![Overview of GBD](assets/model.png)

We introduce **GBD**, a **G**raph **B**eta **D**iffusion that specifically designed to handle the diverse nature of graph data, including both continuous and discrete graph elements. 
Additionally, we propose a modulation technique that enhances the realism of generated graphs by stabilizing critical graph topology while maintaining flexibility for other components. 
In doing so, GBD competes strongly with existing models across multiple general and biochemical graph benchmarks. 

In this repo, we will release (**updating**):

- Environment Dependencies &#x2705;
- Checkpoints &#x1F4CC;
- Reproducing Experiments &#x2705;
  - Dataset preparations 
  - Configurations 
  - Training 
  - Generation and Evaluation
- Visualization &#x2705;


<a name="environment dependencies"></a>
## Environment Dependencies
Our code was built on [DiGress](https://arxiv.org/abs/2209.14734), please refer to the [Environment installation](https://github.com/cvignac/DiGress?tab=readme-ov-file#environment-installation) in **DiGress**.


<a name="reproducing experiments"></a>
## Reproducing Experiments

### 1. Dataset preparations
We provide two commands for generating general graph datasets as follows:

```
python src/datasets/data_generators_gdss.py --dataset <dataset> 
```
where `<dataset>` is one of the general graph datasets: `community_small`, `ego_small` and `grid`. This will create the `<dataset>.pkl` file in the data directory.

```
python src/datasets/data_generators_grum.py --dataset <dataset> --mmd
```
where `<dataset>` is one of the general graph datasets: `planar` and `sbm`. This will create the `<dataset>.pkl` file in the data directory.

For the molecule datasets:
```
python src/datasets/preprocess.py --dataset <dataset>
python src/datasets/preprocess_for_nspdk.py --dataset <dataset>
```
 where `<dataset>` is one of the 2d molecule datasets: `qm9` and `zinc250k`.

**Note that the `.pkl/.pt` files are saved in `./data/raw/<dataset>`**, if the directory is not found, please build the directory manually.

**Optional:**

For `Community-small`, `Ego-small` and `Grid` datasets, please refer to [GDSS](https://github.com/harryjo97/GDSS).\
For `Planar`, `SBM`, `QM9` and `ZINC250k` datasets, please refer to [GruM](https://github.com/harryjo97/GruM/tree/master/GruM_2D#1-dataset-preparations).

For the evaluation of general graph generation tasks, run the following command to compile the [ORCA program](http://www.biolab.si/supp/orca/orca.html).

```
cd src/analysis/orca 
g++ -O2 -std=c++11 -o orca orca.cpp
```

### 2. Configurations
The configurations are provided in the `config/` directory in `YAML` format. Hyperparameters used in the experiments are specified in the Appendix E of our paper.

### 3. Training 
Training GBD on general and biochemical graph benchmark:
```
python src/main.py +experiment=<dataset>.yaml general.gps=[0,1,2,3] 
```

Training GBD with **Concentration Modulation** technique:

- Set `concentration_m = True` in experiment configs for traing GBD with concentration modulation technique!
- We provide specialized concentration modulation strategies for different datasets in `concentration.py`. 

For the **new** dataset, custom concentration modulation strategy can be defined in a similar way with `CustomConcentrationModule` module.



### 4. Generation and Evaluation
![Overview of GBD](assets/sample.png)

Provide the `path` of checkpoints for `general.resume` or `general.test_only`, then run the follow command to generate samples.

```
python src/main.py +experiment=<dataset>.yaml
```
or

```
python src/main.py +experiment=<dataset>.yaml general.resume='to/your/path'
```

<a name="visualization"></a>
## Visualization
![Overview of GBD](assets/eta.png)

Edge generation process of GBD with **concentration modulation defined by node degree** on  `Community-small` dataset.

```
python main.py +experiment=<dataset>.yaml general.sample_visualization=True.
```

Set `general.forward_visualization=True` or `general.sample_visualization=True` in config file for visualization of forward process or reverse process by graph beta diffusion. **(currently only supports general graph)**

<a name="acknowledgements"></a>
## Acknowledgements
This repository was heavily built off of [DiGress](https://github.com/cvignac/DiGress), [GDSS](https://github.com/harryjo97/GDSS) and [GruM](https://github.com/harryjo97/GruM/tree/master/GruM_2D).

<a name="citation"></a>
## Citation
```
@article{liu2024advancing,
  title={Advancing Graph Generation through Beta Diffusion},
  author={Liu, Xinyang and He, Yilin and Chen, Bo and Zhou, Mingyuan},
  journal={arXiv preprint arXiv:2406.09357},
  year={2024}
}
```





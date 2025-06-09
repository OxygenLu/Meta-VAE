# Anomaly detection and localization in lung CT images using Meta-VAE implementation

Baseline: [Anomaly Detection for Medical Images Using Teacher-Student Model with Skip Connections and Multi-scale Anomaly Consistency](https://ieeexplore.ieee.org/document/10540605)

## Enviroment

```bash
conda create -n mvad python=3.11
conda activate mvad
pip install requirements.txt
```

## Usage:

    Run main.py to train and test the model.

## Introduction of the code:

    1. dataset.py for loading the training and testing dataset.

    2. encoder.py for the encoder of the model which is pretrained.

    3. decoder.py which is the opposite of the structure of the encoder.

    4. loss_function.py which is the loss function in this project.

    5. eval_func.py for some useful functions.

    6. main.py, main_raw.py and main_vae_pre.py for training and testing the model.

    7. app.py for visualisation.

    8. test.py and test_vae.py for evaluation.

## Acknowledgement

* This code is adapted from [Skip-TS](https://github.com/Arktis2022/Skip-TS) (baseline) and [RD4AD](https://github.com/hq-deng/RD4AD/tree/main)(visualisation).
* We thank for their elegant and efficient code base.

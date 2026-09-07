# Anchor-Guided Gradient Alignment for Incomplete Multimodal Learning

This repo is the official implementation of _Anchor-Guided Gradient Alignment for Incomplete Multimodal Learning_ accepted by CVPR 2026. 

## Abstract

Vision-language pre-training (VLP) has achieved remarkable performance across diverse multimodal learning (MML) tasks. Recently, many efforts have focused on reconstructing missing modalities to improve the adaptability of VLP models in incomplete MML scenarios. However, these approaches overlook the learning imbalance under severe missing-modality conditions, i.e., the optimization process is dominated by reconstructed samples, thereby weakening complete-sample representations. In this paper, we propose a novel ANchor-guided Gradient Alignment (ANGA) framework to address this issue. Specifically, we first retrieve similar instances to reconstruct the missing modalities, thereby alleviating information deficiency. We then introduce an entropy-driven curriculum that progressively incorporates reliable reconstructed samples together with complete ones to form an optimization anchor, which guides gradient alignment to mitigate learning imbalance. Furthermore, we design a semantic-enhanced adapter that leverages the retrieved instances to generate dynamic prompts, further enhancing the robustness of the VLP model. Extensive experiments on widely used datasets demonstrate the superiority of ANGA over state-of-the-art (SOTA) baselines across various missing-modality scenarios.

## Framework

<img width="1232" alt="image" src="framework.png" />

## Environment Configuration

First, create a new conda env for ANGA:

```shell
conda create -n ANGA python=3.9
```

Next, activate this env and install the dependencies from the requirements.txt:

```shell
conda activate ANGA

pip install -r requirements.txt
```

## Data Preparation

### MM-IMDb

First, download the dataset from this link: https://archive.org/download/mmimdb/mmimdb.tar.gz

Then, place the raw images in folder **dataset/mmimdb/image** and put the json files in folder **dataset/mmimdb/meta_data**.

### HateMemes

First, download the dataset from this link: https://www.kaggle.com/datasets/parthplc/facebook-hateful-meme-dataset

Then, place the raw images in folder **dataset/hatememes/image** and put the json files in folder **dataset/hatememes/metadata**.

Next, replace the **test.json** in metadata with **test_seen.json** downloaded from this link: https://www.kaggle.com/datasets/williamberrios/hateful-memes as the test.json downloaded from the prior website has no label information for evaluation. (Do not change other files, only replace the test.json with test_seen.json)

### Food101

First, download the dataset from this link: https://www.kaggle.com/datasets/gianmarco96/upmcfood101

Then, place the raw images in folder **dataset/food101/image** and put the csv files in folder **dataset/food101/meta_data**.

Note: The image folder follows the structure dataset/xxx/image/xxx.jpg|png|jpeg

## Code Running

### Dataset Initiation

Take the HateMemes dataset for an example, run the following command to init the dataset:

```shell
python src/init_data.py
```

### Training & Evaluation

Run the following command to training our model and evaluate the results:

```shell
python src/train.py
```

## Citation

If you find the code useful for your research, please give us a star ⭐⭐⭐ and consider citing:

```
@inproceedings{guan2026anchorguided,
    author = {Zhi{-}Hao Guan and
                  Longfei Huang and
                  Yang Yang},
    booktitle = {CVPR},
    year = {2026},
    title = {Anchor-Guided Gradient Alignment for Incomplete Multimodal Learning}
}
```

# Supplemental Code for "Colorful Insights from an AI *Khipukamayuq*"

The code and data in this repository can be used to reproduce the model training and analysis from Clindaniel (Under Review) "Colorful Insights from an AI *Khipukamayuq*." Consult the information below as a guide to navigating the repository.

## Environment Setup

The code was written in Python 3.10.14 in an Anaconda 2023.07-2 environment (Anaconda can be downloaded for your operating system [here](https://repo.anaconda.com/archive/)). The dependencies not included in the Anaconda distribution can be installed by running the following in the terminal (with the `requirements.txt` file included in this repository).

```
pip install -r requirements.txt
```

Once all the dependencies are installed, the replication code included in this repository can be run.

## Model Training

The code used to train the "AI khipukamayuq" presented in the manuscript is available in [`training.ipynb`](./training.ipynb) as well as [`hp_search.py`](./hp_search.py) (which runs an optimal hyperparameter search for the model trained in `training.ipynb`). The training and test data used to train and validate the model are included in the [`data/train/`](./data/train/) and [`data/test/`](./data/test/) directories in this repository. In order to replicate the training process, a GPU is recommended (the code in this repository was tested on a NVIDIA T4 GPU in Google Cloud, as well as a NVIDIA RTX 4060 laptop GPU). The pretrained model described in the manuscript is also available, though, in the [`pretrained-bert/`](./pretrained-bert/) directory.

## Data Analysis

Using the trained model, the code in [`analysis.ipynb`](./analysis.ipynb) generates contextual embeddings for each pendant/top cord color in the [Open Khipu Repository (OKR), v2.0.0](https://github.com/khipulab/open-khipu-repository/tree/v2.0.0) (i.e. quantitative representations of how each cord color is used in the context of other cords in a group). These cord color embeddings are then analyzed to identify semantic patterns in color usage in Inka-style khipus in the OKR, as reported in the manuscript. The pre-computed embeddings for each cord are available within the [`data/`](./data/) directory, grouped by color code. The average embeddings for each cord group are also included in the [`data/cg_embeddings`](./data/cg_embeddings/) directory and can be used to replicate the cord group "phrase" and khipu-level "topic" detection portions of the analysis. Finally, the models used for cord-level dimension reduction and clustering are available and their intermediate results are available in the [`clustering/`](./clustering/) directory.
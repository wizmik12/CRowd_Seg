# Learning from crowds for automated histopathological image segmentation

This repo presents the code of the crowdsourcing methods for segmentation of histopathological images. The models proposed are: **CR Global** and **CR Pixel** introduced in  *Crowdsourcing Segmentation of Histopathological Images Using Annotations Provided by Medical Students* and **CR Image** introduced in *Learning from crowds for automated histopathological image segmentation*.

#### Citation

~~~
@inproceedings{lopez2023crowdsourcing,
  title={Crowdsourcing Segmentation of Histopathological Images Using Annotations Provided by Medical Students},
  author={L{\'o}pez-P{\'e}rez, Miguel and Morales-{\'A}lvarez, Pablo and Cooper, Lee AD and Molina, Rafael and Katsaggelos, Aggelos K},
  booktitle={International Conference on Artificial Intelligence in Medicine},
  pages={245--249},
  year={2023},
  organization={Springer}
}
~~~

~~~
@article{lopez2024learning,
  title={Learning from crowds for automated histopathological image segmentation},
  author={L{\'o}pez-P{\'e}rez, Miguel and Morales-{\'A}lvarez, Pablo and Cooper, Lee AD and Felicelli, Christopher and Goldstein, Jeffery and Vadasz, Brian and Molina, Rafael and Katsaggelos, Aggelos K},
  journal={Computerized Medical Imaging and Graphics},
  pages={102327},
  year={2024},
  publisher={Elsevier}
}
~~~

## Data
 [link](https://drive.google.com/drive/folders/17VukoKpwZclRrDcWSK1aYd_lPeqWNM8N?usp=sharing=)
 
## Install Requirements
* Use Miniconda/Anaconda to install the requirements with `conda env create -f environment.yml`
* Activate the environment with `conda activate seg_crowd_env`
* For more information see www.anaconda.com

## Configuration
* To run the model with the dummy dataset, simply use python `src/main.py`
* For experiments there are three levels of configurations:
    1. The default config
    2. The dataset config
    3. The experiment config
* The configuration will be loaded in this order and parameters will be overwritten
* In the configuration, you can change all the hyperparameters of the models and select the desired experiment.
* How to define config paths:
    1. The default config: By argument `-dc [path/to/config.yaml]`
    2. The dataset config: In the default config `data: dataset_config: [path/to/dataset_config.yaml]`
    3. The experiment config: By changing the experiment folder `-ef [path/to/directory]`. Here a file `exp_config.yaml` is expected.
* Example: `python src/main.py -dc ../../experiments/segmentation_tnbc/config.yaml -ef ../../experiments/segmentation_tnbc/linknet`
* You can execute the best models by running `run_all.sh`

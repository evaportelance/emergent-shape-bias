# The Emergence of the Shape Bias Results from Communicative Efficiency

This repository contains the code for the paper:

@inproceedings{portelance2021,
  title={The Emergence of the Shape Bias Results from Communicative Efficiency},
  author={Portelance, Eva and Frank, Michael C. and Jurafsky, Dan and Sordoni, Alessandro and Laroche, Romain},
  booktitle={Proceedings of the 25th Conference on Computational Natural Language Learning (CoNLL)},
  year={2021}
}

## Requirements

You can clone the conda environment available for linux that is in the repository (see linux-conda-env.txt). You will then need to use pip to install a couple additional requirements (see pip-requirement.txt).

## Dataset

The dataset presented in the paper is available for download at this OSF link, please cite this paper if you use it:

https://osf.io/qf5h9/

## Perception and retrieving image feature vectors

The dataset files downloaded from the above link already contain the extracted feature vectors from our pretrained version of the AMDIM model (Bachman et al., 2019), but if you would like to train your own from scratch here is how.

#### Train AMDIM
We combined the images in the train and test directories of the dataset for this, so you will likely have to make a copy of the dataset to do this. Then go into the ./my_amdim directory and you can see the run.sh executable for an example of how to run this as well as the original readme file. It may take anywhere between 10hours and a day depending on your GPU setup.

#### Extract features from AMDIM for all test and train images
This can be done by running the extract_perception_features.py python script. See this file for flag details.


## Experiment 1

#### To train
To train agents in the setup used for experiment 1 use the following example commands. If running agents in the baseline condition with 100% random games and no communicative need for shape run:

python train.py --experiment-name "experiment1_baseline_seed1" --cuda --selfplay --ce-loss --perspective --n-epochs-init 14 --n-epochs-self 14 --n-generations 0 --batch-size 1024 --n-games 64000 --seed 1 --data-path "../game_data/clevr_shape_same/" --hidden-size 256 --hidden-mlp 256 --compression-size 512

Else if controlling the communicative need for shape, here with 40% shape games, run:

python train.py --experiment-name "experiment1_40shapegames_seed1" --cuda --selfplay --ce-loss --perspective --n-epochs-init 14 --n-epochs-self 14 --n-generations 0 --proportion 0.4 --images "shape" --batch-size 1024 --n-games 64000 --seed 1 --data-path "../game_data/clevr_shape_same/" --hidden-size 256 --hidden-mlp 256 --compression-size 512

#### To evaluate
Note that there will be tensorboard log files in your experiment directory once models start training that you can check to track the losses and accuracies of agents.

To retrieve a populations accuracy of test games for all training setups run:

python eval_allfeatures.py --load-name "experiment1_baseline_seed1" --cuda --selfplay --ce-loss --perspective --n-epochs-init 14 --n-epochs-self 14 --n-generations 0 --batch-size 512 --n-games 64000 --seed 1 --data-path "../game_data/clevr_shape_same/" --hidden-size 256 --hidden-mlp 256 --compression-size 512 --csv-general-eval-file "exp1_test_acc.csv"

To retrieve mutual information estimates between agent linguistic representations and labels for all training setups run:

python eval_mi.py --load-name "experiment1_baseline_seed1" --selfplay --ce-loss --perspective --n-epochs-init 100 --n-generations 0 --batch-size 200 --n-games 64000 --hidden-size 256 --hidden-mlp 256 --compression-size 512 --experiments-base "./experiments/experiment1/experiment1-1024/" --data-path "../game_data/clevr_shape_same/" --seed 1 --csv-general-eval-file "exp1_mi_color.csv" --n-classes 8

python eval_mi.py --load-name "experiment1_baseline_seed1" --selfplay --ce-loss --perspective --n-epochs-init 100 --n-generations 0 --batch-size 200 --n-games 64000 --hidden-size 256 --hidden-mlp 256 --compression-size 512 --experiments-base "./experiments/experiment1/experiment1-1024/" --data-path "../game_data/clevr_shape_same/" --seed 1 --csv-general-eval-file "exp1_mi_shape.csv" --n-classes 10


## Experiment 2

#### To train
To train subsequent generations from a given population from experiment 1 in the low communicative need condition run:

python train.py --load-name "experiment1_40shapegames_seed1" --load-gen 0 --experiment-name "experiment2_gen14_lowneed_run1" --cuda --teacher --selfplay --ce-loss --perspective --n-epochs-self 14 --n-epochs-comm 14 --n-epochs-teach 14 --n-games 64000 --batch-size 512 --n-generations 14 --seed 1 --data-path "../game_data/clevr_shape_same/" --hidden-size 256 --hidden-mlp 256 --compression-size 512  --similarity-weight 0 --distillation-weight 0

To train subsequent generations from a given population from experiment 1 in the high communicative need condition run:

python train.py --load-name "experiment1_40shapegames_seed1" --load-gen 0 --experiment-name "experiment2_gen14_highneed_run1" --cuda --teacher --selfplay --ce-loss --perspective --n-epochs-self 14 --n-epochs-comm 14 --n-epochs-teach 14 --n-games 64000 --batch-size 512 --n-generations 14 --seed 1 --data-path "../game_data/clevr_shape_same/" --hidden-size 256 --hidden-mlp 256 --compression-size 512  --similarity-weight 0 --distillation-weight 0 --proportion 0.4 --images "shape"

#### To evaluate
To evaluate run the following command for both conditions :

python eval_allfeatures.py --load-name "experiment2_gen14_lowneed_run1" --cuda --teacher --ce-loss --perspective --n-epochs-self 14 --n-epochs-comm 14 --n-epochs-teach 14 --n-games 64000 --batch-size 512 --n-generations 14 --seed 1 --data-path "../game_data/clevr_shape_same/" --hidden-size 256 --hidden-mlp 256 --compression-size 512 --csv-general-eval-file "exp2_test_acc.csv" --start-gen 1 --end-gen 14 --similarity-weight 0 --distillation-weight 0

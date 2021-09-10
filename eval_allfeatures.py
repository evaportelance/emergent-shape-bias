import torch
import torch.nn as nn

from population import Population
from scripting_utils import summarize_hps
from stats import StatTracker
import click
import os
import sys
from pathlib import Path

## Hyperparameters!
@click.command()
@click.option("--load-name", default=None)
@click.option("--start-gen", default=0)
@click.option("--end-gen", default=0)  # specify new experiment name else save as datetime string
@click.option("--cuda", is_flag=True)
@click.option("--generalize", is_flag=True)
@click.option("--csv-general-eval-file", default="eval_results.csv")
# learn hps
@click.option("--n-epochs-init", default=10, help="nb of epochs to run initial generation")
@click.option("--n-epochs-teach", default=10, help="nb of epochs to run for child-teacher play")
@click.option("--n-epochs-self", default=10, help="nb of epochs to run for selfplay")
@click.option("--n-epochs-comm", default=10, help="nb of epochs to run for community play")
@click.option("--lr", default=1e-3)
@click.option("--batch-size", default=32)
@click.option("--teacher", is_flag=True)
@click.option("--selfplay", is_flag=True)
@click.option("--ce-loss", is_flag=True)
@click.option("--class-weight", default=1)
@click.option("--similarity-weight", default=0)
@click.option("--distillation-weight", default=0)
# game_hps
@click.option("--n-distractors", default=3)
@click.option("--n-games", default=32000)
@click.option("--images", default="random")
@click.option("--perspective", is_flag=True)
@click.option("--proportion", default=1.0)
@click.option("--color-balance", is_flag=True)
# world hps
@click.option("--n-pairs", default=1)
@click.option("--population-size", default=2)
@click.option("--n-generations", default=2)
@click.option("--seed", default=0) # required
@click.option("--data-path", default="../game_data/clevr_1_10000/")
@click.option("--experiments-base", default="experiments")
# vision hp
@click.option("--encoding-size", default=1024)
@click.option("--compression-size", default=256, type=int)

# agent hps
@click.option("--hidden-size", default=128)
@click.option("--hidden-mlp", default=128)
@click.option("--emb-size", default=64)
@click.option("--message-length", default=7)
@click.option("--vocab-size", default=60)


def run(n_epochs_init, n_epochs_teach, n_epochs_self, n_epochs_comm, lr, batch_size, teacher, selfplay, ce_loss, class_weight, similarity_weight, distillation_weight, n_distractors, n_games, data_path, images, proportion, color_balance, n_pairs, population_size, n_generations,seed, load_name, cuda, encoding_size, compression_size, hidden_size, hidden_mlp, emb_size, message_length, vocab_size, generalize, perspective, csv_general_eval_file, start_gen, end_gen, experiments_base):

    data_path = Path(data_path)
    for gen in range(start_gen, (end_gen+1)):
        learn_hps = {"n_epochs_init": n_epochs_init,
                     "n_epochs_teach": n_epochs_teach,
                     "n_epochs_self": n_epochs_self,
                     "n_epochs_comm": n_epochs_comm,
                     "lr": lr,
                     "batch_size": batch_size,
                     "teacher": teacher,
                     "selfplay": selfplay,
                     "ce_loss": ce_loss,
                     "class_weight": class_weight,
                     "similarity_weight": similarity_weight,
                     "distillation_weight": distillation_weight}

        game_hps = {"n_distractors": n_distractors,
                    "n_games": n_games,
                    "data_path": data_path,
                    "images": images,
                    "perspective": perspective,
                    "proportion": proportion,
                    "color_balance": color_balance}

        world_hps = {"n_pairs": n_pairs,
                     "population_size": population_size,
                     "n_generations": n_generations,
                     "seed": seed,
                     "experiment_name": load_name,
                     "load_name": load_name,
                     "load_gen": gen,
                     "cuda": cuda,
                     "generalize": generalize,
                     "csv_general_eval_file": csv_general_eval_file,
                     "experiments_base": experiments_base}

        vision_hps = {"encoding_size": encoding_size,
                      "compression_size": compression_size}

        agent_hps = {"hidden_size": hidden_size,
                     "hidden_mlp": hidden_mlp,
                     "emb_size": emb_size,
                     "message_length": message_length,
                     "vocab_size": vocab_size}

        world = World(world_hps, learn_hps, game_hps, agent_hps, vision_hps)
        print("DEVICE : "+str(world.device))
        world.eval()


class World(object):
    def __init__(self, world_hps, learn_hps, game_hps, agent_hps, vision_hps):
        self.n_pairs = world_hps["n_pairs"]
        self.population_size = world_hps["population_size"]
        self.n_generations = world_hps["n_generations"]
        self.csv_general_eval_file = world_hps["csv_general_eval_file"]

        self.learn_hps = learn_hps
        self.game_hps = game_hps
        self.world_hps = world_hps
        self.agent_hps = agent_hps
        self.vision_hps = vision_hps

        output_dir = os.path.join('experiments', self.world_hps["experiment_name"])
        self.stat_tracker = StatTracker(log_dir=os.path.join(output_dir, "tensorboard_log_results"))

        if self.world_hps["cuda"]:
            self.device = "cuda"
        else:
            self.device = "cpu"

    def eval(self):
        print("\n----------------------------------------")
        print("INITIALIZING POPULATION")
        print("----------------------------------------\n")
        pop = Population(self.population_size, self.n_generations, self.n_pairs,
                         self.learn_hps, self.agent_hps, self.vision_hps, self.game_hps,
                         self.world_hps, self.device, self.stat_tracker, csv_general_eval_file= self.csv_general_eval_file)
        print("\n----------------------------------------")
        print("EVALUATING POPULATION")
        print("----------------------------------------\n")
        pop.init_community_pairs()
        for pair in pop.pairs:
            pop.eval(pair, n_distractors=3, n_games=1000, images="random")
            #pop.eval(pair, n_distractors=3, n_games=1000, images="position")
            pop.eval(pair, n_distractors=3, n_games=1000, images="color")
            pop.eval(pair, n_distractors=3, n_games=1000, images="shape")
            pop.eval(pair, n_distractors=3, n_games=1000, images="material")
            pop.eval(pair, n_distractors=3, n_games=1000, images="size")


if __name__ == "__main__":
    run()

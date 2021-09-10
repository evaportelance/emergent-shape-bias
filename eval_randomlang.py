import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
#from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_classif
import csv
from message_dataset import MessageDataset
from population import Population
#from model_utils import MessageClassifier, MessageMLP, find_lengths, add_eos_to_messages
from stats import AverageMeterSet, StatTracker
from tqdm import tqdm
import click
import os
import sys
from pathlib import Path

## Hyperparameters!
@click.command()
@click.option("--load-name", default=None)
@click.option("--start-gen", default=0)
@click.option("--end-gen", default=0)
@click.option("--cuda", is_flag=True)
@click.option("--generalize", is_flag=True)
@click.option("--csv-general-eval-file", default="langperm_eval_results.csv")
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
        permprops = [0.2, 0.4, 0.6, 0.8, 1.0]
        #permprops = [0.8]
        for permprop in permprops:
            world.randlang_eval(permprop)

class World(object):
    def __init__(self, world_hps, learn_hps, game_hps, agent_hps, vision_hps):
        self.n_pairs = world_hps["n_pairs"]
        self.population_size = world_hps["population_size"]
        self.n_generations = world_hps["n_generations"]
        self.load_name = world_hps["load_name"]
        self.gen = world_hps["load_gen"]
        self.csv_general_eval_file = world_hps["csv_general_eval_file"]

        self.learn_hps = learn_hps
        self.game_hps = game_hps
        self.world_hps = world_hps
        self.agent_hps = agent_hps
        self.vision_hps = vision_hps

        self.output_dir = os.path.join(self.world_hps["experiments_base"], self.world_hps["experiment_name"])
        self.stat_tracker = StatTracker(log_dir=os.path.join(self.output_dir, "tensorboard_log_results"))

        if self.world_hps["cuda"]:
            self.device = "cuda"
        else:
            self.device = "cpu"

    def calculate_mutual_information(self, dataset):
        labels = []
        messages = []
        batch_size = 200
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        bar = tqdm(dataloader)
        for idx, sample in enumerate(bar, start=1):
            message = sample["message"]
            messages.append(message)
            label = sample["label"]
            label = label.squeeze(1)
            labels.append(label)

        messages = torch.cat(messages).numpy()
        labels = torch.cat(labels).numpy()
        mi_mess_label = np.sum(mutual_info_classif(messages, labels))

        return mi_mess_label

    def randlang_eval(self, permprop):
        print("\n----------------------------------------")
        print("INITIALIZING POPULATION")
        print("----------------------------------------\n")
        pop = Population(self.population_size, self.n_generations, self.n_pairs,
                         self.learn_hps, self.agent_hps, self.vision_hps, self.game_hps,
                         self.world_hps, self.device, self.stat_tracker, self.csv_general_eval_file)
        imgs_all = pop.test_set.imgs
        labels_all = pop.test_set.labels
        size = imgs_all.shape[0]
        labels_all = labels_all[:, 1]
        labels_all = np.resize(labels_all, (size, 1))
        n_batches, last_batch_size = divmod(size, 200)
        batches = []
        for i in range(0, n_batches):
            img_batch = torch.tensor(imgs_all[(i*200):(i*200+200)]).float()
            label_batch = torch.tensor(labels_all[(i*200):(i*200+200)])
            batches.append((img_batch,label_batch))
        if last_batch_size > 0:
            final_img_batch = torch.tensor(imgs_all[(n_batches*200):]).float()
            final_label_batch = torch.tensor(labels_all[(n_batches*200):])
            batches.append((final_img_batch,final_label_batch))


        print("\n----------------------------------------")
        print("EVALUATING MUTUAL INFORMATION")
        print("----------------------------------------\n")
        filename = os.path.join("experiments", str("mi_"+self.csv_general_eval_file))
        idx = np.array(range(0,self.agent_hps["vocab_size"]))
        perm_size = int(permprop * len(idx))
        perm_id = np.random.choice(len(idx), size=perm_size, replace=False)
        rand = idx[perm_id]
        np.random.shuffle(rand)
        
        for agent_id, agent in enumerate(pop.child_agents):
            print("\n----------------------------------------")
            print("PERMUTE EMBEDDINGS")
            print("----------------------------------------\n")
            idx[perm_id] = rand
            input = torch.LongTensor([idx]).to(self.device)
            permuted_embeddings = agent.shared_embedding(input).squeeze(0)
            agent.shared_embedding.weight.data.copy_(permuted_embeddings)
            messages = []
            first_embeddings = []
            labels = []
            agent.eval()
            print("\n----------------------------------------")
            print("CREATING MESSAGE DATASET")
            print("----------------------------------------\n")
            for img_batch,label_batch in batches:
                labels.append(label_batch)
                with torch.no_grad():
                    tgt_img = img_batch.to(self.device)
                    message, first_embedding = agent.get_message_firstembedding(tgt_img)
                    first_embeddings.append(first_embedding)
            first_embeddings = torch.cat(first_embeddings).float()
            labels = torch.cat(labels).long()
            embedding_dataset = MessageDataset(first_embeddings, labels)
            mi_emb_label = self.calculate_mutual_information(embedding_dataset)

            with open(filename,'a') as f:
                writer = csv.writer(f)
                writer.writerow([self.load_name, self.gen, permprop, agent_id, mi_emb_label])

        print("\n----------------------------------------")
        print("EVALUATING ACCURACY")
        print("----------------------------------------\n")

        for pair in pop.pairs:
            pop.eval(pair, n_distractors=3, n_games=1000, images="shape", only_rewards=True, permprop=permprop)

if __name__ == "__main__":
    run()

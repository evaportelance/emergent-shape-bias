import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_classif
import csv
from message_dataset import MessageDataset
from population import Population
from model_utils import MessageClassifier, MessageMLP, find_lengths, add_eos_to_messages
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
@click.option("--classifier", default="linear")
@click.option("--n-classes", default=10)
@click.option("--csv-general-eval-file", default="mi_eval_results.csv")
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


def run(n_epochs_init, n_epochs_teach, n_epochs_self, n_epochs_comm, lr, batch_size, teacher, selfplay, ce_loss, class_weight, similarity_weight, distillation_weight, n_distractors, n_games, data_path, images, proportion, color_balance, n_pairs, population_size, n_generations,seed, load_name, cuda, encoding_size, compression_size, hidden_size, hidden_mlp, emb_size, message_length, vocab_size, generalize, perspective, classifier, n_classes, csv_general_eval_file, start_gen, end_gen, experiments_base):

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
                     "classifier": classifier,
                     "n_classes": n_classes,
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
        world.message_eval()

class World(object):
    def __init__(self, world_hps, learn_hps, game_hps, agent_hps, vision_hps):
        self.n_pairs = world_hps["n_pairs"]
        self.population_size = world_hps["population_size"]
        self.n_generations = world_hps["n_generations"]
        self.classifier = world_hps["classifier"]
        self.n_classes = world_hps["n_classes"]
        self.load_name = world_hps["load_name"]
        self.gen = world_hps["load_gen"]
        self.csv_general_eval_file = world_hps["csv_general_eval_file"]

        self.learn_hps = learn_hps
        self.game_hps = game_hps
        self.world_hps = world_hps
        self.agent_hps = agent_hps
        self.vision_hps = vision_hps

        self.output_dir = os.path.join(self.world_hps["experiments_base"], self.world_hps["experiment_name"])
        if self.classifier == "linear":
            #self.stat_tracker = StatTracker(log_dir=os.path.join(self.output_dir, "tensorboard_log_classifiers"))
            self.stat_tracker = StatTracker(log_dir=os.path.join(self.output_dir, "tensorboard_log_classifiers_shape"))
        else:
            self.stat_tracker = StatTracker(log_dir=os.path.join(self.output_dir, "tensorboard_log_mlp"))

        if self.world_hps["cuda"]:
            self.device = "cuda"
        else:
            self.device = "cpu"

    def train_classifier(self, dataset, agent_id, embeddings=False):
        agent_prefix = "gen_{:d}_agent_{:d}".format(self.gen, agent_id)
        if embeddings:
            size = self.agent_hps["emb_size"]
            prefix = str(agent_prefix + '/train/embedding_classifier/')
            classifier_name = str(agent_id) + "_embedding_classifier"
        else:
            size = self.agent_hps["message_length"]
            prefix = str(agent_prefix + '/train/message_classifier/')
            classifier_name = str(agent_id) + "_message_classifier"

        if self.classifier == "linear":
            classifier = MessageClassifier(self.n_classes, size)
        else:
            classifier = MessageMLP(self.n_classes, size)
        classifier = classifier.to(self.device)
        n_epochs = self.learn_hps["n_epochs_init"]
        batch_size = self.learn_hps["batch_size"]
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(classifier.parameters(), lr=self.learn_hps["lr"])

        for epoch in range(0, n_epochs):
            epoch_stats = AverageMeterSet()
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            bar = tqdm(dataloader)
            for idx, sample in enumerate(bar, start=1):
                optimizer.zero_grad()
                message = sample["message"].to(self.device)
                label = sample["label"].to(self.device)
                label = label.squeeze(1)
                pred = classifier(message)
                loss = criterion(pred,label)
                loss.backward()
                optimizer.step()

                choices_probs = F.softmax(pred, -1)
                max_choice = choices_probs.argmax(dim=1)

                accuracy = (max_choice == label).sum().item()
                epoch_stats.update('loss', loss.item(), n=1)
                epoch_stats.update('accuracy', accuracy, n=batch_size)
            self.stat_tracker.record_stats(epoch_stats.averages(epoch, prefix=prefix))

        agent_dir = os.path.join(self.output_dir, "agents", str(self.gen))
        torch.save(classifier.state_dict(), os.path.join(agent_dir, classifier_name))

        print("\n----------------------------------------")
        print("classifier saved in {:s}".format(agent_dir))
        print("----------------------------------------\n")

        return classifier

    def calculate_mutual_information(self, classifier, dataset):
        labels = []
        messages = []
        predictions = []
        batch_size = 200
        classifier.eval()
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        bar = tqdm(dataloader)
        for idx, sample in enumerate(bar, start=1):
            message = sample["message"]
            messages.append(message)
            label = sample["label"]
            label = label.squeeze(1)
            labels.append(label)

            message = message.to(self.device)
            logits = classifier(message)
            choices_probs = F.softmax(logits, -1)
            y_pred = choices_probs.argmax(dim=1)
            predictions.append(y_pred)

        messages = torch.cat(messages).numpy()
        predictions = torch.cat(predictions).numpy()
        labels = torch.cat(labels).numpy()

        mi_pred_label = mutual_info_score(labels, predictions)
        mi_mess_label = np.sum(mutual_info_classif(messages, labels))

        return mi_pred_label, mi_mess_label

    def message_eval(self):
        print("\n----------------------------------------")
        print("INITIALIZING POPULATION")
        print("----------------------------------------\n")
        pop = Population(self.population_size, self.n_generations, self.n_pairs,
                         self.learn_hps, self.agent_hps, self.vision_hps, self.game_hps,
                         self.world_hps, self.device, self.stat_tracker)
        imgs_all = pop.test_set.imgs
        labels_all = pop.test_set.labels
        size = imgs_all.shape[0]
        if self.n_classes == 10:
            labels_all = labels_all[:, 1]
        else:
            labels_all = labels_all[:, 0]
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
        print("EVALUATING POPULATION")
        print("----------------------------------------\n")
        filename = os.path.join("experiments", self.csv_general_eval_file)

        for agent_id, agent in enumerate(pop.child_agents):
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
                    message_lengths = find_lengths(message)
                    cleaned_message = add_eos_to_messages(message, message_lengths, self.agent_hps["message_length"])
                    messages.append(cleaned_message)
                    first_embeddings.append(first_embedding)
            messages = torch.cat(messages).float()
            first_embeddings = torch.cat(first_embeddings).float()
            labels = torch.cat(labels).long()
            message_dataset = MessageDataset(messages, labels, self.classifier)
            message_classifier = self.train_classifier(message_dataset, agent_id)
            mi_messpred_label, mi_mess_label = self.calculate_mutual_information(message_classifier, message_dataset)

            embedding_dataset = MessageDataset(first_embeddings, labels, self.classifier)
            embedding_classifier = self.train_classifier(embedding_dataset, agent_id, embeddings=True)
            mi_embpred_label, mi_emb_label = self.calculate_mutual_information(embedding_classifier, embedding_dataset)

            with open(filename,'a') as f:
                writer = csv.writer(f)
                writer.writerow([self.load_name, self.gen, agent_id, mi_messpred_label, mi_mess_label, mi_embpred_label, mi_emb_label])

if __name__ == "__main__":
    run()

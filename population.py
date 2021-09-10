import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

import os
from tqdm import tqdm
import pickle
import numpy as np
import random
import csv

from agents import Agent
from games import ClevrGame
from model_utils import find_lengths, find_lengths_onehot, get_edit_distance
from stats import AverageMeterSet, StatTracker


class Population(object):
    def __init__(self, population_size, n_generations, n_pairs, learn_hps,
                 agent_hps, vision_hps, game_hps, world_hps, device, stat_tracker,
                 csv_general_eval_file="eval_results.csv"):
        super().__init__()

        self.load_name = world_hps["load_name"]
        self.load_gen = world_hps["load_gen"]
        self.experiment_name = world_hps["experiment_name"]
        self.world_hps = world_hps
        self.csv_general_eval_file = csv_general_eval_file

        if self.experiment_name is None:
            self.experiment_name = "default"
        self.output_dir = os.path.join(self.world_hps["experiments_base"], self.experiment_name)

        self.device = device
        self.stat_tracker = stat_tracker
        self.generalize = world_hps["generalize"]
        self.ce_loss = learn_hps["ce_loss"]

        # set generator seeds
        self.seed = world_hps["seed"]
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # hps dicts
        self.agent_hps = agent_hps
        self.game_hps = game_hps
        self.learn_hps = learn_hps
        self.world_hps = world_hps
        self.vision_hps = vision_hps

        # population info
        self.population_size = population_size
        self.n_generations = n_generations
        self.n_pairs = n_pairs

        # dataloading info
        self.data_path = self.game_hps["data_path"]
        self.n_games = self.game_hps["n_games"]
        self.images = self.game_hps["images"]
        self.n_distractors = self.game_hps["n_distractors"]
        self.perspective = self.game_hps["perspective"]
        self.proportion = self.game_hps["proportion"]
        self.color_balance = self.game_hps["color_balance"]

        # set up datasets
        if self.generalize:
            self.train_set = ClevrGame(self.data_path, perspective=self.perspective, proportion=self.proportion, color_balance = self.color_balance, eval=False)
            self.test_set = ClevrGame(self.data_path, perspective=self.perspective, proportion=self.proportion, color_balance = self.color_balance, eval=False)
        else:
            self.train_set = ClevrGame(self.data_path, perspective=self.perspective, proportion=self.proportion, color_balance = self.color_balance, eval=False)
            self.test_set = ClevrGame(self.data_path, perspective=self.perspective, proportion=self.proportion, color_balance = self.color_balance, eval=True)

        # create agent populations
        self.generation = world_hps["load_gen"]
        self.community_epochs = 0
        self.teacher_epochs = 0
        self.selfplay_epochs = 0
        self.pairs = None
        self.teacher_agents = []
        self.child_agents = []

        self.init_child_agents()
        self.init_community_pairs()

        if self.load_name is not None:
            self.load_generation()

    def init_child_agents(self):
        # remove pointers to previous agents
        self.child_agents = []
        # creates new ramdomly initialized child agents
        for i in range(self.population_size):
            agent = Agent(self.agent_hps, self.vision_hps, self.device)
            agent = agent.to(self.device)
            self.child_agents.append(agent)

    def init_teacher_agents(self):
        # remove pointers to previous agents
        self.teacher_agents = []
        # replace with pointers from this generations children
        self.teacher_agents = self.child_agents
        # stop teacher agent learning ?? can remove this
        for agent in self.teacher_agents:
            for param in agent.parameters():
                param.requires_grad = False


    def init_community_pairs(self):
        # for now let's just assume we only have two agents per generation
        self.pairs = [(0, 1) for i in range(self.world_hps["n_pairs"])]



### TRAINING ###

    def train(self):
        #make directory to save results and save hps for the experiment
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.save_hps()
        if self.load_name is None:
            # train gen 0 with  selfplay and no teacher
            if self.learn_hps["selfplay"]:
                print("\n----------------------------------------")
                print("TRAINING SELFPLAY INITIAL GENERATION ")
                print("----------------------------------------\n")
                for child_agent_id in range(self.population_size):
                    self.train_selfplay(child_agent_id, None, with_teacher=False)

            # train gen 0 with community play
            print("\n----------------------------------------")
            print("TRAINING INITIAL GENERATION")
            print("----------------------------------------\n")
            n_epochs = self.learn_hps["n_epochs_init"]
            for pair in self.pairs:
                self.train_single_pair(pair, n_epochs)
            self.save_generation()

        # train subsequent generations
        for i in range(self.n_generations):
            self.train_generation()


    def train_generation(self):
        if not self.generalize:
            self.init_teacher_agents()
            #keep track of current generation for stat logging
            self.generation += 1
            #create new children
            self.init_child_agents()

        # PHASE 1: self-play for all child agents
        if self.learn_hps["selfplay"]:
            print("\n----------------------------------------")
            print("TRAINING SELFPLAY GENERATION : "+str(self.generation))
            print("----------------------------------------\n")
            #select teacher currently same teacher per generation
            #teacher_agent_id = np.random.choice(range(self.population_size))
            for child_agent_id in range(self.population_size):
                teacher_agent_id = child_agent_id
                self.train_selfplay(child_agent_id, teacher_agent_id)

        # PHASE 2: child-teacher play
        if self.learn_hps["teacher"]:
            print("\n----------------------------------------")
            print("TRAINING CHILD-TEACHER PLAY : "+str(self.generation))
            print("----------------------------------------\n")
            n_epochs = self.learn_hps["n_epochs_teach"]
            #teacher_agent_id = np.random.choice(range(self.population_size))
            for child_agent_id in range(self.population_size):
                teacher_agent_id = child_agent_id
                self.train_single_pair((child_agent_id, teacher_agent_id), n_epochs, game_type="teacher")

        # PHASE 3: community play
        print("\n----------------------------------------")
        print("TRAINING COMMUNITY PLAY : "+str(self.generation))
        print("----------------------------------------\n")
        n_epochs = self.learn_hps["n_epochs_comm"]
        for pair in self.pairs:
            self.train_single_pair(pair, n_epochs)

        # save generation of agents (only child_agents)
        #if self.generation % 10 == 0:
        self.save_generation()
        # Child agents become teachers
        self.init_teacher_agents()

    # Training an agent pair
    def train_single_pair(self, pair, n_epochs, game_type="community"):
        """
        pair: tuple of indices (p1, p2)
        """
        agent1_id = pair[0]
        agent2_id = pair[1]

        # if teacher child play, get first agent from children and second from teachers
        if game_type == "teacher":
            agent1 = self.child_agents[agent1_id]
            # prefixes are for stat tracking
            agent1_prefix = "gen_{:d}_agent_{:d}".format(self.generation, agent1_id)
            agent2 = self.teacher_agents[agent2_id]
            agent2_prefix = "gen_{:d}_agent_{:d}".format((self.generation-1), agent2_id)

        # community play currently involves only playing with agents from the same generation
        else:
            agent1 = self.child_agents[agent1_id]
            agent1_prefix = "gen_{:d}_agent_{:d}".format(self.generation, agent1_id)
            agent2 = self.child_agents[agent2_id]
            agent2_prefix = "gen_{:d}_agent_{:d}".format(self.generation, agent2_id)

        agent1.to(self.device)
        agent2.to(self.device)

        agent1.train()
        agent2.train()


        batch_size = self.learn_hps["batch_size"]
        lr = self.learn_hps["lr"]

        if self.ce_loss:
            criterion = nn.CrossEntropyLoss(reduction='none')
        optimizer = torch.optim.Adam(list(agent1.parameters()) + list(agent2.parameters()), lr)
        steps = 0
        for epoch in range(n_epochs):
            epoch_stats = AverageMeterSet()

            if self.n_distractors is None:  # use curriculum
                n_distractors = self.compute_distractors(epoch)
            else:
                n_distractors = self.n_distractors

            # record performance before training (eg. after selfplay)
            if epoch == 0:
                self.eval(pair, n_distractors, game_type=game_type, images=self.images, eval=False)
                agent1.train()
                agent2.train()

            # create a ramdom set of games for epoch
            dataloader = self.init_dataloader(n_distractors=n_distractors, batch_size=batch_size, num_games=self.n_games, images=self.images)
            bar = tqdm(dataloader)

            current_rewards = []

            for idx, game in enumerate(bar, start=1):
                agents = [agent1, agent2]

                # randomly select sender/receiver assignments
                s = np.random.choice([0, 1])
                r = 1 - s

                sender = agents[s]
                receiver = agents[r]

                optimizer.zero_grad()

                sender_input = game["target_img"].to(self.device)
                receiver_input = game["imgs"].to(self.device)
                labels = game["labels"].to(self.device)

                # sender_logits : (B, message length, 1)
                #                     log p(message_i | image, speaker)
                #
                # LazyImpa: 1/L \sum_l cross-ent(image | l, receiver)
                #           ^ reward for all the sender logits
                #
                # Per-step rewards instead of global:
                # -----------------------------------
                # f o o
                # acc(image | f, receiver), acc(image | fo, receiver), acc(image | foo, receiver)
                # ^reward for logits f      ^reward for o              ^reward for o
                # baseline for each step
                message, sender_logits, sender_entropy, sender_all_probs = sender(tgt_img=sender_input, mode="sender")
                choices, receiver_logits, receiver_entropy, receiver_final_logits, receiver_all_state_probs = receiver(message=message, imgs=receiver_input, mode="receiver")
                message_lengths = find_lengths(message)

                effective_sender_logits = self.mask_sender_logits(sender_logits, message_lengths)

                acc_rewards = self.compute_rewards(choices, labels).detach()
                mean_accuracy = acc_rewards.mean().detach()

                if self.ce_loss:
                    cross_entropy = -criterion(receiver_final_logits, labels)
                    ce_rewards = torch.exp(cross_entropy).detach()
                    #mean_rewards = ce_rewards.mean().detach()
                #else:
                    #mean_rewards = mean_accuracy

                mean_rewards = mean_accuracy
                current_rewards.append(mean_rewards) ### for baseline
                if len(current_rewards) >= 100: ### baseline based on previous 100 games
                    current_rewards = current_rewards[-100:]
                baseline = torch.mean(torch.Tensor(current_rewards)).detach() ### mean over last 100 batches not global
                coeff = self.compute_entropy_coeff(steps, mean_rewards, baseline)

                if self.ce_loss:
                    #loss = self.compute_pair_loss_cross_entropy(effective_sender_logits, ce_rewards, cross_entropy, baseline, sender_entropy, coeff)
                    loss = self.compute_pair_loss_cross_entropy(effective_sender_logits, acc_rewards, cross_entropy, baseline, sender_entropy, coeff)
                else:
                    loss = self.compute_pair_loss(effective_sender_logits, receiver_logits, acc_rewards, baseline, sender_entropy, coeff)

                loss.backward()
                optimizer.step()

                epoch_stats.update('mean_rewards', mean_rewards, n=1)
                epoch_stats.update('mean_accuracy', mean_accuracy, n=1)
                steps +=1

            # keep track of each agents training epoch individually for stat tracking by agent
            if game_type == "teacher":
                agent1.epochs_teacher+= 1
                self.stat_tracker.record_stats(epoch_stats.averages(agent1.epochs_teacher, prefix=str(agent1_prefix + '/train/teacher/')))

                self.teacher_epochs +=1
            else:
                agent1.epochs_community+= 1
                self.stat_tracker.record_stats(epoch_stats.averages(agent1.epochs_community, prefix=str(agent1_prefix + '/train/community/')))
                agent2.epochs_community += 1
                self.stat_tracker.record_stats(epoch_stats.averages(agent2.epochs_community, prefix=str(agent2_prefix + '/train/community/')))

                self.community_epochs +=1

            #  get performance on the test set every X epochs of training
            #if (epoch % 2) == 1:
            self.eval(pair, n_distractors, game_type=game_type, images=self.images, eval=False)
            agent1.train()
            agent2.train()

        return

    def train_selfplay(self, child_agent_id, teacher_agent_id, with_teacher=True):
        """
        child_agent id: int representing indice of child agent
        teacher_agent_id: int representing indice of teacher agent
        """
        #prefix for stat tracking
        child_agent_prefix = "gen_{:d}_agent_{:d}".format(self.generation, child_agent_id)
        child_agent = self.child_agents[child_agent_id]
        child_agent.to(self.device)
        if with_teacher:
            # teacher agent used for message similarity measure
            teacher_agent = self.teacher_agents[teacher_agent_id]
            teacher_agent.to(self.device)

        n_epochs = self.learn_hps["n_epochs_self"]
        batch_size = self.learn_hps["batch_size"]
        lr = self.learn_hps["lr"]

        #only child agent being updated
        optimizer = torch.optim.Adam(list(child_agent.parameters()), lr)
        criterion_class = nn.CrossEntropyLoss(reduction='none')
        criterion_distillation = nn.CrossEntropyLoss(reduction='none')

        steps = 0
        for epoch in range(n_epochs):
            epoch_stats = AverageMeterSet()

            if epoch == 0:
                self.eval(child_agent_id, self.n_distractors, game_type="selfplay", images=self.images, eval=False)
                child_agent.train()

            dataloader = self.init_dataloader(n_distractors=self.n_distractors, batch_size=batch_size, num_games=self.n_games, images=self.images)
            bar = tqdm(dataloader)

            current_rewards = []

            for idx, game in enumerate(bar, start=1):
                optimizer.zero_grad()

                tgt_img = game["target_img"].to(self.device)
                imgs = game["imgs"].to(self.device)
                labels = game["labels"].to(self.device)

                child_messages, choices, max_choices, speaker_logits, speaker_entropy = child_agent(tgt_img=tgt_img, imgs=imgs, mode="selfplay")
                # get classification loss from image choices
                class_loss = criterion_class(choices, labels)

                if with_teacher:
                    # get teacher message for similarity score, and teacher choice for distillation loss
                    teacher_messages, _, _, _ = teacher_agent(tgt_img=tgt_img, mode="sender")
                    child_messages_gpu = child_messages.to(self.device)
                    teacher_choices, _, _, _, _ = teacher_agent(message=child_messages_gpu, imgs=imgs, mode="receiver")
                    # get distillation loss where child matches their message to teacher choices
                    distillation_loss=criterion_distillation(choices, teacher_choices)
                    distillation_accuracy = (max_choices == teacher_choices).sum().item()
                    epoch_stats.update('distillation_accuracy', distillation_accuracy, n=batch_size)
                    # get similarity score between child message and teacher message
                    similarity_score = self.message_similarity_score(child_messages, teacher_messages.cpu(), method="levenshtein")
                    similarity_score = similarity_score.to(self.device)
                    epoch_stats.update('similarity_score', similarity_score.mean(), n=1)
                    mean_rewards = similarity_score.mean().detach()
                    current_rewards.append(mean_rewards) ### for baseline
                    if len(current_rewards) >= 100: ### baseline based on previous 100 games
                        current_rewards = current_rewards[-100:]
                    baseline = torch.mean(torch.Tensor(current_rewards)).detach() ### mean over last 100 batches not global
                    coeff = self.compute_entropy_coeff(steps, mean_rewards, baseline)

                    # combine classification and message similarity score in loss
                    loss = self.compute_selfplay_loss(speaker_logits, class_loss, distillation_loss, similarity_score, baseline, speaker_entropy, coeff)
                else:
                    loss = class_loss.mean()

                loss.backward()
                optimizer.step()

                # accuracy for stat tracking
                accuracy = (max_choices == labels).sum().item()

                epoch_stats.update('loss', loss.item(), n=1)
                epoch_stats.update("accuracy", accuracy, n=batch_size)


            # keep track of selfplay epochs for individual agents for stat tracking
            child_agent.epochs_selfplay += 1
            self.stat_tracker.record_stats(epoch_stats.averages(child_agent.epochs_selfplay, prefix=str(child_agent_prefix + '/train/selfplay/')))

            self.selfplay_epochs +=1

            if (epoch % 2) == 1:
                self.eval(child_agent_id, self.n_distractors, game_type="selfplay", images=self.images, eval=False)
                child_agent.train()
                steps+=1

        return

### EVALUATION ###
    def eval(self, pair, n_distractors, batch_size= 100, n_games= 1000, game_type="community", images="random", eval=True, only_rewards=False, permprop=0.0):
        if game_type == "selfplay":
            print("stabilizing batch norm...")
            _ = self.eval_selfplay(pair, n_distractors, batch_size, n_games, batchnorm=True, images=images, eval=False)  # run fwd pass to stabilize batchnorm
            print("evaluating!")
            _ = self.eval_selfplay(pair, n_distractors, batch_size, n_games, images=images, eval=eval, only_rewards=only_rewards)
        else:
            print("stabilizing batch norm...")
            _ = self.eval_single_pair(pair, n_distractors, batch_size, n_games, game_type, batchnorm=True, images=images, eval=False)  # run fwd pass to stabilize batchnorm
            print("evaluating!")
            _ = self.eval_single_pair(pair, n_distractors, batch_size, n_games, game_type=game_type, images=images, eval=eval, only_rewards=only_rewards, permprop=permprop)

    def eval_single_pair(self, pair, n_distractors, batch_size, n_games, game_type="community", batchnorm=False, images="random", eval=True, only_rewards = False, permprop=0.0):
        """
        pair: tuple of indices (p1, p2)
        """
        agent1_id = pair[0]
        agent2_id = pair[1]

        # same as train, if child teacher play, first agent is child second is teacher
        if game_type == "teacher":
            agent1 = self.child_agents[agent1_id]
            agent1_prefix = "gen_{:d}_agent_{:d}".format(self.generation, agent1_id)
            agent2 = self.teacher_agents[agent2_id]
            agent2_prefix = "gen_{:d}_agent_{:d}".format((self.generation-1), agent2_id)
        else:
            agent1 = self.child_agents[agent1_id]
            agent1_prefix = "gen_{:d}_agent_{:d}".format(self.generation, agent1_id)
            agent2 = self.child_agents[agent2_id]
            agent2_prefix = "gen_{:d}_agent_{:d}".format(self.generation, agent2_id)
        agent1.to(self.device)
        agent2.to(self.device)

        # set agents to eval
        agent1.eval()
        agent2.eval()

        test_stats = AverageMeterSet()
        dataloader = self.init_dataloader(n_distractors=n_distractors, batch_size=batch_size, num_games=n_games, images=images, eval=True)
        bar = tqdm(dataloader)

        all_rewards = []
        all_messages = []
        all_games = []

        for idx, game in enumerate(bar, start=1):
            game_log = {}

            agents = [agent1, agent2]
            s = np.random.choice([0, 1])
            r = 1 - s

            sender = agents[s]
            receiver = agents[r]

            sender_input = game["target_img"].to(self.device)
            receiver_input = game["imgs"].to(self.device)
            labels = game["labels"].to(self.device)

            message, _, _, _ = sender(tgt_img=sender_input, mode="sender")
            choices, _, _, _, _ = receiver(message=message, imgs=receiver_input, mode="receiver")

            rewards = self.compute_rewards(choices, labels).detach().cpu().numpy()
            mean_rewards = rewards.mean()

            test_stats.update('mean_accuracy', mean_rewards, n=1)

            if eval:
                # log message/reward
                all_rewards.append(rewards)
                all_messages.append(message)

                # game log
                game_log["generation"] = self.generation
                game_log["speaker"] = pair[s]
                game_log["receiver"] = pair[r]
                game_log["image_type"] = images
                #game_log["target_img"] = sender_input
                #game_log["all_imgs"] = receiver_input
                game_log["message"] = message
                #game_log["img_choice"] = choices
                game_log["reward"] = rewards

                all_games.append(game_log)

        if not batchnorm and not eval:
            if game_type == "teacher":
                self.stat_tracker.record_stats(test_stats.averages(agent1.epochs_teacher, prefix=str(agent1_prefix + '/test/teacher/')))

                self.stat_tracker.record_stats(test_stats.averages(self.teacher_epochs, prefix='all_gen/test/teacher/'))
            else:
                self.stat_tracker.record_stats(test_stats.averages(agent1.epochs_community, prefix=str(agent1_prefix + '/test/community/')))
                self.stat_tracker.record_stats(test_stats.averages(agent2.epochs_community, prefix=str(agent2_prefix + '/test/community/')))

                self.stat_tracker.record_stats(test_stats.averages(self.community_epochs, prefix='all_gen/test/community/'))
        if eval:
            self.save_eval_results(pair, game_type, all_rewards, all_messages, all_games, images=images, only_rewards=only_rewards, permprop=permprop)

        return

    def eval_selfplay(self, child_agent_id, n_distractors, batch_size, n_games, batchnorm=False, images="random", eval=True, only_rewards=False):
        #prefix for stat tracking
        child_agent_prefix = "gen_{:d}_agent_{:d}".format(self.generation, child_agent_id)
        child_agent = self.child_agents[child_agent_id]
        child_agent.to(self.device)
        child_agent.eval()

        test_stats = AverageMeterSet()
        dataloader = self.init_dataloader(n_distractors=n_distractors,batch_size=batch_size, num_games=n_games, images=images, eval=True)
        bar = tqdm(dataloader)

        all_rewards = []
        all_messages = []
        all_games = []

        for idx, game in enumerate(bar, start=1):
            game_log = {}

            tgt_img = game["target_img"].to(self.device)
            imgs = game["imgs"].to(self.device)
            labels = game["labels"].to(self.device)

            child_messages, choices, max_choices, _, _ = child_agent(tgt_img=tgt_img, imgs=imgs, mode="selfplay")

            accuracy = (max_choices == labels).sum().item()

            test_stats.update('accuracy', accuracy, n=1)

            if eval:
                # log message/reward
                all_rewards.append(accuracy)
                all_messages.append(child_messages)

                # game log
                game_log["generation"] = self.generation
                game_log["agent"] = child_agent_id
                game_log["image_type"] = images
                #game_log["target_img"] = tgt_img
                #game_log["all_imgs"] = imgs
                game_log["message"] = child_messages
                #game_log["img_choice"] = max_choices
                game_log["accuracy"] = accuracy

                all_games.append(game_log)

        if not batchnorm and not eval:
                self.stat_tracker.record_stats(test_stats.averages(child_agent.epochs_selfplay, prefix=str(child_agent_prefix + '/test/selfplay/')))

                self.stat_tracker.record_stats(test_stats.averages(self.selfplay_epochs, prefix='all_gen/test/selfplay/'))
        if eval:
            self.save_eval_results((child_agent_id, child_agent_id), "selfplay", all_rewards, all_messages, all_games, images=images, only_rewards=only_rewards)

        return


### AUXILIARY FUNCTIONS ###

    def compute_distractors(self, epoch):
        if epoch < 32:
            n_distractors = 3
        elif epoch >= 32 and epoch < 64:
            n_distractors = 5
        elif epoch >= 64 and epoch < 96:
            n_distractors = 7
        else:
            n_distractors = 10

        return n_distractors

    def init_dataloader(self, n_distractors, batch_size, num_games, images="random", eval=False):
        dataloader = None
        if eval:
            self.test_set.create_games(num_games, n_distractors, type=images)
            dataloader = DataLoader(self.test_set, batch_size=batch_size, shuffle=True)
        else:
            self.train_set.create_games(num_games, n_distractors, type=images)
            dataloader = DataLoader(self.train_set, batch_size=batch_size, shuffle=True)

        return dataloader

    def compute_rewards(self, receiver_output, labels):
        rewards = (labels == receiver_output).float()

        return rewards

    def compute_selfplay_loss(self, s_logits, class_loss, distillation_loss, similarity_score, baseline, s_entropy, coeff):
       entropy_term = 0
       # listener entropy
       if self.learn_hps["class_weight"] > 0:
           loss = class_loss
       else:
           loss = distillation_loss
       # speaker similarity
       if self.learn_hps["similarity_weight"] > 0:
           loss += torch.sum(s_logits, 1) * -(similarity_score - baseline)
           entropy_term = coeff * s_entropy.mean().detach()
           loss = loss.mean() - entropy_term
       else:
           loss = loss.mean()
       return loss

    def compute_pair_loss_cross_entropy(self, s_logits, cross_entropy_rewards, cross_entropy, baseline, s_entropy, coeff):
       # loss speaker  (log p(message | input) * - (Reward - baseline))
       # change this ce_rewards to probabilites and detach
       loss = torch.sum(s_logits, 1) * -(cross_entropy_rewards - baseline)
       #loss = torch.sum(s_logits, 1) * -(torch.clamp(cross_entropy_rewards - baseline, -1, 1))
       # loss receiver (-log p(label | message))
       # keep this ce_rewards as logprobs and DNot detach
       loss += -cross_entropy
       entropy_term = coeff * s_entropy.mean().detach()
       return loss.mean() - entropy_term

    def compute_pair_loss(self, s_logits, r_logits, rewards, baseline, s_entropy, coeff):
        loss = (torch.sum(s_logits, 1) + r_logits) * -(rewards - baseline) ### Maybe log difference between rewards and baseline

        entropy_term = coeff * s_entropy.mean().detach()

        return loss.mean() - entropy_term

    def compute_entropy_coeff(self, steps, rewards, baseline):
        #if steps < 100000:
        #    coeff = 0.1 - torch.abs((rewards - baseline) * 0.1)
        #else:
        coeff = 0.01

        return coeff

    def mask_sender_logits(self, sender_logits, message_lengths): ### From EGG : to transform all tokens after eos char to zero

        effective_sender_logits = torch.zeros_like(sender_logits)
        max_length = self.agent_hps["message_length"]

        for i in range(max_length):
            not_eosed = (i < message_lengths).float()
            effective_sender_logits[:, i] = sender_logits[:, i] * not_eosed

        return effective_sender_logits

    def message_similarity_score(self, child_messages, teacher_messages, method="levenshtein"):
        # turn all char after eos char to eos char.
        child_message_lengths = find_lengths(child_messages)
        child_messages = self.mask_sender_logits(child_messages, child_message_lengths)
        teacher_message_lengths = find_lengths(teacher_messages)
        teacher_messages = self.mask_sender_logits(teacher_messages, teacher_message_lengths)

        if method == "levenshtein":
            scores = torch.tensor(list(map(get_edit_distance, teacher_messages, child_messages)))
            norm_scores = torch.true_divide(scores, self.agent_hps["message_length"])
            return (1 - norm_scores)
        else:
            return torch.tensor([1.0])

    def compute_reward_stats(self, rewards, images="random", permprop=0.0):
        mean_rewards = np.mean(rewards)
        filename = os.path.join("experiments", self.csv_general_eval_file)
        with open(filename,'a') as f:
            writer = csv.writer(f)
            if permprop > 0.0:
                writer.writerow([self.load_name, self.load_gen, permprop, images, mean_rewards])
            else:
                writer.writerow([self.load_name, self.load_gen, images, mean_rewards])
        #var_rewards = np.var(rewards)
        #std_rewards = np.std(rewards)
        # print("\n----------------------------------------")
        # print("mean reward:", mean_rewards)
        # print("reward variance:", var_rewards)
        # print("reward std:", std_rewards)
        # print("----------------------------------------\n")

    def save_eval_results(self, pair, game_type, all_rewards, all_messages, all_games, images="random", only_rewards = False, permprop=0.0):
        self.compute_reward_stats(all_rewards, images=images, permprop=permprop)
        if not only_rewards:
            generation_dir = "gen_{:d}".format(self.generation)
            pair_dir = "pair_{:d}_{:d}".format(pair[0], pair[1]) + "_" + game_type
            output_dir = os.path.join(self.output_dir,"results", generation_dir, pair_dir)

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            with open(os.path.join(output_dir, "rewards.pkl"), "ab") as f:
                pickle.dump(all_rewards, f)
            with open(os.path.join(output_dir, "messages.pkl"), "ab") as f:
                pickle.dump(all_messages, f)
            with open(os.path.join(output_dir, "game_logs.pkl"), "ab") as f:
                pickle.dump(all_games, f)

            print("\n----------------------------------------")
            print("results saved in {:s}".format(output_dir))
            print("----------------------------------------\n")


    def save_hps(self):
        all_hps = {"agent_hps": self.agent_hps,
                   "game_hps": self.game_hps,
                   "learn_hps": self.learn_hps,
                   "world_hps": self.world_hps,
                   "vision_hps": self.vision_hps
                    }

        save_path = os.path.join(self.output_dir, "hps.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(all_hps, f)

    def save_generation(self):
        agent_dir = os.path.join(self.output_dir, "agents", str(self.generation))
        if not os.path.exists(agent_dir):
            os.makedirs(agent_dir)

        for i in range(self.population_size):
            torch.save(self.child_agents[i].state_dict(), os.path.join(agent_dir, str(i)))

        print("\n----------------------------------------")
        print("generation saved in {:s}".format(agent_dir))
        print("----------------------------------------\n")

    def load_generation(self):
        agent_path = os.path.join(self.world_hps["experiments_base"], self.load_name, "agents", str(self.load_gen))
        for i in range(self.population_size):
            save_path = os.path.join(agent_path, str(i))
            self.child_agents[i].load_state_dict(torch.load(save_path))

        print("\n----------------------------------------")
        print("agents loaded from {:s}".format(agent_path))
        print("----------------------------------------\n")

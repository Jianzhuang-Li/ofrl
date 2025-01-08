import torch
from fisher_brc import FisherBRC
from hand_conf import fbrc_config
from modules import Actor, EnsembledCritic, MixtureGaussianActor
from diffusion.conf import hand_door_conf
from diffusion.data.hand_door_dataset import Hand_Door_With_Next
from diffusion_ac import DiffusionActor
from diffusion.conf import hand_door_conf
from dataset import ReplayBuffer
import torch.utils.data as Data
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np
import argparse
import sys
import random
# import wandb
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter # Create an instance of the object 

parser = argparse.ArgumentParser(description='train fisher behaviour clone offline clone model')
parser.add_argument('--task', type=str, choices=["hopper", "hand_door"], default='hopper', help="The env to be test.")
parser.add_argument('--task', type=str, choices=["diffusion", "guassian"], default='diffusion', help="Use diffusion or gaussian for bc model.")
parser.add_argument('--feature', choices=["train", "eval"], type=str, default="eval", help="train or eval")
args = parser.parse_args()


class FBRCTrainer:
    def __init__(self, cfg=fbrc_config) -> None:
        self.writer = SummaryWriter()
        self.cfg = cfg
        self.device = cfg.device

        self.batch_size = cfg.batch_size

        actor = Actor(cfg.state_dim,
                      cfg.action_dim,
                      cfg.hidden_dim)
        if not ACTOR_DIFFUSION:
            behavior = MixtureGaussianActor(cfg.state_dim,
                                        cfg.action_dim,
                                        cfg.hidden_dim,
                                        cfg.num_bc_actors)
        else:
            behavior = DiffusionActor(hand_door_conf.conf_hand_door).to("cuda")
        critic = EnsembledCritic(cfg.state_dim,
                                 cfg.action_dim,
                                 cfg.hidden_dim)

        self.fbrc = FisherBRC(cfg,
                         actor,
                         behavior,
                         critic)
        
        # self.buffer = ReplayBuffer(cfg.state_dim, cfg.action_dim, cfg.buffer_size)
        # self.buffer.from_json(cfg.dataset_name)
        self.data_specific = hand_door_conf.data_specific
        self.dataset = Hand_Door_With_Next(self.data_specific.data_conf)
        self.dataset_loader = Data.DataLoader(dataset=self.dataset,
                                              # batch_size=self.data_specific.batch_size,
                                              batch_size = 128,
                                              shuffle=self.data_specific.shuffle,
                                              num_workers=self.data_specific.num_workers,
                                              )
        
    def fit_bc(self):
         # for t in tqdm(range(self.cfg.behavior_pretrain_steps), desc="Behavior Pretrain"):
        for t in range(self.cfg.behavior_pretrain_steps):
            # batch = self.buffer.sample(self.batch_size)
            for batch in tqdm(self.dataset_loader, position=0):
                obs, action, reward, discount, obs_next = batch
                states, actions = [obs.to(self.device), action.to(self.device)]

                logging_dict = self.fbrc.behavior_pretrain(states, actions)

                self.writer.add_scalars("Behavior", logging_dict, global_step=self.fbrc.bc_pretrain_steps)
                # wandb.log(logging_dict, step=self.fbrc.bc_pretrain_steps)
            tqdm.write(f"Current Batch: {t + 1}")
            if (t + 1) % self.cfg.save_interval == 0:
                self.fbrc.save_behavior(t+1)

    def pca_fit(self):
        indexs = np.arange(len(self.dataset))
        choised_indexs = random.sample(list(indexs), 1000)
        states = []
        actions = []
        q_values= []
        for index in choised_indexs:
            state, action, r, d, o = self.dataset[index]
            states.append(state)
            actions.append(action)
            with torch.no_grad():
                state = torch.tensor(state, device="cuda")
                action = torch.tensor(action, device="cuda")
                q_value = self.fbrc.critic(state, action).detach().cpu().numpy()
                q_values.append(np.mean(q_value))
        states = np.array(states)
        actions = np.array(actions)
        q_values = np.array(q_values)

        data = np.concatenate([states, actions], axis=1)
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(data)

        plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=q_values, cmap='viridis')
        plt.colorbar(label='Q-value')
        plt.xlabel('pca dimension 1')
        plt.ylabel('pca dimension 2')
        plt.title('Q-value distribution')
        plt.show() 

    def fit(self):
        print(f"Training starts on {self.cfg.device} 🚀")

        """
        with wandb.init(project=self.cfg.project, entity="zzmtsvv", group=f"behavior_{self.cfg.group}", name=self.cfg.name):
            wandb.config.update({k: v for k, v in self.cfg.__dict__.items() if not k.startswith("__")})
        """  
        # self.fit_bc()
        # for t in tqdm(range(self.cfg.behavior_pretrain_steps), desc="Behavior Pretrain"):
        self.fbrc.load_behavior(200)
        """
        for t in range(self.cfg.behavior_pretrain_steps):
            # batch = self.buffer.sample(self.batch_size)
            for batch in tqdm(self.dataset_loader, position=0):
                obs, action, reward, discount, obs_next = batch
                states, actions = [obs.to(self.device), action.to(self.device)]

                logging_dict = self.fbrc.behavior_pretrain(states, actions)

                writer.add_scalars("Behavior", logging_dict, global_step=self.fbrc.bc_pretrain_steps)
                # wandb.log(logging_dict, step=self.fbrc.bc_pretrain_steps)
            tqdm.write(f"Current Batch: {t + 1}")
            if (t + 1) % self.cfg.save_interval == 0:
                self.fbrc.save_behavior(t+1)
        """
        # wandb.finish()
            
        # with wandb.init(project=self.cfg.project, entity="zzmtsvv", group=f"frbc_{self.cfg.group}", name=self.cfg.name):
        #    wandb.config.update({k: v for k, v in self.cfg.__dict__.items() if not k.startswith("__")})

        for t in range(self.cfg.max_timesteps):
                    
            # batch = self.buffer.sample(self.batch_size)
            for batch in tqdm(self.dataset_loader):
                # obs, action, reward, discount, obs_next = batch
                states, actions, rewards,  dones, next_states, = [x.to(self.device) for x in batch]

                logging_dict = self.fbrc.train(states,
                                               actions,
                                               rewards,
                                               next_states,
                                               dones)
                self.writer.add_scalars("fbrc", logging_dict, global_step=self.fbrc.total_iterations)
            if (t + 1) % self.cfg.save_interval == 0:
                self.fbrc.save_fisher(t+1)
            tqdm.write(f"Current Batch: {t + 1}")
            # wandb.log(logging_dict, step=self.fbrc.total_iterations)
        
        # wandb.finish()

if __name__ == "__main__":
    train_model = FBRCTrainer()
    # train_model.fbrc.load_fisher(200)
    # train_model.pca_fit()
    # sys.exit(1)
    if True:
        train_model.fit()
    else:
        env = gym.make('AdroitHandDoor-v1', max_episode_steps=400, render_mode='human')
        obs, _ = env.reset()

        for i in range(400):
            # ac = env.action_space.sample()
            obs = obs.astype(np.float32)
            # ac = train_model.sample(obs)
            with torch.no_grad():
                obs = torch.tensor(obs, device="cuda")
                ac, en = train_model.fbrc.actor(obs)
                # ac = train_model.fbrc.behavior.model.sample(obs)
                ac = ac.detach().to('cpu').numpy()
            obs = env.step(ac[0]*2-1)
            done = obs[4].get('success')
            obs = obs[0]
            env.render()
            if done:
                break
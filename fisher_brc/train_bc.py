import torch
import gymnasium as gym
from fisher_brc import fisher_save_dir
from modules import MixtureGaussianActor
from fisher_brc.conf import hopper
from fisher_brc.conf import hand_door
from diffusion.utils import get_current_datetime_string
import torch.utils.data as Data
from typing import Dict
from diffusion.utils import AttrDict
import sys
import os
import argparse
import numpy as np
# import wandb
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter # Create an instance of the object 

parser = argparse.ArgumentParser(description='train SkillPriorMdl')
parser.add_argument('--task', type=str, choices=["hopper", "hand_door"], default='hopper', help="The env to be test.")
parser.add_argument('--feature', choices=["train", "eval"], type=str, default="eval", help="train or eval")
args = parser.parse_args()

class BCEval:

    def __init__(self, cfg: AttrDict) -> None:
        self.cfg = cfg
        self.device = cfg.device
        self.bc_actor_conf = cfg.bc_actor_conf
        self.bc_eval_conf = cfg.bc_eval_conf
        self.bc_data_conf = cfg.bc_data_conf
        behavior = MixtureGaussianActor(self.bc_actor_conf.state_dim,
                                        self.bc_actor_conf.action_dim,
                                        self.bc_actor_conf.hidden_dim,
                                        self.bc_actor_conf.num_bc_actors)
        self.behavior = behavior.to(self.device)
        self.behavior.eval()
        self.env = self.create_env()
        self.exp_dir = fisher_save_dir + f"/bc/{self.bc_data_conf.data_conf.dataset_id}/{self.bc_eval_conf.load_date}"
        self.load_weight()

    def load_weight(self):
        path = self.exp_dir + f"/batch_{self.bc_eval_conf.load_batch}_guassion_bc.pth"
        behaviour_weight = torch.load(path)
        self.behavior.load_state_dict(behaviour_weight)
        print(f"Behaviour load weight from {path}")

    def create_env(self):
        if self.bc_eval_conf.env_class is None:
            return gym.make(id=self.bc_eval_conf.env_name, **self.bc_eval_conf.env_args)
        else:
            return self.bc_eval_conf.env_class(**self.bc_eval_conf.env_args)
        
    def eval(self):
        obs, _ = self.env.reset()
        done = False
        total_reward = 0
        total_step = 0
        for i in range(500):
            with torch.no_grad():
                obs = obs.astype(np.float32)
                obs = torch.tensor(obs, device=self.device)
                obs = obs.unsqueeze(0)
                pollicy = self.behavior.get_policy(state=obs)
                action = pollicy.sample().detach()[0]
                action = action.clamp(min=-1, max=1).cpu().numpy()
                obs, reward, done, terminal, info = self.env.step(action)
                total_reward += reward
                total_step += 1
                done = done or terminal
                self.env.render()
                if done:
                    break
        self.env.close()
        print(f"total steps: {total_step}, total reward: {total_reward}, average reward: {total_reward/total_step}")

class BCTrainer:
    
    def __init__(self,cfg) -> None:
        self.writer = SummaryWriter()
        self.cfg = cfg
        self.device = cfg.device
        self.bc_data_conf = cfg.bc_data_conf
        self.bc_train_conf = cfg.bc_train_conf
        self.bc_actor_conf = cfg.bc_actor_conf
        self.bc_save_conf = cfg.bc_save_conf
        behavior = MixtureGaussianActor(self.bc_actor_conf.state_dim,
                                        self.bc_actor_conf.action_dim,
                                        self.bc_actor_conf.hidden_dim,
                                        self.bc_actor_conf.num_bc_actors)
      
        self.dataset = self.bc_data_conf.data_class(self.bc_data_conf.data_conf)
        self.dataset_loader = Data.DataLoader(dataset=self.dataset,
                                              batch_size = self.bc_data_conf.batch_size,
                                              shuffle=self.bc_data_conf.shuffle,
                                              num_workers=self.bc_data_conf.num_workers,
                                              )
        
        self.behavior = behavior.to(self.device)
        self.behavior_optim = torch.optim.AdamW(self.behavior.parameters(), lr=self.bc_train_conf.behavior_lr)
        self.bc_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.behavior_optim, self.bc_train_conf.max_timesteps)

        self.bc_log_alpha = torch.tensor([0.0], dtype=torch.float32, device=self.device, requires_grad=True)
        self.bc_alpha_optim = torch.optim.Adam([self.bc_log_alpha], lr=self.bc_train_conf.alpha_lr)
        self.bc_alpha = self.bc_log_alpha.exp().detach()
        self.bc_pretrain_steps = 0
        self.target_entropy = -float(self.bc_actor_conf.action_dim)

        self.exp_dir = fisher_save_dir + f"/bc/{self.bc_data_conf.data_conf.dataset_id}/{get_current_datetime_string()}"
        try:
            os.makedirs(self.exp_dir, mode=0o777)
        except OSError as e:
            print(e)
            sys.exit(1)

    def behavior_pretrain(self, states: torch.Tensor, actions: torch.Tensor) -> Dict[str, float]:
        self.bc_pretrain_steps += 1
        log_prob, entropy = self.behavior.log_prob(states, actions, need_entropy=True)

        behavior_loss = -(self.bc_alpha * entropy + log_prob).mean()
        self.behavior_optim.zero_grad()
        behavior_loss.backward()
        self.behavior_optim.step()

        alpha_loss = (self.bc_log_alpha * (entropy.detach() - self.target_entropy)).mean()
        self.bc_alpha_optim.zero_grad()
        alpha_loss.backward()
        self.bc_alpha_optim.step()
        self.bc_lr_scheduler.step()

        self.bc_alpha = self.bc_log_alpha.exp().detach()
        
        return {
            "behavior_pretrain/loss": behavior_loss.item(),
            "behavior_pretrain/log_prob": log_prob.mean().item(),
            "behavior_pretrain/entropy": entropy.mean().item(),
            "behavior_pretrain/bc_alpha": self.bc_alpha.item(),
            "behavior_pretrain/bc_alpha_loss": alpha_loss.item(),
            "behavior_pretrain/learning_rate": self.bc_lr_scheduler.get_last_lr()[0]
        }
    
    def save_bc(self, batch_num):
        bc_dict = self.behavior.state_dict()
        path = self.exp_dir + f"/batch_{batch_num}_{self.bc_save_conf.save_name}"
        torch.save(bc_dict, path)
    
    def fit_bc(self):
        self.behavior.train()
        for t in range(self.bc_train_conf.behavior_pretrain_steps):
            for batch in tqdm(self.dataset_loader, position=0):
                obs, action = batch
                states, actions = [obs.to(self.device), action.to(self.device)]
                logging_dict = self.behavior_pretrain(states, actions)
                self.writer.add_scalars("Behavior", logging_dict, global_step=self.bc_pretrain_steps)
            tqdm.write(f"Current Batch: {t + 1}")
            if t == 0:
                self.save_bc(0)
            if (t + 1) % self.bc_save_conf.save_freq == 0:
                self.save_bc(t+1)
       
if __name__ == "__main__":
    task_conf = None
    if args.task == "hopper":
        task_conf = hopper.bc_pretrain_conf
    elif args.task == "hand_door":
        task_conf = hand_door.bc_pretrain_conf

    if args.feature == "train":
        trainer = BCTrainer(cfg=task_conf)
        trainer.fit_bc()
    elif args.feature == "eval":
        eval = BCEval(cfg=task_conf)
        eval.eval()
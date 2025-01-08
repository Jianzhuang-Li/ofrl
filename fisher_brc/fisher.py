from typing import Dict, Tuple
import torch
import os
import sys
from torch.autograd import grad
from torch.nn import functional as F
from copy import deepcopy
from config import fbrc_config
from modules import Actor, MixtureGaussianActor, EnsembledCritic
from datetime import datetime

def get_current_datetime_string():
    now = datetime.now()
    formatted_string = now.strftime("%Y_%m_%d_%H_%M")
    return formatted_string

class FisherBRC:
    def __init__(self,
                 cfg: fbrc_config,
                 actor: Actor,
                 behavior: MixtureGaussianActor,
                 critic: EnsembledCritic) -> None:
        self.cfg = cfg
        self.device = cfg.device
        self.discount = cfg.discount
        self.tau = cfg.tau

        self.actor = actor.to(self.device)
        self.actor_optim = torch.optim.AdamW(self.actor.parameters(), lr=cfg.actor_lr)

        self.target_entropy = -float(self.actor.action_dim)
        self.log_alpha = torch.tensor([0.0], dtype=torch.float32, device=self.device, requires_grad=True)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_lr)
        self.alpha = self.log_alpha.exp().detach()

        self.critic = critic.to(self.device)
        self.critic_optim = torch.optim.AdamW(self.critic.parameters(), lr=cfg.critic_lr)
        self.fisher_coef = cfg.fisher_regularization_weight
        
        self.critic_destination = deepcopy(critic).to(self.device)
        self.critic_destination_optim = torch.optim.AdamW(self.critic_destination.parameters(), lr=cfg.critic_lr)

        self.exp_dir = get_current_datetime_string()
        try:
            os.makedirs(self.cfg.model_save_path + self.exp_dir, mode=0o777)
        except OSError as e:
            print(e)
            sys.exit(1)

        with torch.no_grad():
            self.critic_target = deepcopy(critic).to(self.device)
        
        self.behavior = behavior.to(self.device)
        self.reward_bonus = cfg.reward_bonus
        self.total_iterations = 0
        self.bc_pretrain_steps = 0
    
    def train(self,
              states: torch.Tensor,
              actions: torch.Tensor,
              rewards: torch.Tensor,
              next_states: torch.Tensor,
              dones: torch.Tensor) -> Dict[str, float]:
        self.total_iterations += 1

        rewards += self.reward_bonus

        # critic step
        critic_loss, fisher_penalty = self.critic_loss(states, actions, rewards, next_states, dones)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # actor step
        actor_loss, actor_entropy, destination_loss = self.actor_loss(states)

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        self.critic_destination_optim.zero_grad()
        destination_loss.backward()
        self.critic_destination_optim.step()

        # alpha step
        alpha_loss = self.alpha_loss(states)

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        self.alpha = self.log_alpha.exp().detach()

        self.soft_critic_update()

        return {
            "fbrc/alpha_loss": alpha_loss.item(),
            "fbrc/alpha": self.alpha.item(),
            "fbrc/actor_loss": actor_loss.item(),
            "fbrc/actor_entropy": actor_entropy,
            "fbrc/critic_loss": critic_loss.item(),
            "fbrc/destination_loss": destination_loss.item(),
            "fbrc/fisher_penalty": fisher_penalty
        }
    
    def distribution_critic(self,
                            states: torch.Tensor,
                            actions: torch.Tensor,
                            use_target_networks: bool = False) -> torch.Tensor:
        if use_target_networks:
            q_values = self.critic_target(states, actions).unsqueeze(-1)
        else:
            q_values = self.critic(states, actions).unsqueeze(-1)

        log_prob, _ = self.behavior.log_prob(states, actions)
        log_prob = log_prob.unsqueeze(0)

        return q_values + log_prob
    
    def critic_loss(self,
                    states: torch.Tensor,
                    actions: torch.Tensor,
                    rewards: torch.Tensor,
                    next_states: torch.Tensor,
                    dones: torch.Tensor) -> Tuple[torch.Tensor, float]:
        next_actions, _ = self.actor(next_states)
        policy_actions, _ = self.actor(states)

        q_next = self.distribution_critic(next_states,
                                          next_actions,
                                          use_target_networks=True)
        target_q = rewards + self.discount * (1.0 - dones) * q_next.min(0).values

        current_q = self.distribution_critic(states, actions)
        
        offset = self.critic(states, policy_actions)
        offset_o1, offset_o2 = offset[0], offset[1]

        # I am doing the summation because grad function can be called with respect to a scalar,
        # but thisis not the part of the original idea
        offset_grads_o1 = grad(offset_o1.sum(), policy_actions, retain_graph=True, create_graph=True)[0]
        offset_grads_o2 = grad(offset_o2.sum(), policy_actions, retain_graph=True, create_graph=True)[0]

        o1_grad_norm = offset_grads_o1.pow(2).sum(dim=-1)
        o2_grad_norm = offset_grads_o2.pow(2).sum(dim=-1)

        fisher_penalty = (o1_grad_norm + o2_grad_norm).mean()

        loss = F.mse_loss(current_q, target_q) + self.fisher_coef * fisher_penalty
        return loss, fisher_penalty.item()
    
    def alpha_loss(self, states: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            _, log_prob = self.actor(states, need_log_prob=True)
        
        loss = -self.log_alpha * (log_prob + self.target_entropy)
        return loss.mean()

    def actor_loss(self, states: torch.Tensor) -> Tuple[torch.Tensor, float]:
        actions, log_prob = self.actor(states, need_log_prob=True)
        q_values = self.distribution_critic(states, actions)

        q_min = q_values.min(0).values
        batch_entropy = -log_prob.mean().item()

        log_prob = log_prob.unsqueeze(-1)
        assert log_prob.shape == q_min.shape, f"{log_prob.shape} != {q_min.shape}"
        loss = self.alpha * log_prob - q_min

        destination_q = self.critic_destination(states, actions.detach()).unsqueeze(-1)
        destination_loss = F.mse_loss(destination_q.min(0).values, q_min.detach())
        return loss.mean(), batch_entropy, destination_loss

    def load_behavior(self, batch_num):
        weights = torch.load(self.cfg.model_save_path + self.cfg.check_exp + f"/behavior_{batch_num}.pth")
        self.bc_alpha = weights["bc_alpha"]
        self.behavior.load_state_dict(weights["behavior"])

    def fisher_weights(self):
        return {
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor": self.actor.state_dict(),
            "alpha": self.alpha,
        }
    
    def save_fisher(self, batch_num):
        torch.save(self.fisher_weights(), self.cfg.model_save_path + self.exp_dir + f"/fisher_{batch_num}.pth")

    def load_fisher(self, batch_num):
        path = self.cfg.model_save_path + self.cfg.check_exp + f"/fisher_{batch_num}.pth"
        weights = torch.load(path) # TODO
        print(f"load fisher weights from {path}")
        self.critic.load_state_dict(weights["critic"])
        self.critic_target.load_state_dict(weights["critic_target"])
        self.actor.load_state_dict(weights["actor"])
        self.alpha = weights["alpha"]

    
    def soft_critic_update(self):
        for param, tgt_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            tgt_param.data.copy_(self.tau * param.data + (1 - self.tau) * tgt_param.data)
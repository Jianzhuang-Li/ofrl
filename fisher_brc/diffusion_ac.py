from diffusion.models import Model_Cond_Diffusion, Model_mlp_diff_embed
from diffusion.conf import hand_door_conf, hopper_conf
from diffusion.distribution import MutivariateGuassion
from conf.hopper import hopper_diffusion_conf
import torch 
import torch.utils.data as Data
import gymnasium as gym
import numpy as np

class DiffusionActor(torch.nn.Module):

    def __init__(self, conf_):
        super().__init__()
        self.conf = conf_
        self.model_save = conf_.model_save
        self.nn_conf = conf_.nn_model_conf
        self.device = conf_.device  # conf-device
        self.model_load = conf_.model_load
        self.min_action = torch.tensor(conf_.min_action, device=self.device)
        self.max_action = torch.tensor(conf_.max_action, device=self.device)
        self.model_val = conf_.model_val
        self.eps = 1e-6
        # self.fix_log_sigma = torch.ones((1,), device=self.device)
        self.fix_log_sigma = torch.tensor(conf_.fix_log_sigma, device=self.device)
        self.nn_model = Model_mlp_diff_embed(self.nn_conf.x_shape, self.nn_conf.n_hidden, self.nn_conf.y_dim, self.nn_conf.embed_dim).to(self.device)
        self.model = Model_Cond_Diffusion(self.nn_model, self.conf.model_conf)
        self.load_check()
        self.model.eval() # TODO  不是eval模式的话，实测效果差？

    def load_check(self):
        path = self.model_load.exp_path + f"/batch_{self.model_load.batch_num}_" +  self.model_save.file_name
        self.model.load_state_dict(torch.load(path))
        print(f"load checkpoint {self.model_load.batch_num} from {path}")

    def sample(self, states):
        # x_eval_r = states.repeat(2, 1)
        if not isinstance(states, torch.Tensor):
            states = torch.tensor(states, device=self.device)
        if len(states.shape) == 1:
            x_eval_r = states.unsqueeze(0)
        else:
            x_eval_r = states
        if self.model_val.extra_diffusion_steps == 0:
            y_predict = self.model.sample(x_eval_r)
        else:
            y_predict = self.model.sample_extra(x_eval_r, extra_steps=self.model_val.extra_diffusion_steps)
        if len(states.shape) == 1:
            sampled_actions = y_predict[0]
        else:
            sampled_actions = y_predict
        sampled_actions = sampled_actions.clamp(self.min_action + self.eps, self.max_action - self.eps)
        sampled_actions = sampled_actions.detach().cpu().numpy()
        return sampled_actions

    def log_prob(self, states, actions, log_sigma=None, need_entropy=False):
        if log_sigma is None:
            log_sigma = self.fix_log_sigma
        sampled_actions = self.sample(states=states)
        dist = MutivariateGuassion(mu=sampled_actions, log_sigma=log_sigma)
        actions = actions.clamp(self.min_action + self.eps, self.max_action - self.eps)
        log_prob_ = dist.log_prob(val=actions).unsqueeze(-1)
        entropy_ = None
        if need_entropy:
            entropy_ = dist.entropy().unsqueeze(-1)
        return log_prob_, entropy_

       

if __name__ == "__main__":
    actor = DiffusionActor(conf_=hopper_diffusion_conf)
    env = gym.make(hopper_conf.env.env_name, render_mode = hopper_conf.env.render_mode)
    obs, _ = env.reset()

    for i in range(500):
        # ac = env.action_space.sample()
        obs = obs.astype(np.float32)
        with torch.no_grad():
            obs = torch.tensor(obs, device=hopper_diffusion_conf.device)
            ac = actor.sample(obs)
        obs, rew, done, ter, info = env.step(ac)
        env.render()
        if done or ter:
            break
    env.close()

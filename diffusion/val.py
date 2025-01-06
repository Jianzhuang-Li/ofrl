from diffusion.models import Model_Cond_Diffusion, Model_mlp_diff_embed
from envs.zone_hot import ZoneHotEnv
from itertools import product
from diffusion.conf import hand_door_conf
from diffusion.conf import zone_hot_conf
from diffusion.conf import hopper_conf
from diffusion import diffusion_save_dir
from tqdm import tqdm
import torch.optim
import numpy as np
import os
import sys
import argparse
import gymnasium as gym
import torch.utils.data as Data

parser = argparse.ArgumentParser(description='train SkillPriorMdl')
parser.add_argument('--task', type=str, default='hopper', help="The env to be test.")
parser.add_argument('--savedate', type=str, default=None, help="model save date.")
parser.add_argument('--checknum', type=int, default=200, help="batch number to load.")
args = parser.parse_args()

class Val:
    def __init__(self, conf_, val_args):
        self.val_args = val_args
        self.conf = conf_
        self.model_save = conf_.model_save
        self.model_val = conf_.model_val
        self.data_specific = conf_.data_specific    # conf-data_specific
        self.nn_conf = conf_.nn_model_conf
        self.model_conf = conf_.model_conf
        self.device = conf_.device  # conf-device
        self.nn_model = Model_mlp_diff_embed(self.nn_conf.x_shape, self.nn_conf.n_hidden, self.nn_conf.y_dim, self.nn_conf.embed_dim).to(self.device)
        self.model = Model_Cond_Diffusion(self.nn_model, self.conf.model_conf)
        self.exp_dir = diffusion_save_dir + f"/{self.data_specific.data_conf.dataset_id}"
        self.load_check()
        
    def load_check(self):
        path = self.exp_dir + f"/{self.val_args.savedate}/batch_{self.val_args.checknum}_" +  self.model_save.file_name
        self.model.load_state_dict(torch.load(path))
        print(f"load checkpoint {self.val_args.checknum} from {path}")
    

    def eval(self):
        self.model.eval()

    def sample(self, x_eval_):
        x_eval = torch.tensor(x_eval_).to(self.device)
        with torch.no_grad():
            x_eval_r = x_eval.repeat(2, 1)
            # x_eval_r = x_eval
            if self.model_val.extra_diffusion_steps == 0:
                y_predict = self.model.sample(x_eval_r).detach().cpu().numpy()
            else:
                y_predict = self.model.sample_extra(x_eval_r, extra_steps=self.model_val.extra_diffusion_steps).detach().cpu().numpy()

        return y_predict


if __name__ == '__main__':
    # d4rl和现在的gym环境不兼容，跑测试的时候如果渲染模式选择"human", 环境要切换到“spirl-flow”, 并注释掉conf中的数据集
    conf = None
    if args.task == "hand_door":
        conf = hand_door_conf.conf_hand_door
    elif args.task == "hopper":
        conf = hopper_conf.conf_hopper_expert
    elif args.task == "zone_hot":
        conf = zone_hot_conf.conf_zone_hot
    to_train = Val(conf, args)
    to_train.eval()
    env = gym.make(conf.env.env_name, render_mode =conf.env.render_mode)
    obs, _ = env.reset()

    for i in range(500):
        # ac = env.action_space.sample()
        obs = obs.astype(np.float32)
        with torch.no_grad():
            ac = to_train.sample(obs)[0]
        obs, rew, done, ter, info = env.step(ac)
        env.render()
        if done or ter:
            break
    env.close()

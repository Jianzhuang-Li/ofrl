from diffusion.models import Model_Cond_Diffusion, Model_mlp_diff_embed
from diffusion.utils import get_current_datetime_string
from memory_profiler import profile
from itertools import product
from diffusion.conf import hand_door_conf
from diffusion.conf import zone_hot_conf
from diffusion.conf import hopper_conf
from diffusion.conf import walker2d_conf
from tqdm import tqdm
import torch.optim
import numpy as np
import os
import sys
import argparse
import torch.utils.data as Data
import gymnasium as gym


parser = argparse.ArgumentParser(description='train SkillPriorMdl')
parser.add_argument('--task', type=str, default='hopper', help="Which task to train")
parser.add_argument('--load_check', type=bool, default=False, help="If load checkpoint, if set it true, set the exp dir and batch")
parser.add_argument('--create_exp_dir', type=bool, default=True, help="If create new dir for this experience, if false, use checkpoint dir.")
parser.add_argument('--exp_dir', type=str, default=None, help="the address where to load checkpoint")
parser.add_argument('--check_batch', type=int, help="the batch number to load")
args = parser.parse_args()
EXP_NAME = "diffusion"


class Train:
    def __init__(self, conf_, parser_args):
        self.conf = conf_
        self.model_save = conf_.model_save
        self.model_val = conf_.model_val
        self.data_specific = conf_.data_specific    # conf-data_specific
        self.nn_conf = conf_.nn_model_conf
        self.model_conf = conf_.model_conf
        self.device = conf_.device  # conf-device
        self.nn_model = Model_mlp_diff_embed(self.nn_conf.x_shape, self.nn_conf.n_hidden, self.nn_conf.y_dim, self.nn_conf.embed_dim).to(self.device)
        self.model = Model_Cond_Diffusion(self.nn_model, self.conf.model_conf)
        self.optim = None
        self.batch_start = 0
        self.exp_dir = None
        self.current_path = os.path.dirname(os.path.abspath(__file__))
        base_dir = self.current_path + "/out/" + self.data_specific.data_conf.dataset_id
        if parser_args.create_exp_dir:
            self.exp_dir = get_current_datetime_string()
            try:
                self.exp_dir =  base_dir +  "/" + self.exp_dir
                os.makedirs(self.exp_dir, mode=0o777)
            except OSError as e:
                print(e)
                sys.exit(1)
        else:
            self.exp_dir = base_dir +  "/" + parser_args.exp_dir
        assert self.exp_dir is not None and os.path.exists(self.exp_dir)

        if parser_args.load_check:
            self.load_check(parser_args.check_batch)

        self.dataset = self.data_specific.data_class(self.conf.data_specific.data_conf)
        self.dataset_loader = Data.DataLoader(dataset=self.dataset,
                                              batch_size=self.data_specific.batch_size,
                                              shuffle=self.data_specific.shuffle,
                                              num_workers=self.data_specific.num_workers,
                                              )
        
    def load_check(self, batch_check):
        self.batch_start += batch_check
        path =  self.exp_dir + f"/batch_{batch_check}_{self.model_save.file_name}"
        assert os.path.exists(path)
        self.model.load_state_dict(torch.load(path))
        print(f"Diffusion model load checkpoint {batch_check} from {path}")
    

    def eval(self):
        self.model.eval()

    def sample(self, x_eval_):
        x_eval = torch.tensor(x_eval_).to(self.device)
        with torch.no_grad():
            x_eval_r = x_eval.repeat(2, 1)
            if self.model_val.extra_diffusion_steps == 0:
                y_predict = self.model.sample(x_eval_r).detach().cpu().numpy()
            else:
                y_predict = self.model.sample_extra(x_eval_r, extra_steps=self.model_val.extra_diffusion_steps).detach().cpu().numpy()

        return y_predict

    def train(self):
        self.model.to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), self.conf.lrate) 
        self.model.train()

        for ep in range(self.batch_start, self.conf.n_epoch + 1):
            
            self.optim.param_groups[0]["lr"] = self.conf.lrate * ((np.cos((ep/self.conf.n_epoch)*np.pi)+1)/2)

            # train loop
            loss_ep, n_batch = 0, 0
            for x_batch, y_batch in tqdm(self.dataset_loader):
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                loss = self.model.loss_on_batch(x_batch, y_batch)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                loss_ep += loss.detach().item()
                n_batch += 1

            print("batch:{}".format(ep), "loss:{}".format(loss_ep), "mean_loss:{}".format(loss_ep/n_batch))
            if((ep + 1) % self.model_save.save_freq == 0) or ep == 0:
                torch.save(self.model.state_dict(), self.exp_dir + f"/batch_{ep+1}_{self.model_save.file_name}")


if __name__ == '__main__':
    # to_train = Train(hand_door_conf.conf_hand_door)
    # to_train = Train(zone_hot_conf.conf_zone_hot)
    # to_train = Train(hopper_conf.conf_hopper_expert)
    conf = None
    if args.task == 'hopper':
        conf = hopper_conf.conf_hopper_expert
    elif args.task == 'hand_door':
        conf = hand_door_conf.conf_hand_door
    elif args.task == 'zone_hot':
        conf = zone_hot_conf.conf_zone_hot
    to_train = Train(conf, args)
    to_train.train()

    """
    env = gym.make('AdroitHandDoor-v1', max_episode_steps=400, render_mode='human')
    obs, _ = env.reset()

    for i in range(400):
        # ac = env.action_space.sample()
        obs = obs.astype(np.float32)
        ac = to_train.sample(obs)
        obs = env.step(ac[0]*2-1)
        done = obs[4].get('success')
        obs = obs[0]
        env.render()
        if done:
            break
    """
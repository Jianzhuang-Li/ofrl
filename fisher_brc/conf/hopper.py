from diffusion.utils import AttrDict
from diffusion.conf.hopper_conf import nn_model_conf, model_conf, env, data_conf, model_save, model_val
from diffusion.conf.hopper_conf import data_specific
# from diffusion.data.hopper_expert import HopperDataSet
from diffusion import diffusion_save_dir
device = "cuda"

model_load = AttrDict({
    "exp_path": f"{diffusion_save_dir}/{data_conf.dataset_id}/2025_01_06_12_22",
    "batch_num": 200
})

hopper_diffusion_conf = AttrDict({
    "device": device,
    "model_save": model_save,
    "nn_model_conf": nn_model_conf,
    "model_conf": model_conf,
    "diffusion_save_dir": diffusion_save_dir,
    "model_load": model_load,
    "model_val": model_val,
    "fix_log_sigma": 2.0,
    "min_action": env.min_action,
    "max_action": env.max_action
})

# behaviour clone pre-train config
data_class_conf = AttrDict({
    "dataset_id": 'hopper-expert-v2',
    "device": device,
    'phase': 'train'
})

bc_data_conf = AttrDict({
    #"data_class": HopperDataSet,
    "state_dim": 11,
    "action_dim": 3,
    "data_conf": data_class_conf,
    "batch_size": 256,
    "shuffle": True,
    "num_workers": 0
})

bc_actor_conf = AttrDict({
    "state_dim": bc_data_conf.state_dim,
    "action_dim": bc_data_conf.action_dim,
    "hidden_dim": 256,
    "num_bc_actors": 5
})

bc_train_conf = AttrDict({
    "behavior_lr": 3e-4,
    "alpha_lr": 3e-4,
    "batch_size": 256,
    "behavior_pretrain_steps": int(2e2),
    "max_timesteps": 1e6
})

bc_save_conf = AttrDict({
    "save_freq": 20,
    "save_name": "guassion_bc.pth"
})

bc_env_conf = AttrDict({
    "render_mode": "human",
})

bc_eval_conf = AttrDict({
    "env_class": None,
    "env_name": "Hopper-v4",
    "env_args": bc_env_conf,
    "load_batch": 40,
    "load_date": "2025_01_07_19_28"
})

bc_pretrain_conf = AttrDict({
    "device": device,
    "bc_data_conf": bc_data_conf,
    "bc_train_conf": bc_train_conf,
    "bc_save_conf": bc_save_conf,
    "bc_actor_conf": bc_actor_conf,
    "bc_eval_conf": bc_eval_conf
})
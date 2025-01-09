from diffusion.utils import AttrDict
from diffusion.data.hopper_expert import HopperDataSet
# hand door conf
device = 'cuda'
# Used for test
env = AttrDict({
    "env_name": "Hopper-v4",
    "render_mode": "human",
    "min_action": -1.0,
    "max_action": 1,
})
# Used by data_specific.data_class.
# The contains according to your own data class.
data_conf = AttrDict({
    'dataset_id': 'hopper-expert-v2',
    'device': device,
    'phase': 'train'
})
# Used by pytorch dataloader.
data_specific = AttrDict({
    'data_class': HopperDataSet,
    'observation_dim': 11,
    'action_dim': 3,
    'data_conf': data_conf,
    'batch_size': 256,
    'shuffle': True,
    'num_workers': 0,
})

nn_model_conf = AttrDict({
    'x_shape': data_specific.observation_dim,
    'y_dim': data_specific.action_dim,
    'n_hidden': 16,
    'embed_dim': 128,
    'net_type': "fc"
})

model_conf = AttrDict({
    'betas': (1e-4, 0.02),
    'n_t': 50,
    'device': device,
    'drop_prob': 0.1,
    'load_check': False,
    'batch_check': '',
    'guide_w': 0.0,
    'y_dim': data_specific.action_dim,
})

model_save = AttrDict({
    'save_path': './model_save/', # not used
    'file_name': data_conf.dataset_id + '.pt',
    'save_freq': 20,
    'exp_dir': '2025_01_02_19_08'   # not used in diffusion model now
})

model_val = AttrDict({
    'model_path': model_save.save_path + model_save.file_name,
    'extra_diffusion_steps': 32,
})

conf_hopper_expert = AttrDict({
    'device': device,
    'lrate': 1e-4,
    'n_T': 50,
    'n_epoch': 200,
    'batch_size': 32,
    'data_specific': data_specific,
    'nn_model_conf': nn_model_conf,
    'model_conf': model_conf,
    'model_save': model_save,
    'model_val': model_val,
    'env': env
})

from dataclasses import dataclass
import torch


@dataclass
class fbrc_config:
    # Experiment
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_name: str = "Adroit Door"
    seed: int = 42

    state_dim: int = 39
    action_dim: int = 28

    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    behavior_lr: float = 3e-4
    batch_size: int = 256
    buffer_size: int = 1000000
    discount: float = 0.99
    hidden_dim: int = 256
    max_action: float = 1.0
    # max_timesteps: int = int(5e5)
    max_timesteps: int = int(2e2)
    # behavior_pretrain_steps: int = int(5e5)
    behavior_pretrain_steps: int = int(2e2)
    tau: float = 5e-3

    reward_bonus: float = 5.0
    fisher_regularization_weight: float = 0.1
    num_bc_actors: int = 5

    project: str = "FisherBRC"
    group: str = dataset_name
    name: str = dataset_name + "_" + str(seed)


    save_interval = 10
    model_save_path = "./fisher_model/"
    check_exp = "2025_01_02_11_26"
    

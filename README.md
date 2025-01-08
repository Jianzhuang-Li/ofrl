# Offline Reinforcement Learning with Diffusion model and Fisher Divergence

1. Train a diffusion behaviour clone model

    ```
    python diffusion/train.py --task=hopper["zone_hot", "hand_door"]
    ```

2. Train Fisher_Offline Deep reinforcement Learning with Diffusion model.

    ```
    python fisher_brc/tainer.py --task=hopper --bc=diffusion --feature=train
    ```

3. Or train Offline Reinforcement learning with gaussian-mix model.

    ```
    # Train a Gaussian-Mix Behaviour Clone model.
    python fisher_brc/train_bc.py --task=hopper --feature=train
    # Train offline RL with Gaussian-Mix model.
    python fisher_brc/trainer.py --task=hopper --bc=gaussian --feature=train
    ```
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
# the path where save the diffusion model weights.
diffusion_save_dir = os.path.join(current_dir, "out")
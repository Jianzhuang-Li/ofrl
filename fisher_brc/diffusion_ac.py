from diffusion.models import Model_Cond_Diffusion, Model_mlp_diff_embed
from diffusion.conf import hand_door_conf
from diffusion.distribution import MutivariateGuassion
import torch 
import torch.utils.data as Data

EXP_NAME = "diffusion"

class DiffusionActor(torch.nn.Module):

    def __init__(self, conf_):
        super().__init__()
        self.conf = conf_
        self.model_save = conf_.model_save
        self.nn_conf = conf_.nn_model_conf
        self.device = conf_.device  # conf-device
        self.min_action = torch.tensor(-1, device="cuda")
        self.max_action = torch.tensor(1, device="cuda")
        self.eps = 1e-6
        # self.fix_log_sigma = torch.ones((1,), device=self.device)
        self.fix_log_sigma = torch.tensor(2.0, device=self.device)
        self.nn_model = Model_mlp_diff_embed(self.nn_conf.x_shape, self.nn_conf.n_hidden, self.nn_conf.y_dim, self.nn_conf.embed_dim).to(self.device)
        self.model = Model_Cond_Diffusion(self.nn_model, self.conf.model_conf)
        self.load_check(300)

    def load_check(self, batch_check):
        path = self.model_save.save_path + f"{EXP_NAME}/" + f"batch_{batch_check}_" +  self.model_save.file_name
        self.model.load_state_dict(torch.load(path))
        print(f"load checkpoint {batch_check} from {path}")

    def log_prob(self, states, actions, log_sigma=None, need_entropy=False):
        if log_sigma is None:
            log_sigma = self.fix_log_sigma
        sampled_actions = self.model.sample(x_batch=states)
        sampled_actions = sampled_actions.clamp(self.min_action + self.eps, self.max_action - self.eps)
        dist = MutivariateGuassion(mu=sampled_actions, log_sigma=log_sigma)
        actions = actions.clamp(self.min_action + self.eps, self.max_action - self.eps)
        log_prob_ = dist.log_prob(val=actions).unsqueeze(-1)
        entropy_ = None
        if need_entropy:
            entropy_ = dist.entropy().unsqueeze(-1)
        return log_prob_, entropy_

       

if __name__ == "__main__":
    m = DiffusionActor(hand_door_conf.conf_hand_door).to("cuda")
    conf_ = hand_door_conf.conf_hand_door
    dataset = conf_.data_specific.data_class(conf_.data_specific.data_conf)
    dataset_loader = Data.DataLoader(dataset=dataset,
                                    batch_size=conf_.data_specific.batch_size,
                                    shuffle=conf_.data_specific.shuffle,
                                    num_workers=conf_.data_specific.num_workers,
                                    )
    i = 0
    for batch_data in dataset_loader:
        
        states, actions = batch_data
        states = states.to(m.device)
        actions = actions.to(m.device)
        with torch.no_grad():
            log_prob, entropy = m.log_prob(states, actions, need_entropy=True)
        i += 1

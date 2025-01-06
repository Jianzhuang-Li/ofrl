# Auther: Jianzhuang Li
# Date: 2024.12.18
# Reference: https://github.com/microsoft/Imitating-Human-Behaviour-w-Diffusion

import torch.nn as nn
import torch
from diffusion.distribution import  MutivariateGuassion

class MultiLayerLinear(torch.nn.Module):

    def __init__(self, 
                input_dim,
                hidden_dim,
                output_dim,
                layers_num,
                hidden_normal=False,
                active= torch.nn.ReLU
            ):
        super(MultiLayerLinear, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.layers_num = layers_num
        self.active = active
        self.hidden_normal = hidden_normal

        self.net = torch.nn.Sequential()
        self.net.append(torch.nn.Linear(input_dim, hidden_dim))
        self.net.append(torch.nn.ReLU())
        for _ in range(self.layers_num):
            self.net.append(torch.nn.Linear(hidden_dim, hidden_dim))
            self.net.append(torch.nn.ReLU())
        self.net.append(torch.nn.Linear(hidden_dim, output_dim))

    def forward(self, input):
        return self.net(input)

class TimeSiren(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super().__init__()
        self.lin1 = nn.Linear(input_dim, emb_dim, bias=False)
        self.lin2 = nn.Linear(emb_dim, emb_dim)

    def forward(self, x):
        x = torch.sin(self.lin1(x))
        x = self.lin2(x)
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, trans_emb_dim, transformer_dim, n_heads):
        super(TransformerEncoderBlock, self).__init__()
        # mainly going off of https://jalammar.github.io/illustrated-transformer/

        self.trans_emb_dim = trans_emb_dim
        self.transformer_dim = transformer_dim
        self.n_heads = n_heads

        self.input_to_qkv1 = nn.Linear(self.trans_emb_dim, self.transformer_dim * 3)
        self.multi_head_attn1 = nn.MultiheadAttention(self.transformer_dim, num_heads=self.n_heads)
        self.attn1_to_fcn = nn.Linear(self.transformer_dim, self.trans_emb_dim)
        self.attn1_fcn = nn.Sequential(
            nn.Linear(self.trans_emb_dim, self.trans_emb_dim * 4),
            nn.GELU(),
            nn.Linear(self.trans_emb_dim * 4, self.trans_emb_dim),
        )
        self.norm1a = nn.BatchNorm1d(self.trans_emb_dim)
        self.norm1b = nn.BatchNorm1d(self.trans_emb_dim)

    def split_qkv(self, qkv):
        assert qkv.shape[-1] == self.transformer_dim * 3
        q = qkv[:, :, :self.transformer_dim]
        k = qkv[:, :, self.transformer_dim: 2 * self.transformer_dim]
        v = qkv[:, :, 2 * self.transformer_dim:]
        return q, k, v

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        qkvs1 = self.input_to_qkv1(inputs)
        # shape out = [3, batch_size, transformer_dim*3]

        qs1, ks1, vs1 = self.split_qkv(qkvs1)
        # shape out = [3, batch_size, transformer_dim]

        attn1_a = self.multi_head_attn1(qs1, ks1, vs1, need_weights=False)
        attn1_a = attn1_a[0]
        # shape out = [3, batch_size, transformer_dim = trans_emb_dim x n_heads]

        attn1_b = self.attn1_to_fcn(attn1_a)
        attn1_b = attn1_b / 1.414 + inputs / 1.414  # add residual
        # shape out = [3, batch_size, trans_emb_dim]

        # normalise
        attn1_b = self.norm1a(attn1_b.transpose(0, 2).transpose(0, 1))
        attn1_b = attn1_b.transpose(0, 1).transpose(0, 2)
        # batch_norm likes shape = [batch_size, trans_emb_dim, 3]
        # so have to shape like this, then return

        # fully connected layer
        attn1_c = self.attn1_fcn(attn1_b) / 1.414 + attn1_b / 1.414
        # shape out = [3, batch_size, trans_emb_dim]

        # normalise
        # attn1_c = self.norm1b(attn1_c)
        attn1_c = self.norm1b(attn1_c.transpose(0, 2).transpose(0, 1))
        attn1_c = attn1_c.transpose(0, 1).transpose(0, 2)
        return attn1_c


class Model_mlp_diff_embed(nn.Module):
    # this model embeds x, y, t, before input into a fc NN (w residuals)
    def __init__(self, x_dim, n_hidden, y_dim, embed_dim, output_dim=None, use_prev=False):
        super(Model_mlp_diff_embed, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.n_hidden = n_hidden    # input embedding dimension
        self.embed_dim = embed_dim
        self.use_prev = use_prev    # whether x contains previous timestep
        if output_dim is None:
            self.output_dim = self.y_dim    # by default, just output size of action space
        else:
            self.output_dim = output_dim    # sometime overwrite

        # embedding NNS
        if self.use_prev:
            self.x_embed_nn = nn.Sequential(
                nn.Linear(int(x_dim/2), self.embed_dim),
                nn.LeakyReLU(),
                nn.Linear(self.embed_dim, self.embed_dim),
            )
        else:
            self.x_embed_nn = nn.Sequential(
                nn.Linear(x_dim, self.embed_dim),
                nn.LeakyReLU(),
                nn.Linear(self.embed_dim, self.embed_dim),
            )

        self.y_embed_nn = nn.Sequential(
            nn.Linear(y_dim, self.embed_dim),
            nn.LeakyReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )
        self.t_embed_nn = TimeSiren(1, self.embed_dim)

        # transformer layers
        self.n_heads = 16
        self.trans_emb_dim = 64
        self.transformer_dim = self.trans_emb_dim * self.n_heads

        self.t_to_input = nn.Linear(self.embed_dim, self.trans_emb_dim)
        self.y_to_input = nn.Linear(self.embed_dim, self.trans_emb_dim)
        self.x_to_input = nn.Linear(self.embed_dim, self.trans_emb_dim)

        self.pos_embed = TimeSiren(1, self.trans_emb_dim)

        self.transformer_block1 = TransformerEncoderBlock(self.trans_emb_dim, self.transformer_dim, self.n_heads)
        self.transformer_block2 = TransformerEncoderBlock(self.trans_emb_dim, self.transformer_dim, self.n_heads)
        self.transformer_block3 = TransformerEncoderBlock(self.trans_emb_dim, self.transformer_dim, self.n_heads)
        self.transformer_block4 = TransformerEncoderBlock(self.trans_emb_dim, self.transformer_dim, self.n_heads)

        if self.use_prev:
            self.final = nn.Linear(self.trans_emb_dim*4, self.output_dim)   # final layer paras
            # self.distribution_encoder = MultiLayerLinear(self.trans_emb_dim*4, 64, self.output_dim * 2, 4)
        else:
            # self.distribution_encoder = MultiLayerLinear(self.trans_emb_dim*3, 64, self.output_dim * 2, 4)
            self.final = nn.Linear(self.trans_emb_dim*3, self.output_dim)

    def forward(self, y, x, t, context_mask):
        # embed y, x, t
        if self.use_prev:
            x_e = self.x_embed_nn(x[:, :int(self.x_dim/2)])
            x_e_prev = self.x_embed_nn(x[:, int(self.x_dim/2):])
        else:
            x_e = self.x_embed_nn(x)    # no prev hist
            x_e_prev = None

        y_e = self.y_embed_nn(y)
        t_e = self.t_embed_nn(t)

        # mask out context embedding, x_e, if context_mask == 1
        context_mask = context_mask.repeat(x_e.shape[1], 1).T
        x_e = x_e * (-1 * (1 - context_mask))
        if self.use_prev:
            x_e_prev = x_e_prev * (-1 * (1 - context_mask))

        # pass through transformer encoder
        net_output = self.forward_transformer(x_e, x_e_prev, y_e, t_e, x, y, t)

        return net_output

    def forward_fcnn(self, x_e, x_e_prev, y_e, t_e, x, y, t):
        if self.use_prev:
            net_input = torch.cat((x_e, x_e_prev, y_e, t_e), 1)
        else:
            net_input = torch.cat((x_e, y_e, t_e), 1)
        nn1 = self.fc1(net_input)
        nn2 = self.fc2(torch.cat((nn1 / 1.414, y, t), 1)) + nn1 / 1.414  # residual and concat inputs again
        nn3 = self.fc3(torch.cat((nn2 / 1.414, y, t), 1)) + nn2 / 1.414
        net_output = self.fc4(torch.cat((nn3, y, t), 1))
        return net_output

    def forward_transformer(self, x_e, x_e_prev, y_e, t_e, x, y, t):
        # roughly following this: https://jalammar.github.io/illustrated-transformer/

        t_input = self.t_to_input(t_e)
        y_input = self.y_to_input(y_e)
        x_input = self.x_to_input(x_e)
        x_input_prev = None
        if self.use_prev:
            x_input_prev = self.x_to_input(x_e_prev)
        # shape out = [batch_size, trans_emb_dim]

        # add 'positional' encoding 添加“位置”编码
        # note, here position refers to order tokens are fed into transformer
        t_input += self.pos_embed(torch.zeros(x.shape[0], 1).to(x.device) + 1.0)
        y_input += self.pos_embed(torch.zeros(x.shape[0], 1).to(x.device) + 2.0)
        x_input += self.pos_embed(torch.zeros(x.shape[0], 1).to(x.device) + 3.0)

        if self.use_prev:
            x_input_prev += self.pos_embed(torch.zeros(x.shape[0], 1).to(x.device) + 4.0)

        if self.use_prev:
            inputs1 = torch.cat(
                (
                    t_input[None, :, :],
                    y_input[None, :, :],
                    x_input[None, :, :],
                    x_input_prev[None, :, :],
                ),
                0,
            )
        else:
            # print(t_input.shape, y_input.shape, x_input.shape)
            inputs1 = torch.cat(
                (
                    t_input[None, :, :],
                    y_input[None, :, :],
                    x_input[None, :, :]), 0)
        # shape out = [3, batch_size, trans_emb_dim]

        block1 = self.transformer_block1(inputs1)
        block2 = self.transformer_block2(block1)
        block3 = self.transformer_block3(block2)
        block4 = self.transformer_block4(block3)

        # flatten and add final linear layer
        # transformer_out = block2
        transformer_out = block4
        transformer_out = transformer_out.transpose(0, 1)  # roll batch to first dim
        # shape out = [batch_size, 3, trans_emb_dim]
        # 将多维张量（tensor）转换为一维张量的函数
        flat = torch.flatten(transformer_out, start_dim=1, end_dim=2)
        # shape out = [batch_size, 3 x trans_emb_dim]

        out = self.final(flat)
        # out = MutivariateGuassion(self.distribution_encoder(flat))
        # shape out = [batch_size, n_dim]
        return out


def ddpm_schedules(beta1, beta2, n_t, is_linear=True):
    """
    :param beta1:
    :param beta2:
    :param n_t:
    :param is_linear:
    :return:
    return pre-computed schedules for DDPM sampling, training process
    """
    if is_linear:
        beta_t = (beta2 - beta1) * torch.arange(-1, n_t, dtype=torch.float32) / (n_t-1) + beta1
    else:
        beta_t = ((beta2 - beta1) * torch.square(torch.arange(-1, n_t, dtype=torch.float32)) /
                  torch.max(torch.square(torch.range(-1, n_t, dtype=torch.float32)))+beta1)
    beta_t[0] = beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alpha_bar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrt_ab = torch.sqrt(alpha_bar_t)
    one_over_sqrt_a = 1 / torch.sqrt(alpha_t)

    sqrt_mab = torch.sqrt(1 - alpha_bar_t)
    mab_over_sqrt_mab_inv = (1 - alpha_t) / sqrt_mab

    return {
        'alpha_t': alpha_t,  # \alpha_t
        'one_over_sqrt_a': one_over_sqrt_a,  # 1/\sqrt{\alpha_t}
        'sqrt_bata_t': sqrt_beta_t,  # \sqrt{\bata_t}
        'alpha_bar_t': alpha_bar_t,  # \bar{\alpha_t}
        'sqrt_ab': sqrt_ab,  # \sqrt{\bar{\alpha_t}}
        'sqrt_mab': sqrt_mab,   # \sqrt{1-\bar{\alpha_t}}
        'mab_over_sqrt_mab': mab_over_sqrt_mab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }


class Model_Cond_Diffusion(nn.Module):
    def __init__(self, nn_model, model_conf_, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nn_model = nn_model    # model used to predict noise
        self.model_conf = model_conf_
        self.betas = model_conf_.betas
        self.n_t = model_conf_.n_t
        self.drop_prob = model_conf_.drop_prob
        self.guide_w = model_conf_.guide_w
        self.device = model_conf_.device    # device
        self.loss_mse = nn.MSELoss()
        for k, v in ddpm_schedules(self.betas[0], self.betas[1], self.model_conf.n_t).items():
            self.register_buffer(k, v)

    def loss_on_batch(self, x_batch, y_batch):
        # randomly choose a step t
        _ts = torch.randint(1, self.n_t + 1, (y_batch.shape[0], 1)).to(self.device)

        # dropout context with some probability, use bernoulli distribution
        context_mask = torch.bernoulli(torch.zeros(x_batch.shape[0]) + self.drop_prob).to(self.device)

        # randomly sample some noise, noise ~ N(0, 1)
        noise = torch.randn_like(y_batch).to(self.device)

        # y_batch_mu = self.sqrt_ab[_ts] * y_batch
        # y_batch_log_sigma = 0.5 * torch.log(self.sqrt_mab[_ts] * torch.ones_like(y_batch))


        # add noise to clean target actions
        y_t = self.sqrt_ab[_ts] * y_batch + self.sqrt_mab[_ts] * noise
        # y_t = MutivariateGuassion(y_batch_mu, y_batch_log_sigma).sample()

        # predict the noise by nn_model
        noise_predict_batch = self.nn_model(y_t, x_batch, _ts / self.n_t, context_mask)

        # return mse between predicted and true noise
        return self.loss_mse(noise, noise_predict_batch)
        # return self.loss_mse(y_batch, noise_predict_batch)
    
    def sample_dis(self, x_batch, return_y_trace=False, extract_embedding=False):
         # also use this as a shortcut to avoid doubling batch when guide_w is zero
        is_zero = False
        if -1e-3 < self.guide_w < 1e-3:
            is_zero = True
        # how many noisy actions to begin with
        n_sample = x_batch.shape[0]

        y_shape = (n_sample, self.model_conf.y_dim)
        # sample initial noise, y_0 ~ N(0, 1),
        y_i = torch.randn(y_shape).to(self.device)
        if not is_zero:
            if len(x_batch.shape) > 2:
                # repeat x_batch twice, so can use guided diffusion
                x_batch = x_batch.repeat(2, 1, 1, 1)
            else:
                # repeat x_batch twice, so can use guided diffusion
                x_batch = x_batch.repeat(2, 1)

            # half of context will be zero
            context_mask = torch.zeros(x_batch.shape[0]).to(self.device)
            context_mask[n_sample:] = 1.0  # makes second half of batch context free
        else:
            context_mask = torch.zeros(x_batch.shape[0]).to(self.device)

        if extract_embedding:
            x_embed = self.nn_model.embed_context(x_batch)

        # run denoising chain
        y_i_store = []  # if want to trace how y_i evolved
        for i in range(self.n_t, 0, -1):
            t_is = torch.tensor([i / self.n_t]).to(self.device)
            t_is = t_is.repeat(n_sample, 1)

            if not is_zero:
                # double batch
                y_i = y_i.repeat(2, 1)
                t_is = t_is.repeat(2, 1)

            # z = torch.randn(y_shape).to(self.device) if i > 1 else 0

            # split predictions and compute weighting
            if extract_embedding:
                eps = self.nn_model(y_i, x_batch, t_is, context_mask, x_embed)
            else:
                eps = self.nn_model(y_i, x_batch, t_is, context_mask)
            # if not is_zero:
            #     eps1 = eps[:n_sample]
            #     eps2 = eps[n_sample:]
            #     eps = (1 + self.guide_w) * eps1 - self.guide_w * eps2
            #     y_i = y_i[:n_sample]
            # y_i = self.one_over_sqrt_a[i] * (y_i - eps * self.mab_over_sqrt_mab[i]) + self.sqrt_bata_t[i] * z
            y_i = eps
            if return_y_trace and (i % 20 == 0 or i == self.n_t or i < 8):
                y_i_store.append(y_i.detach().cpu().numpy())

        if return_y_trace:
            return y_i, y_i_store
        else:
            return y_i

    def sample(self, x_batch, return_y_trace=False, extract_embedding=False):
        # also use this as a shortcut to avoid doubling batch when guide_w is zero
        is_zero = False
        if -1e-3 < self.guide_w < 1e-3:
            is_zero = True

        # how many noisy actions to begin with
        n_sample = x_batch.shape[0]

        y_shape = (n_sample, self.model_conf.y_dim)

        # sample initial noise, y_0 ~ N(0, 1),
        y_i = torch.randn(y_shape).to(self.device)

        if not is_zero:
            if len(x_batch.shape) > 2:
                # repeat x_batch twice, so can use guided diffusion
                x_batch = x_batch.repeat(2, 1, 1, 1)
            else:
                # repeat x_batch twice, so can use guided diffusion
                x_batch = x_batch.repeat(2, 1)

            # half of context will be zero
            context_mask = torch.zeros(x_batch.shape[0]).to(self.device)
            context_mask[n_sample:] = 1.0  # makes second half of batch context free
        else:
            context_mask = torch.zeros(x_batch.shape[0]).to(self.device)

        if extract_embedding:
            x_embed = self.nn_model.embed_context(x_batch)

        # run denoising chain
        y_i_store = []  # if want to trace how y_i evolved
        for i in range(self.n_t, 0, -1):
            t_is = torch.tensor([i / self.n_t]).to(self.device)
            t_is = t_is.repeat(n_sample, 1)

            if not is_zero:
                # double batch
                y_i = y_i.repeat(2, 1)
                t_is = t_is.repeat(2, 1)

            z = torch.randn(y_shape).to(self.device) if i > 1 else 0

            # split predictions and compute weighting
            if extract_embedding:
                eps = self.nn_model(y_i, x_batch, t_is, context_mask, x_embed)
            else:
                eps = self.nn_model(y_i, x_batch, t_is, context_mask)
            if not is_zero:
                eps1 = eps[:n_sample]
                eps2 = eps[n_sample:]
                eps = (1 + self.guide_w) * eps1 - self.guide_w * eps2
                y_i = y_i[:n_sample]
            y_i = self.one_over_sqrt_a[i] * (y_i - eps * self.mab_over_sqrt_mab[i]) + self.sqrt_bata_t[i] * z
            if return_y_trace and (i % 20 == 0 or i == self.n_t or i < 8):
                y_i_store.append(y_i.detach().cpu().numpy())

        if return_y_trace:
            return y_i, y_i_store
        else:
            return y_i

    def sample_extra(self, x_batch, extra_steps=4, return_y_trace=False):
        # also use this as a shortcut to avoid doubling batch when guide_w is zero
        is_zero = False
        if -1e-3 < self.guide_w < 1e-3:
            is_zero = True

        # how many noisy actions to begin with
        n_sample = x_batch.shape[0]

        # y_shape = (n_sample, self.y_dim)
        y_shape = (n_sample, self.model_conf.y_dim)

        # sample initial noise, y_0 ~ N(0, 1),
        y_i = torch.randn(y_shape).to(self.device)

        if not is_zero:
            if len(x_batch.shape) > 2:
                # repeat x_batch twice, so can use guided diffusion
                x_batch = x_batch.repeat(2, 1, 1, 1)
            else:
                # repeat x_batch twice, so can use guided diffusion
                x_batch = x_batch.repeat(2, 1)
            # half of context will be zero
            context_mask = torch.zeros(x_batch.shape[0]).to(self.device)
            context_mask[n_sample:] = 1.0  # makes second half of batch context free
        else:
            # context_mask = torch.zeros_like(x_batch[:,0]).to(self.device)
            context_mask = torch.zeros(x_batch.shape[0]).to(self.device)

        # run denoising chain
        y_i_store = []  # if want to trace how y_i evolved
        # for i_dummy in range(self.n_T, 0, -1):
        for i_dummy in range(self.n_t, -extra_steps, -1):
            i = max(i_dummy, 1)
            t_is = torch.tensor([i / self.n_t]).to(self.device)
            t_is = t_is.repeat(n_sample, 1)

            if not is_zero:
                # double batch
                y_i = y_i.repeat(2, 1)
                t_is = t_is.repeat(2, 1)

            z = torch.randn(y_shape).to(self.device) if i > 1 else 0

            # split predictions and compute weighting
            eps = self.nn_model(y_i, x_batch, t_is, context_mask)
            if not is_zero:
                eps1 = eps[:n_sample]
                eps2 = eps[n_sample:]
                eps = (1 + self.guide_w) * eps1 - self.guide_w * eps2
                y_i = y_i[:n_sample]
            y_i = self.one_over_sqrt_a[i] * (y_i - eps * self.mab_over_sqrt_mab[i]) + self.sqrt_bata_t[i] * z
            if return_y_trace and (i % 20 == 0 or i == self.n_t or i < 8):
                y_i_store.append(y_i.detach().cpu().numpy())

        if return_y_trace:
            return y_i, y_i_store
        else:
            return y_i

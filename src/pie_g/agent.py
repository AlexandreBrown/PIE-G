# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights
from torchvision.models import resnet18
from torchvision import transforms
import pie_g.utils as utils
from pie_g.utils import random_overlay
from segdac.agents.stochastic_agent import StochasticAgent
from segdac.agents.sampling_strategy import SamplingStrategy
from tensordict import TensorDict


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(
            -1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype
        )[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(
            0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype
        )
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


class Encoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
        )

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h


class ResEncoder(nn.Module):
    def __init__(self, obs_shape, device):
        super(ResEncoder, self).__init__()
        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
        self.transform = transforms.Compose(
            [transforms.Resize(256), transforms.CenterCrop(224)]
        )

        for param in self.model.parameters():
            param.requires_grad = False

        self.num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Identity()
        self.repr_dim = 1024
        self.image_channel = 3
        x = torch.randn([32] + list(obs_shape), device=device)
        with torch.no_grad():
            out_shape = self.forward_conv(x).shape
        self.out_dim = out_shape[1]
        self.fc = nn.Linear(self.out_dim, self.repr_dim)
        self.ln = nn.LayerNorm(self.repr_dim)
        #
        # Initialization
        nn.init.orthogonal_(self.fc.weight.data)
        self.fc.bias.data.fill_(0.0)

    @torch.no_grad()
    def forward_conv(self, obs, flatten=True):
        obs = obs / 255.0 - 0.5
        time_step = obs.shape[1] // self.image_channel
        obs = obs.view(
            obs.shape[0], time_step, self.image_channel, obs.shape[-2], obs.shape[-1]
        )
        obs = obs.view(
            obs.shape[0] * time_step, self.image_channel, obs.shape[-2], obs.shape[-1]
        )

        for name, module in self.model._modules.items():
            obs = module(obs)
            if name == "layer2":
                break

        conv = obs.view(
            obs.size(0) // time_step, time_step, obs.size(1), obs.size(2), obs.size(3)
        )
        if time_step > 1:
            conv_current = conv[:, 1:, :, :, :]
            conv_prev = conv_current - conv[:, : time_step - 1, :, :, :].detach()
            conv = torch.cat([conv_current, conv_prev], axis=1)
            conv = conv.view(
                conv.size(0), conv.size(1) * conv.size(2), conv.size(3), conv.size(4)
            )
        if flatten:
            conv = conv.view(conv.size(0), -1)

        return conv

    def forward(self, obs):
        conv = self.forward_conv(obs)
        out = self.fc(conv)
        out = self.ln(out)
        # obs = self.model(self.transform(obs.to(torch.float32)) / 255.0 - 0.5)
        return out


class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(repr_dim, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh()
        )

        self.policy = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_shape),
        )

        self.apply(utils.weight_init)

    def forward(self, obs, std):
        h = self.trunk(obs)

        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(repr_dim, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh()
        )

        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_shape, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2


class PIEGAgent(StochasticAgent):
    def __init__(
        self,
        obs_shape,
        action_dim,
        device,
        gamma,
        lr,
        feature_dim,
        hidden_dim,
        critic_target_tau,
        update_every_steps,
        stddev_schedule,
        stddev_clip,
        places365_data_dir_path,
        env_action_scaler,
    ):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.env_action_scaler = env_action_scaler
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.discount = gamma
        self.env_step = 0
        self.places365_data_dir_path = places365_data_dir_path

        # models
        encoder = ResEncoder(obs_shape, device=device).to(device)
        actor = Actor(encoder.repr_dim, action_dim, feature_dim, hidden_dim).to(device)
        super().__init__(actor=actor, env_action_scaler=env_action_scaler)

        self.encoder = encoder

        self.critic = Critic(
            self.encoder.repr_dim, action_dim, feature_dim, hidden_dim
        ).to(device)
        self.critic_target = Critic(
            self.encoder.repr_dim, action_dim, feature_dim, hidden_dim
        ).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # data augmentation
        self.aug = RandomShiftsAug(pad=4)

        self.train()
        self.critic_target.train()

    def get_action(self, observation: torch.Tensor) -> torch.Tensor:
        obs = self.encoder(observation)
        stddev = utils.schedule(self.stddev_schedule, self.env_step)
        dist = self.actor(obs, stddev)

        if self.sampling_strategy == SamplingStrategy.RANDOM:
            action = dist.sample(clip=None)
            self.env_step += 1
        else:
            action = dist.mean

        return action

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def update(self, train_data: TensorDict, env_step: int) -> dict:
        self.train()

        obs = train_data["pixels_transformed"]
        action = train_data["action"]
        reward = train_data["next"]["reward"]
        discount = self.discount
        next_obs = train_data["next"]["pixels_transformed"]

        metrics = dict()

        if env_step % self.update_every_steps != 0:
            return metrics

        # augment
        obs = self.aug(obs.float())
        original_obs = obs.clone()
        next_obs = self.aug(next_obs.float())
        # encode
        obs = self.encoder(obs)

        # strong augmentation
        aug_obs = self.encoder(
            random_overlay(
                original_obs, data_dir=self.places365_data_dir_path, device=self.device
            )
        )

        with torch.no_grad():
            next_obs = self.encoder(next_obs)

        # update critic
        metrics.update(
            self.update_critic(
                obs, action, reward, discount, next_obs, env_step, aug_obs
            )
        )

        # update actor
        metrics.update(self.update_actor(obs.detach(), env_step))

        # update critic target
        utils.soft_update_params(
            self.critic, self.critic_target, self.critic_target_tau
        )

        return metrics

    def update_critic(self, obs, action, reward, discount, next_obs, step, aug_obs):
        metrics = dict()
        self.actor.eval()
        self.critic_target.eval()

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        aug_Q1, aug_Q2 = self.critic(aug_obs, action)
        aug_loss = F.mse_loss(aug_Q1, target_Q) + F.mse_loss(aug_Q2, target_Q)

        critic_loss = 0.5 * (critic_loss + aug_loss)

        metrics["critic_target_q"] = target_Q.mean().item()
        metrics["critic_q1"] = Q1.mean().item()
        metrics["critic_q2"] = Q2.mean().item()
        metrics["critic_loss_mean"] = critic_loss.item()

        # optimize encoder and critic
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        return metrics

    def update_actor(self, obs, step):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        metrics["actor_loss"] = actor_loss.item()

        return metrics

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
from torch.distributions import constraints
from torch.distributions.transformed_distribution import TransformedDistribution

_str_to_activation = {
    'relu': nn.ReLU(),
    'elu' : nn.ELU(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'softplus': nn.Softplus(),
    'identity': nn.Identity(),
}


class RSSM(nn.Module):

    def __init__(self, action_size, stoch_size, deter_size,  hidden_size, obs_embed_size, activation):

        super().__init__()

        self.action_size = action_size
        self.stoch_size  = stoch_size   
        self.deter_size  = deter_size   # GRU hidden units
        self.hidden_size = hidden_size  # intermediate fc_layers hidden units 
        self.embedding_size = obs_embed_size

        self.act_fn = _str_to_activation[activation]
        self.rnn = nn.GRUCell(self.deter_size, self.deter_size)

        self.fc_state_action = nn.Linear(self.stoch_size + self.action_size, self.deter_size)
        self.fc_embed_prior = nn.Linear(self.deter_size, self.hidden_size)
        self.fc_state_prior  = nn.Linear(self.hidden_size, 2*self.stoch_size)
        self.fc_embed_posterior = nn.Linear(self.embedding_size + self.deter_size, self.hidden_size)
        self.fc_state_posterior = nn.Linear(self.hidden_size, 2*self.stoch_size)


    def init_state(self, batch_size, device):

        return dict(
            mean = torch.zeros(batch_size, self.stoch_size).to(device),
            std  = torch.zeros(batch_size, self.stoch_size).to(device),
            stoch = torch.zeros(batch_size, self.stoch_size).to(device),
            deter = torch.zeros(batch_size, self.deter_size).to(device))

    def get_dist(self, mean, std):

        distribution = distributions.Normal(mean, std)
        distribution = distributions.independent.Independent(distribution, 1)
        return distribution

    def observe_step(self, prev_state, prev_action, obs_embed, nonterm=1.0):

        prior = self.imagine_step(prev_state, prev_action, nonterm)
        posterior_embed = self.act_fn(self.fc_embed_posterior(torch.cat([obs_embed, prior['deter']], dim=-1)))
        posterior = self.fc_state_posterior(posterior_embed)
        mean, std = torch.chunk(posterior, 2, dim=-1)
        std = F.softplus(std) + 0.1
        sample = mean + torch.randn_like(mean) * std

        posterior = {'mean': mean, 'std': std, 'stoch': sample, 'deter': prior['deter']}
        return prior, posterior

    def imagine_step(self, prev_state, prev_action, nonterm=1.0):

        state_action = self.act_fn(self.fc_state_action(torch.cat([prev_state['stoch']*nonterm, prev_action], dim=-1)))
        deter = self.rnn(state_action, prev_state['deter']*nonterm)
        prior_embed = self.act_fn(self.fc_embed_prior(deter))
        mean, std = torch.chunk(self.fc_state_prior(prior_embed), 2, dim=-1)
        std = F.softplus(std) + 0.1
        sample = mean + torch.randn_like(mean) * std

        prior = {'mean': mean, 'std': std, 'stoch': sample, 'deter': deter}
        return prior

    def observe_rollout(self, obs_embed, actions, nonterms, prev_state, horizon):

        priors = []
        posteriors = []

        for t in range(horizon):
            prev_action = actions[t]* nonterms[t]
            prior_state, posterior_state = self.observe_step(prev_state, prev_action, obs_embed[t], nonterms[t])
            priors.append(prior_state)
            posteriors.append(posterior_state)
            prev_state = posterior_state

        priors = self.stack_states(priors, dim=0)
        posteriors = self.stack_states(posteriors, dim=0)

        return priors, posteriors

    def imagine_rollout(self, actor, prev_state, horizon):

        rssm_state = prev_state
        next_states = []

        for t in range(horizon):
            action = actor(torch.cat([rssm_state['stoch'], rssm_state['deter']], dim=-1).detach())
            rssm_state = self.imagine_step(rssm_state, action)
            next_states.append(rssm_state)

        next_states = self.stack_states(next_states)
        return next_states

    def stack_states(self, states, dim=0):

        return dict(
            mean = torch.stack([state['mean'] for state in states], dim=dim),
            std  = torch.stack([state['std'] for state in states], dim=dim),
            stoch = torch.stack([state['stoch'] for state in states], dim=dim),
            deter = torch.stack([state['deter'] for state in states], dim=dim))

    def detach_state(self, state):

        return dict(
            mean = state['mean'].detach(),
            std  = state['std'].detach(),
            stoch = state['stoch'].detach(),
            deter = state['deter'].detach())

    def seq_to_batch(self, state):

        return dict(
            mean = torch.reshape(state['mean'], (state['mean'].shape[0]* state['mean'].shape[1], *state['mean'].shape[2:])),
            std = torch.reshape(state['std'], (state['std'].shape[0]* state['std'].shape[1], *state['std'].shape[2:])),
            stoch = torch.reshape(state['stoch'], (state['stoch'].shape[0]* state['stoch'].shape[1], *state['stoch'].shape[2:])),
            deter = torch.reshape(state['deter'], (state['deter'].shape[0]* state['deter'].shape[1], *state['deter'].shape[2:])))


class ConvEncoder(nn.Module):
    def __init__(self, input_shape, embed_size, activation, depth=32):
        """
        Beginner-friendly version of ConvEncoder.
        Args:
            input_shape: tuple, e.g., (3, 64, 64)
            embed_size: desired output embedding size
            activation: string key (e.g. 'relu', 'elu')
            depth: base number of channels
        """
        super().__init__()
        self.input_shape = input_shape
        self.embed_size = embed_size
        self.depth = depth
        self.act_fn = _str_to_activation[activation]
        self.kernels = [4,4,4,4]

        # Define layers explicitly (no nn.Sequential)
        self.h1 = nn.Conv2d(input_shape[0], 1 * self.depth, self.kernels[0], stride=2)
        self.h2 = nn.Conv2d(1 * self.depth, 2 * self.depth, self.kernels[1], stride=2)
        self.h3 = nn.Conv2d(2 * self.depth, 4 * self.depth, self.kernels[2], stride=2)
        self.h4 = nn.Conv2d(4 * self.depth, 8 * self.depth, self.kernels[3], stride=2)

        # Fully connected layer
        self.fc = nn.Identity() if self.embed_size == 1024 else nn.Linear(1024, self.embed_size)

    def forward(self, inputs):
        x = inputs.reshape(-1, *self.input_shape) 

        x = self.act_fn(self.h1(x))
        x = self.act_fn(self.h2(x))
        x = self.act_fn(self.h3(x))
        x = self.act_fn(self.h4(x))

        x = torch.reshape(x, (*inputs.shape[:-3], -1))
        x = self.fc(x)

        return x

class ConvDecoder(nn.Module):
    def __init__(self, stoch_size, deter_size, output_shape, activation, depth=32):
        """         
        Args:
            stoch_size: Size of stochastic state vector
            deter_size: Size of deterministic state vector
            output_shape: Shape of the reconstructed image (C, H, W)
            activation: Activation function as a string (e.g., 'relu', 'elu')
            depth: Base number of channels
        """
        super().__init__()
        self.output_shape = output_shape
        self.depth = depth
        self.act_fn = _str_to_activation[activation]
        self.kernels = [5, 5, 6, 6]  

        # Fully connected layer to reshape feature vector into initial conv shape
        self.dense = nn.Linear(stoch_size + deter_size, 32 * depth)

        
        self.h1 = nn.ConvTranspose2d(32 * depth, 4 * depth, self.kernels[0], stride=2)
        self.h2 = nn.ConvTranspose2d(4 * depth, 2 * depth, self.kernels[1], stride=2)
        self.h3 = nn.ConvTranspose2d(2 * depth, 1 * depth, self.kernels[2], stride=2)
        self.h4 = nn.ConvTranspose2d(1 * depth, output_shape[0], self.kernels[3], stride=2)

    def forward(self, features):
        """
        Args:
            features: Latent feature vector of shape [B, stoch_size + deter_size]

        Returns:
            Normal distribution over reconstructed image.
        """
        batch_shape = features.shape[:-1]  # Preserve batch dimensions
        x = self.dense(features)
        x = torch.reshape(x, [-1, 32*self.depth, 1, 1])  # Reshape for transposed convs

        
        x = self.act_fn(self.h1(x))
        x = self.act_fn(self.h2(x))
        x = self.act_fn(self.h3(x))
        mean = self.h4(x)  

        
        mean = mean.view(*batch_shape, *self.output_shape)

        # Return as a normal distribution with fixed std dev
        out_dist = distributions.Independent(distributions.Normal(mean, 1.0), len(self.output_shape))
        return out_dist

# used for reward and value models
class DenseDecoder(nn.Module):

    def __init__(self, stoch_size, deter_size, output_shape, n_layers, units, activation, dist):

        super().__init__()

        self.input_size = stoch_size + deter_size
        self.output_shape = output_shape
        self.n_layers = n_layers
        self.units = units
        self.act_fn = _str_to_activation[activation]
        self.dist = dist

        layers=[]

        for i in range(self.n_layers):
            in_ch = self.input_size if i==0 else self.units
            out_ch = self.units
            layers.append(nn.Linear(in_ch, out_ch))
            layers.append(self.act_fn) 

        layers.append(nn.Linear(self.units, int(np.prod(self.output_shape))))

        self.model = nn.Sequential(*layers)

    def forward(self, features):

        out = self.model(features)

        if self.dist == 'normal':
            return distributions.independent.Independent(
                distributions.Normal(out, 1), len(self.output_shape))
        if self.dist == 'binary':
            return distributions.independent.Independent(
                distributions.Bernoulli(logits =out), len(self.output_shape))
        if self.dist == 'none':
            return out

        raise NotImplementedError(self.dist)

class ActionDecoder(nn.Module):

    def __init__(self, action_size, stoch_size, deter_size, n_layers, units, 
                        activation, min_std=1e-4, init_std=5, mean_scale=5):

        super().__init__()

        self.action_size = action_size
        self.stoch_size = stoch_size
        self.deter_size = deter_size
        self.units = units  
        self.act_fn = _str_to_activation[activation]
        self.n_layers = n_layers

        self._min_std = min_std
        self._init_std = init_std
        self._mean_scale = mean_scale

        layers = []
        for i in range(self.n_layers):
            in_ch = self.stoch_size + self.deter_size if i==0 else self.units
            out_ch = self.units
            layers.append(nn.Linear(in_ch, out_ch))
            layers.append(self.act_fn)

        layers.append(nn.Linear(self.units, 2*self.action_size))
        self.action_model = nn.Sequential(*layers)

    def forward(self, features, deter=False):

        out = self.action_model(features)
        mean, std = torch.chunk(out, 2, dim=-1) 

        raw_init_std = np.log(np.exp(self._init_std)-1)
        action_mean = self._mean_scale * torch.tanh(mean / self._mean_scale)
        action_std = F.softplus(std + raw_init_std) + self._min_std

        dist = distributions.Normal(action_mean, action_std)
        dist = TransformedDistribution(dist, TanhBijector())
        dist = distributions.independent.Independent(dist, 1)
        dist = SampleDist(dist)

        if deter:
            return dist.mode()
        else:
            return dist.rsample()

    def add_exploration(self, action, action_noise=0.3):

        return torch.clamp(distributions.Normal(action, action_noise).rsample(), -1, 1)


class TanhBijector(distributions.Transform):

    def __init__(self):
        super().__init__()
        self.bijective = True
        self.domain = constraints.real
        self.codomain = constraints.interval(-1.0, 1.0)

    @property
    def sign(self): return 1.

    def _call(self, x): return torch.tanh(x)

    def atanh(self, x):
        return 0.5 * torch.log((1 + x) / (1 - x))

    def _inverse(self, y: torch.Tensor):
        y = torch.where(
            (torch.abs(y) <= 1.),
            torch.clamp(y, -0.99999997, 0.99999997),
            y)
        y = self.atanh(y)
        return y

    def log_abs_det_jacobian(self, x, y):
        return 2. * (np.log(2) - x - F.softplus(-2. * x))

class SampleDist:

    def __init__(self, dist, samples=100):
        self._dist = dist
        self._samples = samples

    @property
    def name(self):
        return 'SampleDist'

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def mean(self):
        sample = self._dist.rsample(self._samples)
        return torch.mean(sample, 0)

    def mode(self):
        dist = self._dist.expand((self._samples, *self._dist.batch_shape))
        sample = dist.rsample()
        logprob = dist.log_prob(sample)
        batch_size = sample.size(1)
        feature_size = sample.size(2)
        indices = torch.argmax(logprob, dim=0).reshape(1, batch_size, 1).expand(1, batch_size, feature_size)
        return torch.gather(sample, 0, indices).squeeze(0)

    def entropy(self):
        dist = self._dist.expand((self._samples, *self._dist.batch_shape))
        sample = dist.rsample()
        logprob = dist.log_prob(sample)
        return -torch.mean(logprob, 0)

    def sample(self):
        return self._dist.sample()

###########################################################
# Pretrained encoders for DINOv2, ConvNeXt, EfficientNet
###########################################################
import torch
import torch.nn as nn

# For DINOv2
from transformers import AutoImageProcessor, Dinov2Model

# For EfficientNet, ConvNeXt
import timm

def build_pretrained_encoder(encoder_type="dino", output_dim=1024, freeze=True, block_index=3):
    """
    Returns a nn.Module that extracts features from the selected encoder.
    encoder_type: "dino", "convnext", or "efficientnet"
    output_dim: the final projection size (could be None if you want native size)
    freeze: whether to freeze the encoder weights
    """
    if encoder_type.lower() == "dino":
        print("Using DinoV2 Encoder")
        return DinoV2BaseEncoder(output_dim=output_dim, freeze=freeze)
    elif encoder_type.lower() == "convnext":
        print("Using ConvNeXtV2Tiny Encoder")
        return ConvNeXtV2TinyEncoder(output_dim=output_dim, freeze=freeze)
    elif encoder_type.lower() == "efficientnet":
        print("Using EfficientNetB0 Encoder", freeze)
        return EfficientNetB0Encoder(output_dim=output_dim, freeze=freeze, block_index=block_index)
    else:
        raise ValueError(f"Unknown encoder_type: {encoder_type}")


class DinoV2BaseEncoder(nn.Module):
    """
    Loads a DINOv2-Base model from Hugging Face, without a classification head.
    """
    def __init__(self, output_dim=1024, freeze=True):
        super().__init__()
        # Download the DINOv2-Base checkpoint
        self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
        self.processor.size = {'height': 128, 'width': 128}
        self.encoder = Dinov2Model.from_pretrained("facebook/dinov2-small")
        self.orig_emb_dim = self.encoder.config.hidden_size  # 768 by default for 'base'
        
        # Optional projection to unify dims (e.g. 768 -> 1024)
        if output_dim is not None and output_dim != self.orig_emb_dim:
            self.proj = nn.Linear(self.orig_emb_dim, output_dim)
            self.output_dim = output_dim
        else:
            self.proj = nn.Identity()
            self.output_dim = self.orig_emb_dim
        
        # Freeze or not
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, images):
        """
        images: shape [B, 3, H, W], float in [0..1]
        returns: [B, output_dim] (the final embedding)
        """
        # Hugging Face DINO expects [0..1] -> transformed pixel_values
        # We'll do minimal preprocessing here; see preprocess_for_encoder
        inputs = self.processor(images, return_tensors="pt")['pixel_values'].to(images.device)
        outputs = self.encoder(inputs)
        pooled = outputs.pooler_output  # [B, 768]
        return self.proj(pooled)


class ConvNeXtV2TinyEncoder(nn.Module):
    """
    Loads ConvNeXtV2-Tiny from timm with preprocessing.
    """
    def __init__(self, output_dim=1024, freeze=True):
        super().__init__()
        model_name = "convnextv2_tiny.fcmae_in22k"

        self.encoder = timm.create_model(model_name, pretrained=True, num_classes=0)

        self.data_config = timm.data.resolve_model_data_config(self.encoder)
        self.transforms = timm.data.create_transform(
            **self.data_config, is_training=False, use_prefetcher=False
        )

        self.orig_emb_dim = self.encoder.num_features
        if output_dim is not None and output_dim != self.orig_emb_dim:
            self.proj = nn.Linear(self.orig_emb_dim, output_dim)
            self.output_dim = output_dim
        else:
            self.proj = nn.Identity()
            self.output_dim = self.orig_emb_dim

        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, images):
        """
        images: [B, 3, H, W], float in [0..1]
        """
        images = self.transforms(images)
        feats = self.encoder(images)
        return self.proj(feats)


import timm
import torch
import torch.nn as nn

class EfficientNetB0Encoder(nn.Module):
    """
    Loads EfficientNet-B0 from timm with preprocessing.
    Optionally extracts features from an intermediate block specified by block_index.
    If pool_size is provided, applies AdaptiveAvgPool2d with that output size.
    Otherwise, it directly flattens the feature map.
    Finally, applies a projection to map the flattened features to output_dim.
    Freezing of the encoder is maintained as in your current code.
    """
    def __init__(self, output_dim=1024, freeze=True, block_index=3, pool_size=None):
        """
        Args:
            output_dim (int): Desired output embedding dimension.
            freeze (bool): If True, freeze the encoder parameters.
            block_index (int or None): If provided, extracts features from that block
                                       (e.g., 2 means use features from self.encoder.blocks[2]).
                                       If None, uses the final features (forward_features).
            pool_size (tuple or None): If provided, applies AdaptiveAvgPool2d with this output size.
                                       If None, no additional pooling is applied (features are directly flattened).
        """
        super().__init__()
        model_name = "efficientnet_b0"
        self.encoder = timm.create_model(model_name, pretrained=True, num_classes=0)

        self.data_config = timm.data.resolve_model_data_config(self.encoder)
        target_size = 112  # You can try 96, 112, 128, etc.
        self.data_config['input_size'] = (3, target_size, target_size)
        self.transforms = timm.data.create_transform(
            **self.data_config, is_training=False, use_prefetcher=False
        )
        
        self.block_index = block_index

        # Determine output channels:
        # If block_index is None, we'll use final features (from forward_features)
        # Otherwise, we extract from an earlier block.
        block_channels = {0: 16, 1: 24, 2: 40, 3: 80, 4: 112, 5: 192, 6: 320}
        if self.block_index is None:
            self.out_channels = self.encoder.num_features  # typically 1280 for EfficientNet-B0.
        else:
            self.out_channels = block_channels.get(self.block_index, None)
            if self.out_channels is None:
                raise ValueError("Invalid block_index provided.")

        self.pool_size = pool_size
        if self.pool_size is not None:
            # Use specified pooling; output dimensions will be exactly pool_size.
            self.pool = nn.AdaptiveAvgPool2d(self.pool_size)
            flattened_dim = self.out_channels * self.pool_size[0] * self.pool_size[1]
        else:
            self.pool = None
            # Dynamically compute the flattened dimension by passing a dummy input.
            # Create a dummy tensor with the target input size.
            dummy = torch.zeros(1, 3, target_size, target_size)
            dummy = self.transforms(dummy)
            if self.block_index is None:
                feats_dummy = self.encoder.forward_features(dummy)
            else:
                x = self.encoder.conv_stem(dummy)
                x = self.encoder.bn1(x)
                for idx in range(self.block_index + 1):
                    x = self.encoder.blocks[idx](x)
                feats_dummy = x
            # feats_dummy should have shape [1, C, H, W]
            _, c, h, w = feats_dummy.shape
            flattened_dim = c * h * w

        # Set up the projection layer.
        if output_dim != flattened_dim:
            print(f"[INFO] Projecting from {flattened_dim} to {output_dim}")
            self.proj = nn.Linear(flattened_dim, output_dim)
            self.output_dim = output_dim
        else:
            self.proj = nn.Identity()
            self.output_dim = flattened_dim

        # Freezing the encoder parameters.
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
            # Keep the encoder in train mode so BN layers update their running stats.
            self.encoder.train()
            for m in self.encoder.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    m.track_running_stats = True
                    m.momentum = 0.1

    def forward(self, images):
        """
        Args:
            images: Tensor of shape [B, 3, H, W] with float values in [0, 1].
        Returns:
            Projected embeddings of shape [B, output_dim].
        """
        images = self.transforms(images)

        if self.block_index is None:
            feats = self.encoder.forward_features(images)  # shape: [B, 1280, H', W']
        else:
            x = self.encoder.conv_stem(images)
            x = self.encoder.bn1(x)
            for idx in range(self.block_index + 1):
                x = self.encoder.blocks[idx](x)
            feats = x  # shape: [B, out_channels, H', W']

        if self.pool is not None:
            x = self.pool(feats)
        else:
            x = feats
        flat = x.flatten(1)
        return self.proj(flat)

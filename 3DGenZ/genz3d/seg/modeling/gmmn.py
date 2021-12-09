import torch
from torch import nn


class GMMNnetwork(nn.Module):
    def __init__(
            self,
            noise_dim,
            embed_dim,
            hidden_size,
            feature_dim,
            embed_feature_size=0,
            semantic_reconstruction=False,
    ):
        super().__init__()
        embed_dim = embed_dim + embed_feature_size
        def block(in_feat, out_feat):
            layers = [nn.Linear(in_feat, out_feat)]
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Dropout(p=0.5))
            return layers

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        if hidden_size:
            self.model = nn.Sequential(
                *block(noise_dim + embed_dim, hidden_size),
                nn.Linear(hidden_size, feature_dim),
            )
        else:
            self.model = nn.Linear(noise_dim + embed_dim, feature_dim)

        self.model.apply(init_weights)
        self.semantic_reconstruction = semantic_reconstruction
        if self.semantic_reconstruction:
            self.semantic_reconstruction_layer = nn.Linear(
                feature_dim, noise_dim + embed_dim
            )

    def forward(self, embd, noise):
        features = self.model(torch.cat((embd, noise), 1))
        if self.semantic_reconstruction:
            semantic = self.semantic_reconstruction_layer(features)
            return features, semantic
        else:
            return features

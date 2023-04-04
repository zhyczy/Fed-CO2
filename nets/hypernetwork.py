from collections import OrderedDict

import torch.nn.functional as F
from torch import nn


class Hyper_base(nn.Module):
   
    def __init__(
            self, n_nodes, embedding_dim, in_channels=3, out_dim=10, hidden_dim=100,
            n_hidden=1):
        super().__init__()

        self.client_sample = n_nodes
        self.out_dim = out_dim
        self.embeddings = nn.Embedding(num_embeddings=n_nodes, embedding_dim=embedding_dim)

        layers = [
            nn.Linear(embedding_dim, hidden_dim),
        ]
        for _ in range(n_hidden):
            layers.append(nn.ReLU(inplace=True))
            layers.append(
                nn.Linear(hidden_dim, hidden_dim),
            )

        self.mlp = nn.Sequential(*layers)
        self.l3_weights = nn.Linear(hidden_dim, 10 * 4096)
        self.l3_bias = nn.Linear(hidden_dim, 10)


    def forward(self, idx, test):
        emd = self.embeddings(idx)
        features = self.mlp(emd)
        if test == False:
            weights = [OrderedDict()  for x in range(self.client_sample)]
            head_weight = self.l3_weights(features).view(-1, 10, 4096)
            head_bias = self.l3_bias(features).view(-1, 10)
            for nn in range(self.client_sample):
                weights[nn]["head.weight"] = head_weight[nn]
                weights[nn]["head.bias"] = head_bias[nn]
        else:
            weights = OrderedDict({
                "head.weight": self.l3_weights(features).view(10, 4096),
                "head.bias": self.l3_bias(features).view(-1),
            })
        return weights


class Hyper_bn_P(nn.Module):
   
    def __init__(
            self, n_nodes, embedding_dim, in_channels=3, out_dim=10, hidden_dim=100,
            n_hidden=1):
        super().__init__()

        self.client_sample = n_nodes
        self.out_dim = out_dim
        self.embeddings = nn.Embedding(num_embeddings=n_nodes, embedding_dim=embedding_dim)

        layers = [
            nn.Linear(embedding_dim, hidden_dim),
        ]
        for _ in range(n_hidden):
            layers.append(nn.ReLU(inplace=True))
            layers.append(
                nn.Linear(hidden_dim, hidden_dim),
            )

        self.mlp = nn.Sequential(*layers)

        self.bn1_weights = nn.Linear(hidden_dim, 64)
        self.bn1_bias = nn.Linear(hidden_dim, 64)
        self.bn2_weights = nn.Linear(hidden_dim, 192)
        self.bn2_bias = nn.Linear(hidden_dim, 192)
        self.bn3_weights = nn.Linear(hidden_dim, 384)
        self.bn3_bias = nn.Linear(hidden_dim, 384)
        self.bn4_weights = nn.Linear(hidden_dim, 256)
        self.bn4_bias = nn.Linear(hidden_dim, 256)

        self.bn5_weights = nn.Linear(hidden_dim, 256)
        self.bn5_bias = nn.Linear(hidden_dim, 256)
        self.bn6_weights = nn.Linear(hidden_dim, 4096)
        self.bn6_bias = nn.Linear(hidden_dim, 4096)
        self.bn7_weights = nn.Linear(hidden_dim, 4096)
        self.bn7_bias = nn.Linear(hidden_dim, 4096)
        self.l3_weights = nn.Linear(hidden_dim, 10 * 4096)
        self.l3_bias = nn.Linear(hidden_dim, 10)


    def forward(self, idx, test):
        emd = self.embeddings(idx)
        features = self.mlp(emd)
        if test == False:
            weights = [OrderedDict()  for x in range(self.client_sample)]
            bn1_w = self.bn1_weights(features).view(-1, 64)
            bn1_b = self.bn1_bias(features).view(-1, 64)
            bn2_w = self.bn2_weights(features).view(-1, 192)
            bn2_b = self.bn2_bias(features).view(-1, 192)
            bn3_w = self.bn3_weights(features).view(-1, 384)
            bn3_b = self.bn3_bias(features).view(-1, 384)
            bn4_w = self.bn4_weights(features).view(-1, 256)
            bn4_b = self.bn4_bias(features).view(-1, 256)

            bn5_w = self.bn5_weights(features).view(-1, 256)
            bn5_b = self.bn5_bias(features).view(-1, 256)
            bn6_w = self.bn6_weights(features).view(-1, 4096)
            bn6_b = self.bn6_bias(features).view(-1, 4096)
            bn7_w = self.bn7_weights(features).view(-1, 4096)
            bn7_b = self.bn7_bias(features).view(-1, 4096)
            head_weight = self.l3_weights(features).view(-1, 10, 4096)
            head_bias = self.l3_bias(features).view(-1, 10)
            for nn in range(self.client_sample):
                weights[nn]["features.bn1.weight"] = bn1_w[nn]
                weights[nn]["features.bn1.bias"] = bn1_b[nn]
                weights[nn]["features.bn2.weight"] = bn2_w[nn]
                weights[nn]["features.bn2.bias"] = bn2_b[nn]
                weights[nn]["features.bn3.weight"] = bn3_w[nn]
                weights[nn]["features.bn3.bias"] = bn3_b[nn]
                weights[nn]["features.bn4.weight"] = bn4_w[nn]
                weights[nn]["features.bn4.bias"] = bn4_b[nn]

                weights[nn]["features.bn5.weight"] = bn5_w[nn]
                weights[nn]["features.bn5.bias"] = bn5_b[nn]
                weights[nn]["classifier.bn6.weight"] = bn6_w[nn]
                weights[nn]["classifier.bn6.bias"] = bn6_b[nn]
                weights[nn]["classifier.bn7.weight"] = bn7_w[nn]
                weights[nn]["classifier.bn7.bias"] = bn7_b[nn]
                weights[nn]["head.weight"] = head_weight[nn]
                weights[nn]["head.bias"] = head_bias[nn]
        else:
            weights = OrderedDict({
                "features.bn1.weight": self.bn1_weights(features).view(-1),
                "features.bn1.bias": self.bn1_bias(features).view(-1),
                "features.bn2.weight": self.bn2_weights(features).view(-1),
                "features.bn2.bias": self.bn2_bias(features).view(-1),
                "features.bn3.weight": self.bn3_weights(features).view(-1),
                "features.bn3.bias": self.bn3_bias(features).view(-1),
                "features.bn4.weight": self.bn4_weights(features).view(-1),
                "features.bn4.bias": self.bn4_bias(features).view(-1),

                "features.bn5.weight": self.bn5_weights(features).view(-1),
                "features.bn5.bias": self.bn5_bias(features).view(-1),
                "classifier.bn6.weight": self.bn6_weights(features).view(-1),
                "classifier.bn6.bias": self.bn6_bias(features).view(-1),
                "classifier.bn7.weight": self.bn7_weights(features).view(-1),
                "classifier.bn7.bias": self.bn7_bias(features).view(-1),
                "head.weight": self.l3_weights(features).view(10, 4096),
                "head.bias": self.l3_bias(features).view(-1),
            })
        return weights


class Hyper_bn_pure(nn.Module):
   
    def __init__(
            self, n_nodes, embedding_dim, in_channels=3, out_dim=10, hidden_dim=100,
            n_hidden=1):
        super().__init__()

        self.client_sample = n_nodes
        self.out_dim = out_dim
        self.embeddings = nn.Embedding(num_embeddings=n_nodes, embedding_dim=embedding_dim)

        layers = [
            nn.Linear(embedding_dim, hidden_dim),
        ]
        for _ in range(n_hidden):
            layers.append(nn.ReLU(inplace=True))
            layers.append(
                nn.Linear(hidden_dim, hidden_dim),
            )

        self.mlp = nn.Sequential(*layers)

        self.bn1_weights = nn.Linear(hidden_dim, 64)
        self.bn1_bias = nn.Linear(hidden_dim, 64)
        self.bn2_weights = nn.Linear(hidden_dim, 192)
        self.bn2_bias = nn.Linear(hidden_dim, 192)
        self.bn3_weights = nn.Linear(hidden_dim, 384)
        self.bn3_bias = nn.Linear(hidden_dim, 384)
        self.bn4_weights = nn.Linear(hidden_dim, 256)
        self.bn4_bias = nn.Linear(hidden_dim, 256)

        self.bn5_weights = nn.Linear(hidden_dim, 256)
        self.bn5_bias = nn.Linear(hidden_dim, 256)
        self.bn6_weights = nn.Linear(hidden_dim, 4096)
        self.bn6_bias = nn.Linear(hidden_dim, 4096)
        self.bn7_weights = nn.Linear(hidden_dim, 4096)
        self.bn7_bias = nn.Linear(hidden_dim, 4096)


    def forward(self, idx, test):
        emd = self.embeddings(idx)
        features = self.mlp(emd)
        if test == False:
            weights = [OrderedDict()  for x in range(self.client_sample)]
            bn1_w = self.bn1_weights(features).view(-1, 64)
            bn1_b = self.bn1_bias(features).view(-1, 64)
            bn2_w = self.bn2_weights(features).view(-1, 192)
            bn2_b = self.bn2_bias(features).view(-1, 192)
            bn3_w = self.bn3_weights(features).view(-1, 384)
            bn3_b = self.bn3_bias(features).view(-1, 384)
            bn4_w = self.bn4_weights(features).view(-1, 256)
            bn4_b = self.bn4_bias(features).view(-1, 256)

            bn5_w = self.bn5_weights(features).view(-1, 256)
            bn5_b = self.bn5_bias(features).view(-1, 256)
            bn6_w = self.bn6_weights(features).view(-1, 4096)
            bn6_b = self.bn6_bias(features).view(-1, 4096)
            bn7_w = self.bn7_weights(features).view(-1, 4096)
            bn7_b = self.bn7_bias(features).view(-1, 4096)
            for nn in range(self.client_sample):
                weights[nn]["features.bn1.weight"] = bn1_w[nn]
                weights[nn]["features.bn1.bias"] = bn1_b[nn]
                weights[nn]["features.bn2.weight"] = bn2_w[nn]
                weights[nn]["features.bn2.bias"] = bn2_b[nn]
                weights[nn]["features.bn3.weight"] = bn3_w[nn]
                weights[nn]["features.bn3.bias"] = bn3_b[nn]
                weights[nn]["features.bn4.weight"] = bn4_w[nn]
                weights[nn]["features.bn4.bias"] = bn4_b[nn]

                weights[nn]["features.bn5.weight"] = bn5_w[nn]
                weights[nn]["features.bn5.bias"] = bn5_b[nn]
                weights[nn]["classifier.bn6.weight"] = bn6_w[nn]
                weights[nn]["classifier.bn6.bias"] = bn6_b[nn]
                weights[nn]["classifier.bn7.weight"] = bn7_w[nn]
                weights[nn]["classifier.bn7.bias"] = bn7_b[nn]
        else:
            weights = OrderedDict({
                "features.bn1.weight": self.bn1_weights(features).view(-1),
                "features.bn1.bias": self.bn1_bias(features).view(-1),
                "features.bn2.weight": self.bn2_weights(features).view(-1),
                "features.bn2.bias": self.bn2_bias(features).view(-1),
                "features.bn3.weight": self.bn3_weights(features).view(-1),
                "features.bn3.bias": self.bn3_bias(features).view(-1),
                "features.bn4.weight": self.bn4_weights(features).view(-1),
                "features.bn4.bias": self.bn4_bias(features).view(-1),

                "features.bn5.weight": self.bn5_weights(features).view(-1),
                "features.bn5.bias": self.bn5_bias(features).view(-1),
                "classifier.bn6.weight": self.bn6_weights(features).view(-1),
                "classifier.bn6.bias": self.bn6_bias(features).view(-1),
                "classifier.bn7.weight": self.bn7_weights(features).view(-1),
                "classifier.bn7.bias": self.bn7_bias(features).view(-1),
            })
        return weights


class Hyper_bn_PG(nn.Module):
   
    def __init__(
            self, n_nodes, embedding_dim, in_channels=3, out_dim=10, hidden_dim=100,
            n_hidden=1):
        super().__init__()

        self.client_sample = n_nodes
        self.out_dim = out_dim
        self.embeddings = nn.Embedding(num_embeddings=n_nodes, embedding_dim=embedding_dim)

        layers = [
            nn.Linear(embedding_dim, hidden_dim),
        ]
        for _ in range(n_hidden):
            layers.append(nn.ReLU(inplace=True))
            layers.append(
                nn.Linear(hidden_dim, hidden_dim),
            )

        self.mlp = nn.Sequential(*layers)

        # G branch
        self.bn1_weights_G = nn.Linear(hidden_dim, 64)
        self.bn1_bias_G = nn.Linear(hidden_dim, 64)
        self.bn2_weights_G = nn.Linear(hidden_dim, 192)
        self.bn2_bias_G = nn.Linear(hidden_dim, 192)
        self.bn3_weights_G = nn.Linear(hidden_dim, 384)
        self.bn3_bias_G = nn.Linear(hidden_dim, 384)
        self.bn4_weights_G = nn.Linear(hidden_dim, 256)
        self.bn4_bias_G = nn.Linear(hidden_dim, 256)

        self.bn5_weights_G = nn.Linear(hidden_dim, 256)
        self.bn5_bias_G = nn.Linear(hidden_dim, 256)
        self.bn6_weights_G = nn.Linear(hidden_dim, 4096)
        self.bn6_bias_G = nn.Linear(hidden_dim, 4096)
        self.bn7_weights_G = nn.Linear(hidden_dim, 4096)
        self.bn7_bias_G = nn.Linear(hidden_dim, 4096)

        # P branch
        self.bn1_weights = nn.Linear(hidden_dim, 64)
        self.bn1_bias = nn.Linear(hidden_dim, 64)
        self.bn2_weights = nn.Linear(hidden_dim, 192)
        self.bn2_bias = nn.Linear(hidden_dim, 192)
        self.bn3_weights = nn.Linear(hidden_dim, 384)
        self.bn3_bias = nn.Linear(hidden_dim, 384)
        self.bn4_weights = nn.Linear(hidden_dim, 256)
        self.bn4_bias = nn.Linear(hidden_dim, 256)

        self.bn5_weights = nn.Linear(hidden_dim, 256)
        self.bn5_bias = nn.Linear(hidden_dim, 256)
        self.bn6_weights = nn.Linear(hidden_dim, 4096)
        self.bn6_bias = nn.Linear(hidden_dim, 4096)
        self.bn7_weights = nn.Linear(hidden_dim, 4096)
        self.bn7_bias = nn.Linear(hidden_dim, 4096)
        self.l3_weights = nn.Linear(hidden_dim, 10 * 4096)
        self.l3_bias = nn.Linear(hidden_dim, 10)


    def forward(self, idx, test):
        emd = self.embeddings(idx)
        features = self.mlp(emd)
        if test == False:
            P_weights = [OrderedDict()  for x in range(self.client_sample)]
            G_weights = [OrderedDict()  for x in range(self.client_sample)]
            # G branch
            bn1_w_G = self.bn1_weights_G(features).view(-1, 64)
            bn1_b_G = self.bn1_bias_G(features).view(-1, 64)
            bn2_w_G = self.bn2_weights_G(features).view(-1, 192)
            bn2_b_G = self.bn2_bias_G(features).view(-1, 192)
            bn3_w_G = self.bn3_weights_G(features).view(-1, 384)
            bn3_b_G = self.bn3_bias_G(features).view(-1, 384)
            bn4_w_G = self.bn4_weights_G(features).view(-1, 256)
            bn4_b_G = self.bn4_bias_G(features).view(-1, 256)

            bn5_w_G = self.bn5_weights_G(features).view(-1, 256)
            bn5_b_G = self.bn5_bias_G(features).view(-1, 256)
            bn6_w_G = self.bn6_weights_G(features).view(-1, 4096)
            bn6_b_G = self.bn6_bias_G(features).view(-1, 4096)
            bn7_w_G = self.bn7_weights_G(features).view(-1, 4096)
            bn7_b_G = self.bn7_bias_G(features).view(-1, 4096)

            # P branch
            bn1_w = self.bn1_weights(features).view(-1, 64)
            bn1_b = self.bn1_bias(features).view(-1, 64)
            bn2_w = self.bn2_weights(features).view(-1, 192)
            bn2_b = self.bn2_bias(features).view(-1, 192)
            bn3_w = self.bn3_weights(features).view(-1, 384)
            bn3_b = self.bn3_bias(features).view(-1, 384)
            bn4_w = self.bn4_weights(features).view(-1, 256)
            bn4_b = self.bn4_bias(features).view(-1, 256)

            bn5_w = self.bn5_weights(features).view(-1, 256)
            bn5_b = self.bn5_bias(features).view(-1, 256)
            bn6_w = self.bn6_weights(features).view(-1, 4096)
            bn6_b = self.bn6_bias(features).view(-1, 4096)
            bn7_w = self.bn7_weights(features).view(-1, 4096)
            bn7_b = self.bn7_bias(features).view(-1, 4096)
            head_weight = self.l3_weights(features).view(-1, 10, 4096)
            head_bias = self.l3_bias(features).view(-1, 10)

            for nn in range(self.client_sample):
                P_weights[nn]["features.bn1.weight"] = bn1_w[nn]
                P_weights[nn]["features.bn1.bias"] = bn1_b[nn]
                P_weights[nn]["features.bn2.weight"] = bn2_w[nn]
                P_weights[nn]["features.bn2.bias"] = bn2_b[nn]
                P_weights[nn]["features.bn3.weight"] = bn3_w[nn]
                P_weights[nn]["features.bn3.bias"] = bn3_b[nn]
                P_weights[nn]["features.bn4.weight"] = bn4_w[nn]
                P_weights[nn]["features.bn4.bias"] = bn4_b[nn]

                P_weights[nn]["features.bn5.weight"] = bn5_w[nn]
                P_weights[nn]["features.bn5.bias"] = bn5_b[nn]
                P_weights[nn]["classifier.bn6.weight"] = bn6_w[nn]
                P_weights[nn]["classifier.bn6.bias"] = bn6_b[nn]
                P_weights[nn]["classifier.bn7.weight"] = bn7_w[nn]
                P_weights[nn]["classifier.bn7.bias"] = bn7_b[nn]
                P_weights[nn]["head.weight"] = head_weight[nn]
                P_weights[nn]["head.bias"] = head_bias[nn]

                G_weights[nn]["features.bn1.weight"] = bn1_w_G[nn]
                G_weights[nn]["features.bn1.bias"] = bn1_b_G[nn]
                G_weights[nn]["features.bn2.weight"] = bn2_w_G[nn]
                G_weights[nn]["features.bn2.bias"] = bn2_b_G[nn]
                G_weights[nn]["features.bn3.weight"] = bn3_w_G[nn]
                G_weights[nn]["features.bn3.bias"] = bn3_b_G[nn]
                G_weights[nn]["features.bn4.weight"] = bn4_w_G[nn]
                G_weights[nn]["features.bn4.bias"] = bn4_b_G[nn]

                G_weights[nn]["features.bn5.weight"] = bn5_w_G[nn]
                G_weights[nn]["features.bn5.bias"] = bn5_b_G[nn]
                G_weights[nn]["classifier.bn6.weight"] = bn6_w_G[nn]
                G_weights[nn]["classifier.bn6.bias"] = bn6_b_G[nn]
                G_weights[nn]["classifier.bn7.weight"] = bn7_w_G[nn]
                G_weights[nn]["classifier.bn7.bias"] = bn7_b_G[nn]
        else:
            P_weights = OrderedDict({
                "features.bn1.weight": self.bn1_weights(features).view(-1),
                "features.bn1.bias": self.bn1_bias(features).view(-1),
                "features.bn2.weight": self.bn2_weights(features).view(-1),
                "features.bn2.bias": self.bn2_bias(features).view(-1),
                "features.bn3.weight": self.bn3_weights(features).view(-1),
                "features.bn3.bias": self.bn3_bias(features).view(-1),
                "features.bn4.weight": self.bn4_weights(features).view(-1),
                "features.bn4.bias": self.bn4_bias(features).view(-1),

                "features.bn5.weight": self.bn5_weights(features).view(-1),
                "features.bn5.bias": self.bn5_bias(features).view(-1),
                "classifier.bn6.weight": self.bn6_weights(features).view(-1),
                "classifier.bn6.bias": self.bn6_bias(features).view(-1),
                "classifier.bn7.weight": self.bn7_weights(features).view(-1),
                "classifier.bn7.bias": self.bn7_bias(features).view(-1),
                "head.weight": self.l3_weights(features).view(10, 4096),
                "head.bias": self.l3_bias(features).view(-1),
            })
            G_weights = OrderedDict({
                "features.bn1.weight": self.bn1_weights_G(features).view(-1),
                "features.bn1.bias": self.bn1_bias_G(features).view(-1),
                "features.bn2.weight": self.bn2_weights_G(features).view(-1),
                "features.bn2.bias": self.bn2_bias_G(features).view(-1),
                "features.bn3.weight": self.bn3_weights_G(features).view(-1),
                "features.bn3.bias": self.bn3_bias_G(features).view(-1),
                "features.bn4.weight": self.bn4_weights_G(features).view(-1),
                "features.bn4.bias": self.bn4_bias_G(features).view(-1),

                "features.bn5.weight": self.bn5_weights_G(features).view(-1),
                "features.bn5.bias": self.bn5_bias_G(features).view(-1),
                "classifier.bn6.weight": self.bn6_weights_G(features).view(-1),
                "classifier.bn6.bias": self.bn6_bias_G(features).view(-1),
                "classifier.bn7.weight": self.bn7_weights_G(features).view(-1),
                "classifier.bn7.bias": self.bn7_bias_G(features).view(-1),
            })
        return P_weights, G_weights


class Hyper_bn_pure_PG(nn.Module):
   
    def __init__(
            self, n_nodes, embedding_dim, in_channels=3, out_dim=10, hidden_dim=100,
            n_hidden=1):
        super().__init__()

        self.client_sample = n_nodes
        self.out_dim = out_dim
        self.embeddings = nn.Embedding(num_embeddings=n_nodes, embedding_dim=embedding_dim)

        layers = [
            nn.Linear(embedding_dim, hidden_dim),
        ]
        for _ in range(n_hidden):
            layers.append(nn.ReLU(inplace=True))
            layers.append(
                nn.Linear(hidden_dim, hidden_dim),
            )

        self.mlp = nn.Sequential(*layers)

        # G branch
        self.bn1_weights_G = nn.Linear(hidden_dim, 64)
        self.bn1_bias_G = nn.Linear(hidden_dim, 64)
        self.bn2_weights_G = nn.Linear(hidden_dim, 192)
        self.bn2_bias_G = nn.Linear(hidden_dim, 192)
        self.bn3_weights_G = nn.Linear(hidden_dim, 384)
        self.bn3_bias_G = nn.Linear(hidden_dim, 384)
        self.bn4_weights_G = nn.Linear(hidden_dim, 256)
        self.bn4_bias_G = nn.Linear(hidden_dim, 256)

        self.bn5_weights_G = nn.Linear(hidden_dim, 256)
        self.bn5_bias_G = nn.Linear(hidden_dim, 256)
        self.bn6_weights_G = nn.Linear(hidden_dim, 4096)
        self.bn6_bias_G = nn.Linear(hidden_dim, 4096)
        self.bn7_weights_G = nn.Linear(hidden_dim, 4096)
        self.bn7_bias_G = nn.Linear(hidden_dim, 4096)

        # P branch
        self.bn1_weights = nn.Linear(hidden_dim, 64)
        self.bn1_bias = nn.Linear(hidden_dim, 64)
        self.bn2_weights = nn.Linear(hidden_dim, 192)
        self.bn2_bias = nn.Linear(hidden_dim, 192)
        self.bn3_weights = nn.Linear(hidden_dim, 384)
        self.bn3_bias = nn.Linear(hidden_dim, 384)
        self.bn4_weights = nn.Linear(hidden_dim, 256)
        self.bn4_bias = nn.Linear(hidden_dim, 256)

        self.bn5_weights = nn.Linear(hidden_dim, 256)
        self.bn5_bias = nn.Linear(hidden_dim, 256)
        self.bn6_weights = nn.Linear(hidden_dim, 4096)
        self.bn6_bias = nn.Linear(hidden_dim, 4096)
        self.bn7_weights = nn.Linear(hidden_dim, 4096)
        self.bn7_bias = nn.Linear(hidden_dim, 4096)


    def forward(self, idx, test):
        emd = self.embeddings(idx)
        features = self.mlp(emd)
        if test == False:
            P_weights = [OrderedDict()  for x in range(self.client_sample)]
            G_weights = [OrderedDict()  for x in range(self.client_sample)]
            # G branch
            bn1_w_G = self.bn1_weights_G(features).view(-1, 64)
            bn1_b_G = self.bn1_bias_G(features).view(-1, 64)
            bn2_w_G = self.bn2_weights_G(features).view(-1, 192)
            bn2_b_G = self.bn2_bias_G(features).view(-1, 192)
            bn3_w_G = self.bn3_weights_G(features).view(-1, 384)
            bn3_b_G = self.bn3_bias_G(features).view(-1, 384)
            bn4_w_G = self.bn4_weights_G(features).view(-1, 256)
            bn4_b_G = self.bn4_bias_G(features).view(-1, 256)

            bn5_w_G = self.bn5_weights_G(features).view(-1, 256)
            bn5_b_G = self.bn5_bias_G(features).view(-1, 256)
            bn6_w_G = self.bn6_weights_G(features).view(-1, 4096)
            bn6_b_G = self.bn6_bias_G(features).view(-1, 4096)
            bn7_w_G = self.bn7_weights_G(features).view(-1, 4096)
            bn7_b_G = self.bn7_bias_G(features).view(-1, 4096)

            # P branch
            bn1_w = self.bn1_weights(features).view(-1, 64)
            bn1_b = self.bn1_bias(features).view(-1, 64)
            bn2_w = self.bn2_weights(features).view(-1, 192)
            bn2_b = self.bn2_bias(features).view(-1, 192)
            bn3_w = self.bn3_weights(features).view(-1, 384)
            bn3_b = self.bn3_bias(features).view(-1, 384)
            bn4_w = self.bn4_weights(features).view(-1, 256)
            bn4_b = self.bn4_bias(features).view(-1, 256)

            bn5_w = self.bn5_weights(features).view(-1, 256)
            bn5_b = self.bn5_bias(features).view(-1, 256)
            bn6_w = self.bn6_weights(features).view(-1, 4096)
            bn6_b = self.bn6_bias(features).view(-1, 4096)
            bn7_w = self.bn7_weights(features).view(-1, 4096)
            bn7_b = self.bn7_bias(features).view(-1, 4096)

            for nn in range(self.client_sample):
                P_weights[nn]["features.bn1.weight"] = bn1_w[nn]
                P_weights[nn]["features.bn1.bias"] = bn1_b[nn]
                P_weights[nn]["features.bn2.weight"] = bn2_w[nn]
                P_weights[nn]["features.bn2.bias"] = bn2_b[nn]
                P_weights[nn]["features.bn3.weight"] = bn3_w[nn]
                P_weights[nn]["features.bn3.bias"] = bn3_b[nn]
                P_weights[nn]["features.bn4.weight"] = bn4_w[nn]
                P_weights[nn]["features.bn4.bias"] = bn4_b[nn]

                P_weights[nn]["features.bn5.weight"] = bn5_w[nn]
                P_weights[nn]["features.bn5.bias"] = bn5_b[nn]
                P_weights[nn]["classifier.bn6.weight"] = bn6_w[nn]
                P_weights[nn]["classifier.bn6.bias"] = bn6_b[nn]
                P_weights[nn]["classifier.bn7.weight"] = bn7_w[nn]
                P_weights[nn]["classifier.bn7.bias"] = bn7_b[nn]

                G_weights[nn]["features.bn1.weight"] = bn1_w_G[nn]
                G_weights[nn]["features.bn1.bias"] = bn1_b_G[nn]
                G_weights[nn]["features.bn2.weight"] = bn2_w_G[nn]
                G_weights[nn]["features.bn2.bias"] = bn2_b_G[nn]
                G_weights[nn]["features.bn3.weight"] = bn3_w_G[nn]
                G_weights[nn]["features.bn3.bias"] = bn3_b_G[nn]
                G_weights[nn]["features.bn4.weight"] = bn4_w_G[nn]
                G_weights[nn]["features.bn4.bias"] = bn4_b_G[nn]

                G_weights[nn]["features.bn5.weight"] = bn5_w_G[nn]
                G_weights[nn]["features.bn5.bias"] = bn5_b_G[nn]
                G_weights[nn]["classifier.bn6.weight"] = bn6_w_G[nn]
                G_weights[nn]["classifier.bn6.bias"] = bn6_b_G[nn]
                G_weights[nn]["classifier.bn7.weight"] = bn7_w_G[nn]
                G_weights[nn]["classifier.bn7.bias"] = bn7_b_G[nn]
        else:
            P_weights = OrderedDict({
                "features.bn1.weight": self.bn1_weights(features).view(-1),
                "features.bn1.bias": self.bn1_bias(features).view(-1),
                "features.bn2.weight": self.bn2_weights(features).view(-1),
                "features.bn2.bias": self.bn2_bias(features).view(-1),
                "features.bn3.weight": self.bn3_weights(features).view(-1),
                "features.bn3.bias": self.bn3_bias(features).view(-1),
                "features.bn4.weight": self.bn4_weights(features).view(-1),
                "features.bn4.bias": self.bn4_bias(features).view(-1),

                "features.bn5.weight": self.bn5_weights(features).view(-1),
                "features.bn5.bias": self.bn5_bias(features).view(-1),
                "classifier.bn6.weight": self.bn6_weights(features).view(-1),
                "classifier.bn6.bias": self.bn6_bias(features).view(-1),
                "classifier.bn7.weight": self.bn7_weights(features).view(-1),
                "classifier.bn7.bias": self.bn7_bias(features).view(-1),
            })
            G_weights = OrderedDict({
                "features.bn1.weight": self.bn1_weights_G(features).view(-1),
                "features.bn1.bias": self.bn1_bias_G(features).view(-1),
                "features.bn2.weight": self.bn2_weights_G(features).view(-1),
                "features.bn2.bias": self.bn2_bias_G(features).view(-1),
                "features.bn3.weight": self.bn3_weights_G(features).view(-1),
                "features.bn3.bias": self.bn3_bias_G(features).view(-1),
                "features.bn4.weight": self.bn4_weights_G(features).view(-1),
                "features.bn4.bias": self.bn4_bias_G(features).view(-1),

                "features.bn5.weight": self.bn5_weights_G(features).view(-1),
                "features.bn5.bias": self.bn5_bias_G(features).view(-1),
                "classifier.bn6.weight": self.bn6_weights_G(features).view(-1),
                "classifier.bn6.bias": self.bn6_bias_G(features).view(-1),
                "classifier.bn7.weight": self.bn7_weights_G(features).view(-1),
                "classifier.bn7.bias": self.bn7_bias_G(features).view(-1),
            })
        return P_weights, G_weights


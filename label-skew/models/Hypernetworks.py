from collections import OrderedDict

import torch.nn.functional as F
import torch
from torch import nn
from torch.nn.utils import spectral_norm


class ViTHyper(nn.Module):

    def __init__(self, n_nodes, embedding_dim, hidden_dim, dim, client_sample, heads=8, dim_head=64, n_hidden=1, depth=6,
                 spec_norm=False):
        super(ViTHyper, self).__init__()
        self.dim = dim
        self.inner_dim = dim_head * heads
        self.depth = depth
        self.client_sample = client_sample
        # embedding layer
        self.embeddings = nn.Embedding(num_embeddings=n_nodes, embedding_dim=embedding_dim)

        layers = [
            spectral_norm(nn.Linear(embedding_dim, hidden_dim)) if spec_norm else nn.Linear(embedding_dim, hidden_dim),
        ]
        for _ in range(n_hidden):
            layers.append(nn.ReLU(inplace=True))
            layers.append(
                spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if spec_norm else nn.Linear(hidden_dim, hidden_dim),
            )

        self.mlp = nn.Sequential(*layers)

        self.to_qkv_value_list=nn.ModuleList([])
        for d in range(self.depth):
            to_qkv_value = nn.Linear(hidden_dim, self.dim * self.inner_dim * 3)
            self.to_qkv_value_list.append(to_qkv_value)

    def finetune(self, emd):
        features = self.mlp(emd)  
        weights=OrderedDict()
        for d in range(self.depth):
            layer_d_qkv_value_hyper = self.to_qkv_value_list[d]
            layer_d_qkv_value = layer_d_qkv_value_hyper(features).view(self.inner_dim * 3,self.dim)
            weights["transformer.layers."+str(d)+".0.fn.to_qkv.weight"]=layer_d_qkv_value
        return weights


    def forward(self, idx, test):
        weights = 0
        emd = self.embeddings(idx)
        features = self.mlp(emd)
        if test == False:
            weights = [OrderedDict()  for x in range(self.client_sample)]
            for d in range(self.depth):
                layer_d_qkv_value_hyper = self.to_qkv_value_list[d]
                layer_d_qkv_value = layer_d_qkv_value_hyper(features).view(-1,self.inner_dim * 3,self.dim)
                for nn in range(self.client_sample):
                    weights[nn]["transformer.layers."+str(d)+".0.fn.to_qkv.weight"]=layer_d_qkv_value[nn]
        else:
            weights=OrderedDict()
            for d in range(self.depth):
                layer_d_qkv_value_hyper = self.to_qkv_value_list[d]
                layer_d_qkv_value = layer_d_qkv_value_hyper(features).view(self.inner_dim * 3,self.dim)
                weights["transformer.layers."+str(d)+".0.fn.to_qkv.weight"]=layer_d_qkv_value
        return weights


class Layer_ViTHyper(nn.Module):

    def __init__(self, args, n_nodes, embedding_dim, hidden_dim, dim, client_sample, heads=8, 
        dim_head=64, n_hidden=1, depth=6):
        super(Layer_ViTHyper, self).__init__()
        self.dim = dim
        self.inner_dim = dim_head * heads
        self.depth = depth
        self.client_sample = client_sample
        self.device = args.device
        self.task_embedding_dim = embedding_dim

        # embedding layer
        if args.version == 7:
            print("C+L version1")
            self.layer_id_embeddings = nn.Embedding(n_nodes * depth, self.task_embedding_dim)
            self.embeddings = nn.Embedding(num_embeddings=n_nodes, embedding_dim=embedding_dim)
            self.mode = 1
            layers = [nn.Linear(int(embedding_dim * 2), hidden_dim),]

        elif args.version == 8:
            print("C+L version2")
            self.layer_id_embeddings = nn.Embedding(depth, self.task_embedding_dim)
            self.embeddings = nn.Embedding(num_embeddings=n_nodes, embedding_dim=embedding_dim)
            self.mode = 2 
            layers = [nn.Linear(int(embedding_dim * 2), hidden_dim),]

        elif args.version == 9:
            print("L version3")
            self.layer_id_embeddings = nn.Embedding(n_nodes * depth, self.task_embedding_dim)
            self.mode = 3
            layers = [nn.Linear(int(embedding_dim), hidden_dim),]
        
        for _ in range(n_hidden):
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(hidden_dim, hidden_dim),)

        self.mlp = nn.Sequential(*layers)

        self.to_qkv_value = nn.Linear(hidden_dim, self.dim * self.inner_dim * 3)


    def get_embedding(self, task_embedding, layer_id):
        """Concatenates the task embedding with the embedding for the layer id and
        returns the final joint embedding."""
        
        if self.mode == 3:
            layer_embedding = self.layer_id_embeddings(torch.tensor([layer_id], device = self.device))
            embeddings = layer_embedding.view(1, -1)
        else:
            layer_embedding = self.layer_id_embeddings(torch.tensor([layer_id], device = self.device))
            embeddings = torch.cat([task_embedding.view(1, -1), layer_embedding.view(1, -1)], axis=1)
        return embeddings


    def forward(self, idx, test):
        weights = 0
        if self.mode==3:
            client_emd = torch.zeros([1,100]).to(self.device)
        else:
            client_emd = self.embeddings(idx)
        if test == False:
            weights = [OrderedDict() for x in range(self.client_sample)]
            for d in range(self.depth):
                emd = 0
                for i in range(idx.shape[1]):
                    if self.mode==2:
                        layer_id = d
                    else:
                        layer_id = idx[:, i] * self.depth + d
                    embeddings = self.get_embedding(client_emd[:, i], layer_id)
                    if i == 0:
                        emd = embeddings
                    else:
                        emd = torch.cat([emd, embeddings], axis=0)
                features = self.mlp(emd)
                layer_d_qkv_value = self.to_qkv_value(features).view(-1, self.inner_dim * 3, self.dim)
                for nn in range(self.client_sample):
                    weights[nn]["transformer.layers." + str(d) + ".0.fn.to_qkv.weight"] = layer_d_qkv_value[nn]

        else:
            weights=OrderedDict()
            for d in range(self.depth):
                if self.mode==2:
                    layer_id = d
                else:
                    layer_id = idx[0] * self.depth + d
                emd = self.get_embedding(client_emd, layer_id)
                features = self.mlp(emd)
                layer_d_qkv_value = self.to_qkv_value(features).view(self.inner_dim * 3,self.dim)
                weights["transformer.layers."+str(d)+".0.fn.to_qkv.weight"]=layer_d_qkv_value
        return weights


class ProtoHyper(nn.Module):

    def __init__(self, n_nodes, hidden_dim, dim, client_sample, args, heads=8, dim_head=64, n_hidden=1, depth=6, 
                 spec_norm=False):
        super(ProtoHyper, self).__init__()
        self.dim = dim
        self.inner_dim = dim_head * heads
        self.depth = depth
        self.client_sample = client_sample

        if args.similarity:
            if args.dataset == 'cifar10':
                layers = [
                    spectral_norm(nn.Linear(10*10, hidden_dim)) if spec_norm else nn.Linear(10*10, hidden_dim),
                ]
                self.emb = False
            
            elif args.dataset == 'cifar100':
                layers = [
                    spectral_norm(nn.Linear(100*100, hidden_dim)) if spec_norm else nn.Linear(100*100, hidden_dim),
                ]
                self.emb = False

        else:
            if args.dataset == 'cifar10':
                if args.partition=='noniid-labeluni' and args.position_embedding:
                    layers = [
                        spectral_norm(nn.Linear(2*128, hidden_dim)) if spec_norm else nn.Linear(2*128, hidden_dim),
                    ]
                    self.class_embedding = [nn.Parameter(torch.randn(1, 128)) for x in range(10) ]
                    self.emb = True
                else:
                    layers = [
                        spectral_norm(nn.Linear(10*128, hidden_dim)) if spec_norm else nn.Linear(10*128, hidden_dim),
                    ]
                    self.emb = False
            elif args.dataset == 'cifar100':
                if args.partition=='noniid-labeluni' and args.position_embedding:
                    layers = [
                        spectral_norm(nn.Linear(10*128, hidden_dim)) if spec_norm else nn.Linear(10*128, hidden_dim),
                    ]
                    self.class_embedding = [nn.Parameter(torch.randn(1, 128)) for x in range(100) ]
                    self.emb = True
                else:
                    layers = [
                        spectral_norm(nn.Linear(100*128, hidden_dim)) if spec_norm else nn.Linear(100*128, hidden_dim),
                    ]
                    self.emb = False


        for _ in range(n_hidden):
            layers.append(nn.ReLU(inplace=True))
            layers.append(
                spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if spec_norm else nn.Linear(hidden_dim, hidden_dim),
            )

        self.mlp = nn.Sequential(*layers)

        self.to_qkv_value_list=nn.ModuleList([])
        for d in range(self.depth):
            to_qkv_value = nn.Linear(hidden_dim, self.dim * self.inner_dim * 3)
            self.to_qkv_value_list.append(to_qkv_value)


    def pos_embedding(self, prototype, class_id_list, device):
        for nn in range(len(class_id_list)):
            if nn == 0:
                temp = self.class_embedding[class_id_list[nn]]
            else:
                temp = torch.cat((temp, self.class_embedding[class_id_list[nn]]), 1)
        temp = temp.to(device)
        # print(temp)
        # print(temp.shape)
        # print(prototype)
        # print(prototype.shape)
        return prototype + temp

    def show_embedding(self):
        print("Embedding: ", self.class_embedding[0])

    def forward(self, client_feature, test):
        weights = 0
        features = self.mlp(client_feature)
        if test == False:
            weights = [OrderedDict()  for x in range(self.client_sample)]
            for d in range(self.depth):
                layer_d_qkv_value_hyper = self.to_qkv_value_list[d]
                layer_d_qkv_value = layer_d_qkv_value_hyper(features).view(-1,self.inner_dim * 3,self.dim)
                for nn in range(self.client_sample):
                    weights[nn]["transformer.layers."+str(d)+".0.fn.to_qkv.weight"]=layer_d_qkv_value[nn]
        else:
            weights=OrderedDict()
            for d in range(self.depth):
                layer_d_qkv_value_hyper = self.to_qkv_value_list[d]
                layer_d_qkv_value = layer_d_qkv_value_hyper(features).view(self.inner_dim * 3,self.dim)
                weights["transformer.layers."+str(d)+".0.fn.to_qkv.weight"]=layer_d_qkv_value
        return weights


class ShakesHyper(nn.Module):

    def __init__(self, n_nodes, embedding_dim, hidden_dim, dim, client_sample, heads=8, dim_head=64, n_hidden=1, depth=6,
                 spec_norm=False):
        super(ShakesHyper, self).__init__()
        self.dim = dim
        self.inner_dim = dim_head * heads
        self.depth = depth
        self.client_sample = client_sample
        # embedding layer
        self.embeddings = nn.Embedding(num_embeddings=n_nodes, embedding_dim=embedding_dim)

        layers = [
            spectral_norm(nn.Linear(embedding_dim, hidden_dim)) if spec_norm else nn.Linear(embedding_dim, hidden_dim),
        ]
        for _ in range(n_hidden):
            layers.append(nn.ReLU(inplace=True))
            layers.append(
                spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if spec_norm else nn.Linear(hidden_dim, hidden_dim),
            )

        self.mlp = nn.Sequential(*layers)

        self.wqs_value_list=nn.ModuleList([])
        self.wks_value_list=nn.ModuleList([])
        self.wvs_value_list=nn.ModuleList([])

        for d in range(self.depth):
            wq_value = nn.Linear(hidden_dim, dim * heads * dim_head)
            wk_value = nn.Linear(hidden_dim, dim * heads * dim_head)
            wv_value = nn.Linear(hidden_dim, dim * heads * dim_head)
            self.wqs_value_list.append(wq_value)
            self.wks_value_list.append(wk_value)
            self.wvs_value_list.append(wv_value)


    def finetune(self, emd):
        features = self.mlp(emd)
        weights=OrderedDict()
        for d in range(self.depth):
            layer_d_q_value_hyper = self.wqs_value_list[d]
            layer_d_q_value = layer_d_q_value_hyper(features).view(self.inner_dim ,self.dim)
            layer_d_k_value_hyper = self.wks_value_list[d]
            layer_d_k_value = layer_d_k_value_hyper(features).view(self.inner_dim ,self.dim)
            layer_d_v_value_hyper = self.wvs_value_list[d]
            layer_d_v_value = layer_d_v_value_hyper(features).view(self.inner_dim ,self.dim)
            weights["encoder.layer_stack."+str(d)+".slf_attn.w_qs.weight"]=layer_d_q_value
            weights["encoder.layer_stack."+str(d)+".slf_attn.w_ks.weight"]=layer_d_k_value
            weights["encoder.layer_stack."+str(d)+".slf_attn.w_vs.weight"]=layer_d_v_value
        return weights

    def forward(self, idx, test):
        weights = 0
        emd = self.embeddings(idx)
        features = self.mlp(emd)
        if test == False:
            weights = [OrderedDict()  for x in range(self.client_sample)]
            for d in range(self.depth):
                layer_d_q_value_hyper = self.wqs_value_list[d]
                layer_d_q_value = layer_d_q_value_hyper(features).view(-1, self.inner_dim ,self.dim)
                layer_d_k_value_hyper = self.wks_value_list[d]
                layer_d_k_value = layer_d_k_value_hyper(features).view(-1, self.inner_dim ,self.dim)
                layer_d_v_value_hyper = self.wvs_value_list[d]
                layer_d_v_value = layer_d_v_value_hyper(features).view(-1, self.inner_dim ,self.dim)
                for nn in range(self.client_sample):
                    weights[nn]["encoder.layer_stack."+str(d)+".slf_attn.w_qs.weight"]=layer_d_q_value[nn]
                    weights[nn]["encoder.layer_stack."+str(d)+".slf_attn.w_ks.weight"]=layer_d_k_value[nn]
                    weights[nn]["encoder.layer_stack."+str(d)+".slf_attn.w_vs.weight"]=layer_d_v_value[nn]
        else:
            weights=OrderedDict()
            for d in range(self.depth):
                layer_d_q_value_hyper = self.wqs_value_list[d]
                layer_d_q_value = layer_d_q_value_hyper(features).view(self.inner_dim ,self.dim)
                layer_d_k_value_hyper = self.wks_value_list[d]
                layer_d_k_value = layer_d_k_value_hyper(features).view(self.inner_dim ,self.dim)
                layer_d_v_value_hyper = self.wvs_value_list[d]
                layer_d_v_value = layer_d_v_value_hyper(features).view(self.inner_dim ,self.dim)
                weights["encoder.layer_stack."+str(d)+".slf_attn.w_qs.weight"]=layer_d_q_value
                weights["encoder.layer_stack."+str(d)+".slf_attn.w_ks.weight"]=layer_d_k_value
                weights["encoder.layer_stack."+str(d)+".slf_attn.w_vs.weight"]=layer_d_v_value
        return weights


class Layer_ShakesHyper(nn.Module):

    def __init__(self, n_nodes, embedding_dim, hidden_dim, dim, client_sample, heads=8, 
                dim_head=64, n_hidden=1, depth=6, device="cpu"):
        super(Layer_ShakesHyper, self).__init__()
        self.dim = dim
        self.inner_dim = dim_head * heads
        self.depth = depth
        self.client_sample = client_sample
        self.task_embedding_dim = embedding_dim
        self.device = device

        # embedding layer
        self.embeddings = nn.Embedding(num_embeddings=n_nodes, embedding_dim=embedding_dim)
        self.layer_id_embeddings = nn.Embedding(n_nodes * depth, self.task_embedding_dim)

        layers = [nn.Linear(2*embedding_dim, hidden_dim),]
        for _ in range(n_hidden):
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(hidden_dim, hidden_dim),)

        self.mlp = nn.Sequential(*layers)

        self.wq_value_hyper = nn.Linear(hidden_dim, dim * heads * dim_head)
        self.wk_value_hyper = nn.Linear(hidden_dim, dim * heads * dim_head)
        self.wv_value_hyper = nn.Linear(hidden_dim, dim * heads * dim_head)
    

    def forward(self, idx, test):
        weights = 0
        client_embedding = self.embeddings(idx)
        if test == False:
            weights = [OrderedDict()  for x in range(self.client_sample)]
            for d in range(self.depth):
                emd = 0
                for i in range(idx.shape[1]):
                    layer_id = idx[:, i] * self.depth + d
                    task_embedding = client_embedding[:, i]
                    layer_embedding = self.layer_id_embeddings(torch.tensor([layer_id]).to(self.device))
                    embeds = torch.cat([task_embedding.view(1, -1), layer_embedding.view(1, -1)], axis=1)
                    if i == 0:
                        emd = embeds
                    else:
                        emd = torch.cat([emd, embeds], axis=0)
                features = self.mlp(emd)
                layer_d_q_value = self.wq_value_hyper(features).view(-1, self.inner_dim ,self.dim)
                layer_d_k_value = self.wk_value_hyper(features).view(-1, self.inner_dim ,self.dim)
                layer_d_v_value = self.wv_value_hyper(features).view(-1, self.inner_dim ,self.dim)
                for nn in range(self.client_sample):
                    weights[nn]["encoder.layer_stack."+str(d)+".slf_attn.w_qs.weight"]=layer_d_q_value[nn]
                    weights[nn]["encoder.layer_stack."+str(d)+".slf_attn.w_ks.weight"]=layer_d_k_value[nn]
                    weights[nn]["encoder.layer_stack."+str(d)+".slf_attn.w_vs.weight"]=layer_d_v_value[nn]
        else:
            weights=OrderedDict()
            for d in range(self.depth):
                layer_id = idx[0] * self.depth + d
                layer_embedding = self.layer_id_embeddings(torch.tensor([layer_id]).to(self.device))
                emd = torch.cat([client_embedding.view(1, -1), layer_embedding.view(1, -1)], axis=1)
                features = self.mlp(emd)
                layer_d_q_value = self.wq_value_hyper(features).view(self.inner_dim ,self.dim)
                layer_d_k_value = self.wk_value_hyper(features).view(self.inner_dim ,self.dim)
                layer_d_v_value = self.wv_value_hyper(features).view(self.inner_dim ,self.dim)
                weights["encoder.layer_stack."+str(d)+".slf_attn.w_qs.weight"]=layer_d_q_value
                weights["encoder.layer_stack."+str(d)+".slf_attn.w_ks.weight"]=layer_d_k_value
                weights["encoder.layer_stack."+str(d)+".slf_attn.w_vs.weight"]=layer_d_v_value
        return weights


class Relation_net(nn.Module):
    def __init__(self, embedding_hash):
        super(Relation_net, self).__init__()
        self.dim = embedding_hash.size(1)
        self.client_num = embedding_hash.size(0)
        self.embedding_hash = embedding_hash
        self.lin1 = nn.Linear(self.dim*2, 32)
        self.lin2 = nn.Linear(32, 8)
        self.lin3 = nn.Linear(8, 1)
        self.relu = nn.ReLU()

    def forward(self, self_embedding):
        # print(self_embedding.shape)
        emb = self_embedding.unsqueeze(0).expand(self.client_num, self.dim)
        # print(emb.shape)
        # print(emb)
        emb = torch.cat([self.embedding_hash, emb], 1)
        # print(emb.shape)
        # print(emb)
        l1 = self.relu(self.lin1(emb))
        l2 = self.relu(self.lin2(l1))
        weights = self.relu(self.lin3(l2))
        # print(weights)
        row_sum = weights.sum()
        weights = torch.div(weights, row_sum)
        # print(weights.shape)
        # print(weights)
        return weights
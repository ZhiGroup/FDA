import torch
import torch.nn as nn
import torch_geometric
from kdbnet.gvp import GVP, GVPConvLayer, LayerNorm

class Prot3DGraphModel(nn.Module):
    def __init__(self,
        d_vocab=21, d_embed=20,
        d_dihedrals=6, d_pretrained_emb=1280, d_edge=39,
        d_gcn=[128, 256, 256],
    ):
        super(Prot3DGraphModel, self).__init__()
        d_gcn_in = d_gcn[0]
        self.embed = nn.Embedding(d_vocab, d_embed)
        self.proj_node = nn.Linear(d_embed + d_dihedrals + d_pretrained_emb, d_gcn_in)
        self.proj_edge = nn.Linear(d_edge, d_gcn_in)
        gcn_layer_sizes = [d_gcn_in] + d_gcn
        layers = []
        for i in range(len(gcn_layer_sizes) - 1):            
            layers.append((
                torch_geometric.nn.TransformerConv(
                    gcn_layer_sizes[i], gcn_layer_sizes[i + 1], edge_dim=d_gcn_in),
                'x, edge_index, edge_attr -> x'
            ))            
            layers.append(nn.LeakyReLU())            
        
        self.gcn = torch_geometric.nn.Sequential(
            'x, edge_index, edge_attr', layers)        
        self.pool = torch_geometric.nn.global_mean_pool
        

    def forward(self, data):
        x, edge_index = data.seq, data.edge_index
        batch = data.batch

        x = self.embed(x)
        s = data.node_s
        emb = data.seq_emb
        x = torch.cat([x, s, emb], dim=-1)

        edge_attr = data.edge_s

        x = self.proj_node(x)
        edge_attr = self.proj_edge(edge_attr)

        x = self.gcn(x, edge_index, edge_attr)
        x = torch_geometric.nn.global_mean_pool(x, batch)
        return x



class DrugGVPModel(nn.Module):
    def __init__(self, 
        node_in_dim=[66, 1], node_h_dim=[128, 64],
        edge_in_dim=[16, 1], edge_h_dim=[32, 1],
        num_layers=3, drop_rate=0.1
    ):
        """
        Parameters
        ----------
        node_in_dim : list of int
            Input dimension of drug node features (si, vi).
            Scalar node feartures have shape (N, si).
            Vector node features have shape (N, vi, 3).
        node_h_dims : list of int
            Hidden dimension of drug node features (so, vo).
            Scalar node feartures have shape (N, so).
            Vector node features have shape (N, vo, 3).
        """
        super(DrugGVPModel, self).__init__()
        self.W_v = nn.Sequential(
            LayerNorm(node_in_dim),
            GVP(node_in_dim, node_h_dim, activations=(None, None))
        )
        self.W_e = nn.Sequential(
            LayerNorm(edge_in_dim),
            GVP(edge_in_dim, edge_h_dim, activations=(None, None))
        )

        self.layers = nn.ModuleList(
                GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate)
            for _ in range(num_layers))

        ns, _ = node_h_dim
        self.W_out = nn.Sequential(
            LayerNorm(node_h_dim),
            GVP(node_h_dim, (ns, 0)))

    def forward(self, xd):
        # Unpack input data
        h_V = (xd.node_s, xd.node_v)
        h_E = (xd.edge_s, xd.edge_v)
        edge_index = xd.edge_index
        batch = xd.batch

        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)
        for layer in self.layers:
            h_V = layer(h_V, edge_index, h_E)
        out = self.W_out(h_V)

        # per-graph mean
        out = torch_geometric.nn.global_add_pool(out, batch)

        return out


class DTAModel(nn.Module):
    def __init__(self,
            prot_emb_dim=1280,
            prot_gcn_dims=[128, 256, 256],
            prot_fc_dims=[1024, 128],
            drug_node_in_dim=[66, 1], drug_node_h_dims=[128, 64],
            drug_edge_in_dim=[16, 1], drug_edge_h_dims=[32, 1],            
            drug_fc_dims=[1024, 128],
            mlp_dims=[1024, 512], mlp_dropout=0.25):
        super(DTAModel, self).__init__()

        self.drug_model = DrugGVPModel(
            node_in_dim=drug_node_in_dim, node_h_dim=drug_node_h_dims,
            edge_in_dim=drug_edge_in_dim, edge_h_dim=drug_edge_h_dims,
        )
        drug_emb_dim = drug_node_h_dims[0]

        self.prot_model = Prot3DGraphModel(
            d_pretrained_emb=prot_emb_dim, d_gcn=prot_gcn_dims
        )
        prot_emb_dim = prot_gcn_dims[-1]

        self.drug_fc = self.get_fc_layers(
            [drug_emb_dim] + drug_fc_dims,
            dropout=mlp_dropout, batchnorm=False,
            no_last_dropout=True, no_last_activation=True)
       
        self.prot_fc = self.get_fc_layers(
            [prot_emb_dim] + prot_fc_dims,
            dropout=mlp_dropout, batchnorm=False,
            no_last_dropout=True, no_last_activation=True)

        self.top_fc = self.get_fc_layers(
            [drug_fc_dims[-1] + prot_fc_dims[-1]] + mlp_dims + [1],
            dropout=mlp_dropout, batchnorm=False,
            no_last_dropout=True, no_last_activation=True)

    def get_fc_layers(self, hidden_sizes,
            dropout=0, batchnorm=False,
            no_last_dropout=True, no_last_activation=True):
        act_fn = torch.nn.LeakyReLU()
        layers = []
        for i, (in_dim, out_dim) in enumerate(zip(hidden_sizes[:-1], hidden_sizes[1:])):
            layers.append(nn.Linear(in_dim, out_dim))
            if not no_last_activation or i != len(hidden_sizes) - 2:
                layers.append(act_fn)
            if dropout > 0:
                if not no_last_dropout or i != len(hidden_sizes) - 2:
                    layers.append(nn.Dropout(dropout))
            if batchnorm and i != len(hidden_sizes) - 2:
                layers.append(nn.BatchNorm1d(out_dim))
        return nn.Sequential(*layers)

    def forward(self, xd, xp):
        xd = self.drug_model(xd)
        xp = self.prot_model(xp)

        xd = self.drug_fc(xd)
        xp = self.prot_fc(xp)

        x = torch.cat([xd, xp], dim=1)
        x = self.top_fc(x)
        return x

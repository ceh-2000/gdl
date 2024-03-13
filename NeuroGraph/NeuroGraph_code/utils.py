import torch
from torch.nn import Linear
from torch import nn
from torch_geometric.nn import global_max_pool
from torch_geometric.nn import aggr
import torch.nn.functional as F
from torch_geometric.nn import APPNP, MLP, GCNConv, GINConv, SAGEConv, GraphConv, TransformerConv, ChebConv, GATConv, SGConv, GeneralConv
from torch.nn import Conv1d, MaxPool1d, ModuleList
import random
import math
from torch.nn import BatchNorm1d

softmax = torch.nn.LogSoftmax(dim=1)
        # self.attention = Attention(input_dim1, hidden_channels)


### BASELINE MLP FOR COMPARISON
class BasicMLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim *4),
            nn.BatchNorm1d(hidden_dim *4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim * 10),
            nn.BatchNorm1d(hidden_dim * 10),
            nn.ReLU(),
            nn.Linear(hidden_dim * 10, hidden_dim * 10),
            nn.BatchNorm1d(hidden_dim * 10),
            nn.ReLU(),
            nn.Linear(hidden_dim * 10, hidden_dim * 10),
            nn.BatchNorm1d(hidden_dim * 10),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim * 10, hidden_dim * 10),
            nn.BatchNorm1d(hidden_dim * 10),
            nn.ReLU(),
            nn.Linear(hidden_dim * 10, hidden_dim * 10),
            nn.BatchNorm1d(hidden_dim * 10),
            nn.ReLU(),
            nn.Linear(hidden_dim * 10, hidden_dim * 10),
            nn.BatchNorm1d(hidden_dim * 10),
            nn.ReLU(),
            nn.Linear(hidden_dim * 10, hidden_dim * 10),
            nn.BatchNorm1d(hidden_dim * 10),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim * 10, hidden_dim * 6),
            nn.BatchNorm1d(hidden_dim * 6),
            nn.ReLU(),
            nn.Linear(hidden_dim * 6, hidden_dim * 4),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim//2, hidden_dim//2),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(),
            nn.Linear((hidden_dim//2), output_dim),
        )

    def forward(self, data):
        x, batch = data.x, data.batch
        x = self.mlp(x)
        x = global_add_pool(x, batch)
        return x


### ORIGINAL NEUROGRAPH MODEL
class ResidualGNNs(torch.nn.Module):
    def __init__(self,args, train_dataset, hidden_channels,hidden, num_layers, GNN, k=0.6):
        super().__init__()
        self.convs = ModuleList()
        self.aggr = aggr.MeanAggregation()
        self.hidden_channels = hidden_channels
        num_features = train_dataset.num_features
        if args.model=="ChebConv":
            if num_layers>0:
                self.convs.append(GNN(num_features, hidden_channels,K=5))
                for i in range(0, num_layers - 1):
                    self.convs.append(GNN(hidden_channels, hidden_channels,K=5))
        else:
            if num_layers>0:
                self.convs.append(GNN(num_features, hidden_channels))
                for i in range(0, num_layers - 1):
                    self.convs.append(GNN(hidden_channels, hidden_channels))
        
        input_dim1 = int(((num_features * num_features)/2)- (num_features/2)+(hidden_channels*num_layers))
        input_dim = int(((num_features * num_features)/2)- (num_features/2))
        self.bn = nn.BatchNorm1d(input_dim)
        self.bnh = nn.BatchNorm1d(hidden_channels*num_layers)

        self.mlp = nn.Sequential(
            nn.Linear(input_dim1, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden, hidden//2),
            nn.BatchNorm1d(hidden//2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden//2, hidden//2),
            nn.BatchNorm1d(hidden//2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear((hidden//2), args.num_classes),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # for conv in self.convs:
        #     x = conv(x, edge_index).relu()
        
        xs = [x]        
        for conv in self.convs:
            xs += [conv(xs[-1], edge_index).tanh()]
        h = []
        for i, xx in enumerate(xs):
            if i== 0:
                xx = xx.reshape(data.num_graphs, x.shape[1],-1)
                x = torch.stack([t.triu().flatten()[t.triu().flatten().nonzero(as_tuple=True)] for t in xx])
                x = self.bn(x)
            else:
                # xx = xx.reshape(data.num_graphs, x.shape[1],-1)
                xx = self.aggr(xx,batch)
                # h.append(torch.stack([t.flatten() for t in xx]))
                h.append(xx)
        
        h = torch.cat(h,dim=1)
        h = self.bnh(h)
        # x = torch.stack(h, dim=0)
        x = torch.cat((x,h),dim=1)
        x = self.mlp(x)
        return x








#### Including ATTENTION Between GNNs (between channels)
import torch
import torch.nn as nn
from torch.nn import ModuleList
from torch_geometric.nn import ChebConv  # Assuming you're using PyTorch Geometric for GNNs


class Attention1(nn.Module):
    def __init__(self, in_channels, out_channels, alpha):
        super(Attention1, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.softmax = nn.Softmax(dim=-1)
        self.alpha = alpha

    def forward(self, x):
        if self.training:
            if torch.rand(1).item() < self.alpha:
                attn_weights = self.softmax(self.linear(x))
                output = x * attn_weights
            else:
                output = x
        else:
            if self.alpha > 0.5:
                attn_weights = self.softmax(self.linear(x))
                output = x * attn_weights
            else:
                output = x
        return output

class ResidualAttentionalGNN1(torch.nn.Module):
    '''
    The Attention class applies attention at the node level. It computes attention weights based 
    on the node features and applies them element-wise to the node features.
    During the forward pass in ResidualAttentionalGNN1, the attention mechanism is applied after
    each graph convolution operation (conv(x, edge_index)). It applies attention to node features
    independently within each layer of the GNN.
    '''
    def __init__(self, args, train_dataset,hidden_channels,hidden,num_layers,GNN,alpha=0.5, k=0.6):
        super().__init__()
        self.convs = ModuleList()
        self.attention_layers = ModuleList()
        self.aggr = aggr.MeanAggregation()
        self.hidden_channels = hidden_channels
        self.alpha = alpha
        num_features = train_dataset.num_features

        for i in range(num_layers):
            if i == 0:
                # Assuming the first layer has the largest size
                self.convs.append(GNN(num_features, hidden_channels))
                self.attention_layers.append(Attention1(hidden_channels, hidden_channels, self.alpha))
                input_dim = int(((num_features * num_features) / 2) - (num_features / 2) + (hidden_channels * num_layers))
                self.bn = nn.BatchNorm1d(input_dim)
                torch.nn.init.constant_(self.bn.running_mean, 0)  # Initialize running_mean
            else:
                self.convs.append(GNN(hidden_channels, hidden_channels))
                self.attention_layers.append(Attention1(hidden_channels, hidden_channels, self.alpha))
                input_dim1 = int(((num_features * num_features) / 2) - (num_features / 2))

        print(num_features)
        print(num_layers)

        self.bn = nn.BatchNorm1d(998504)  # Set correct input dimension
        torch.nn.init.constant_(self.bn.running_mean, 0)  # Initialize running_mean
        self.bnh = nn.BatchNorm1d(hidden_channels*num_layers)

        self.mlp = nn.Sequential(
            nn.Linear(in_features=998600, out_features=64, bias=True),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden, hidden//2),
            nn.BatchNorm1d(hidden//2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden//2, hidden//2),
            nn.BatchNorm1d(hidden//2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear((hidden//2), args.num_classes),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        xs = [x]
        for conv, att in zip(self.convs, self.attention_layers):
            x = conv(x, edge_index).tanh()
            x = att(x)  # Applying attention
            xs.append(x)
        
        h = []
        for i, xx in enumerate(xs):
            if i == 0:
                xx = xx.reshape(data.num_graphs, x.shape[1], -1)
                x = torch.stack([t.triu().flatten()[t.triu().flatten().nonzero(as_tuple=True)] for t in xx])
                x = self.bn(x)
            else:
                xx = self.aggr(xx, batch)
                h.append(xx)
        
        h = torch.cat(h, dim=1)
        h = self.bnh(h)
        x = torch.cat((x, h), dim=1)
        x = self.mlp(x)
        return x



#### ATTENTION BETWEEN EDGES 
class Attention2(nn.Module):
    def __init__(self, in_channels, out_channels, alpha=0.5):
        super(Attention2, self).__init__()
        self.linear = nn.Linear(in_channels * 2, out_channels)
        self.alpha = alpha
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, edge_index):
        row, col = edge_index
        edge_features = torch.cat([x[row], x[col]], dim=1)  # Concatenate node features
        attn_weights = self.softmax(self.linear(edge_features))
        output = x * (1 - self.alpha) + attn_weights * self.alpha * x[col].clone().detach()
        return output

class Attention2(nn.Module):
    def __init__(self, in_channels, out_channels, alpha=0.5):
        super(Attention2, self).__init__()
        self.linear = nn.Linear(in_channels * 2, out_channels)
        self.alpha = alpha
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, edge_index):
        row, col = edge_index
        edge_features = torch.cat([x[row], x[col]], dim=1)  # Concatenate node features
        attn_weights = self.softmax(self.linear(edge_features))

        # Reshape x[col] to match attn_weights in the first dimension
        x_col_reshaped = x[col].view(-1, 1, x.shape[1])  # (num_edges, 1, feature_dim)

        output = x * (1 - self.alpha) + attn_weights * self.alpha * x_col_reshaped.clone().detach()
        return output

class ResidualAtentionalGNN2(torch.nn.Module):
    '''
    The Attention class applies attention at the edge level. It concatenates the features of the 
    nodes connected by each edge, then computes attention weights based on this concatenated edge 
    feature representation.
    During the forward pass in ResidualAtentionalGNN2, the attention mechanism is applied within
    the loop over layers (for conv, att in zip(self.convs, self.attention_layers)). Here, 
    attention is applied between node pairs connected by edges in the graph.
    '''
    def __init__(self, args, train_dataset, hidden_channels, hidden, num_layers, GNN, k=0.6, alpha=0.5):
        super().__init__()
        self.convs = ModuleList()
        self.attention_layers = ModuleList()
        self.aggr = MeanAggregation()
        self.hidden_channels = hidden_channels
        self.alpha=alpha
        num_features = train_dataset.num_features

        
        for i in range(num_layers):
            if i == 0:
                self.convs.append(GNN(num_features, hidden_channels))
                self.attention_layers.append(Attention2(hidden_channels, hidden_channels, self.alpha))
            else:
                self.convs.append(GNN(hidden_channels, hidden_channels))
                self.attention_layers.append(Attention2(hidden_channels, hidden_channels, self.alpha))
        

        input_dim1 = int(((num_features * num_features)/2) - (num_features/2) + (hidden_channels*num_layers))
        input_dim = int(((num_features * num_features)/2) - (num_features/2))
        self.bn = nn.BatchNorm1d(input_dim)
        self.bnh = nn.BatchNorm1d(hidden_channels*num_layers)

        self.mlp = nn.Sequential(
            nn.Linear(input_dim1, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden, hidden//2),
            nn.BatchNorm1d(hidden//2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden//2, hidden//2),
            nn.BatchNorm1d(hidden//2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear((hidden//2), args.num_classes),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        xs = [x]
        for conv, att in zip(self.convs, self.attention_layers):
            x = conv(x, edge_index).tanh()
            x = att(x, edge_index)  # Applying attention between nodes
            xs.append(x)
        
        h = []
        for i, xx in enumerate(xs):
            if i == 0:
                xx = xx.reshape(data.num_graphs, x.shape[1], -1)
                x = torch.stack([t.triu().flatten()[t.triu().flatten().nonzero(as_tuple=True)] for t in xx])
                x = self.bn(x)
            else:
                xx = self.aggr(xx, batch)
                h.append(xx)
        
        h = torch.cat(h, dim=1)
        h = self.bnh(h)
        x = torch.cat((x, h), dim=1)
        x = self.mlp(x)
        return x


####ANOTHER ONE GRAPH LEVEL
import torch
import torch.nn as nn

class Attention4(nn.Module):
    def __init__(self, in_channels, out_channels, alpha):
        super(Attention4, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.softmax = nn.Softmax(dim=-1)
        self.alpha = alpha

    def forward(self, x):
        if self.training:
            if torch.rand(1).item() < self.alpha:
                attn_weights = self.softmax(self.linear(x))
                output = x * attn_weights
            else:
                output = x
        else:
            if self.alpha > 0.5:
                attn_weights = self.softmax(self.linear(x))
                output = x * attn_weights
            else:
                output = x
        return output


class GraphLevelAttention(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphLevelAttention, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.attention = nn.Linear(out_features, 1)

    def forward(self, x, batch):
        x = F.relu(self.linear(x))
        attention_weights = F.softmax(self.attention(x), dim=0)
        x = torch.sum(x * attention_weights, dim=0)
        return x

class ResidualAttentionalGNN4(torch.nn.Module):
    def __init__(self, args, train_dataset, hidden_channels, hidden, num_layers, GNN, alpha=0.5, k=0.6):
        super().__init__()
        self.convs = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.graph_level_attention = GraphLevelAttention(hidden_channels*num_layers, hidden_channels)
        self.aggr = MeanAggregation()
        self.hidden_channels = hidden_channels
        self.alpha = alpha
        num_features = train_dataset.num_features

        for i in range(num_layers):
            if i == 0:
                self.convs.append(GNN(num_features, hidden_channels))
                self.attention_layers.append(Attention4(hidden_channels, hidden_channels, self.alpha))
            else:
                self.convs.append(GNN(hidden_channels, hidden_channels))
                self.attention_layers.append(Attention4(hidden_channels, hidden_channels, self.alpha))
        

        input_dim1 = int(((num_features * num_features)/2) - (num_features/2) + (hidden_channels*num_layers))
        input_dim = int(((num_features * num_features)/2) - (num_features/2))
        self.bn = nn.BatchNorm1d(input_dim) #input dim
        self.bnh = nn.BatchNorm1d(hidden_channels*num_layers)

        self.mlp = nn.Sequential(
            nn.Linear(input_dim1, hidden), #input_dim1
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden, hidden//2),
            nn.BatchNorm1d(hidden//2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden//2, hidden//2),
            nn.BatchNorm1d(hidden//2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear((hidden//2), args.num_classes),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        xs = [x]
        for conv, att in zip(self.convs, self.attention_layers):
            x = conv(x, edge_index).tanh()
            x = att(x)  # Applying attention
            xs.append(x)
        
        h = []
        for i, xx in enumerate(xs):
            if i == 0:
                xx = xx.reshape(data.num_graphs, x.shape[1], -1)
                x = torch.stack([t.triu().flatten()[t.triu().flatten().nonzero(as_tuple=True)] for t in xx])
                x = self.bn(x)
            else:
                xx = self.aggr(xx, batch)
                h.append(xx)
        
        h = torch.cat(h, dim=1)
        h = self.graph_level_attention(h, batch)  # Applying graph-level attention
        h = self.bnh(h)
        x = torch.cat((x, h), dim=1)
        x = self.mlp(x)
        return x





#### NEW IMPLEMENTATION WITH ATTENTION
###

#import torch
#import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_mean_pool
#from torch_geometric.nn import BatchNorm1d

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, MeanAggregation

class GNNWithAttention(MessagePassing):
    def __init__(self, gnn_layer, alpha=0.5, heads=1, concat=False, **kwargs):
        super(GNNWithAttention, self).__init__(aggr='mean', **kwargs)
        self.gnn_layer = gnn_layer
        self.alpha = alpha
        self.heads = heads
        self.concat = concat

        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * gnn_layer.out_channels))
        nn.init.xavier_uniform_(self.att)

    def forward(self, x, edge_index):
        x = self.gnn_layer(x, edge_index)

        if torch.rand(1).item() < self.alpha:
            x = x.view(-1, self.heads, self.gnn_layer.out_channels)
            alpha = (torch.cat([x, x], dim=-1) * self.att).sum(dim=-1)
            alpha = torch.softmax(alpha, dim=1)
            x = (x * alpha.view(-1, self.heads, 1)).sum(dim=1)

        return x

class ResidualAttentionalGNN3(torch.nn.Module):
    def __init__(self, args, train_dataset, hidden_channels, hidden, num_layers, gnn_layer, alpha=0.5, k=0.6):
        super().__init__()
        
        #self.convs = nn.ModuleList([GNNWithAttention(gnn_layer(num_features, hidden_channels), alpha=alpha) for _ in range(num_layers)])
        
        self.aggr = MeanAggregation()
        self.hidden_channels = hidden_channels
        self.alpha = alpha
        num_features = train_dataset.num_features
        self.convs = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                self.convs.append(GNNWithAttention(gnn_layer(num_features, hidden_channels), alpha=alpha))
            else:
                self.convs.append(GNNWithAttention(gnn_layer(hidden_channels, hidden_channels), alpha=alpha))

        input_dim1 = int(((num_features * num_features)/2) - (num_features/2) + (hidden_channels*num_layers))
        input_dim = int(((num_features * num_features)/2) - (num_features/2))
        #self.bn = nn.BatchNorm1d(input_dim)
        self.bn=nn.BatchNorm1d(998504)
        self.bnh = nn.BatchNorm1d(hidden_channels*num_layers)

        self.mlp = nn.Sequential(
            #nn.Linear(input_dim1, hidden),
            nn.Linear(998600,hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden, hidden//2),
            nn.BatchNorm1d(hidden//2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden//2, hidden//2),
            nn.BatchNorm1d(hidden//2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear((hidden//2), args.num_classes),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        xs = [x]
        for conv in self.convs:
            x = conv(x, edge_index)
            xs.append(x)
        
        h = []
        for i, xx in enumerate(xs):
            if i == 0:
                xx = xx.reshape(data.num_graphs, x.shape[1], -1)
                x = torch.stack([t.triu().flatten()[t.triu().flatten().nonzero(as_tuple=True)] for t in xx])
                x = self.bn(x)
            else:
                xx = self.aggr(xx, batch)
                h.append(xx)
        
        h = torch.cat(h, dim=1)
        h = self.bnh(h)
        x = torch.cat((x, h), dim=1)
        x = self.mlp(x)
        return x

        
#class GNNWithStochasticAttention(MessagePassing):
#    ''' 
#    This implementation extends MessagePassing, a class provided by PyTorch Geometric, to
#    define a custom GNN layer with attention mechanisms, which is applied between nodes.
#    '''
#    def __init__(self, in_channels, out_channels, heads=1, alpha=0.5, **kwargs):
#        super(GNNWithStochasticAttention, self).__init__(aggr='mean', **kwargs)
#        self.in_channels = in_channels
#        self.out_channels = out_channels
#        self.heads = heads
#        self.alpha = alpha
#
#        self.lin = nn.Linear(in_channels, heads * out_channels)
#        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * out_channels))
#
#        nn.init.xavier_uniform_(self.lin.weight)
#        nn.init.xavier_uniform_(self.att)
#
#    def forward(self, x, edge_index):
#        x = self.lin(x).view(-1, self.heads, self.out_channels)
#
#        return self.propagate(edge_index, x=x)
#
#    def message(self, edge_index_i, x_i, x_j):
#        x_i = x_i.view(-1, self.heads, self.out_channels)
#        x_j = x_j.view(-1, self.heads, self.out_channels)
#
#        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
#        alpha = torch.softmax(alpha, dim=1)
#
#        # Dropout-like behavior
#        mask = torch.rand_like(alpha) < self.alpha
#        alpha = alpha * mask / self.alpha
#
#        return x_j * alpha.view(-1, self.heads, 1)
        

#class ResidualStochasticAttentionalGNNs(torch.nn.Module):
#    def __init__(self,args, train_dataset, hidden_channels,hidden, num_layers, GNN, #k=0.6,attention_heads=1, attalpha=0.5):
#        super().__init__()
#        self.convs = ModuleList()
#        self.aggr = aggr.MeanAggregation()
#        self.hidden_channels = hidden_channels
#        num_features = train_dataset.num_features
#        self.attention_heads = attention_heads
#        self.attalpha=attalpha

#        #Initialize GNN layers with attention
#        for _ in range(num_layers):
#            self.convs.append(GNNWithStochasticAttention(num_features, hidden_channels, #heads=self.attention_heads, alpha=self.attalpha))

        
#        input_dim1 = int(((num_features * num_features)/2)- (num_features/2)+(hidden_channels*num_layers))
#        input_dim = int(((num_features * num_features)/2)- (num_features/2))
#        self.bn = nn.BatchNorm1d(input_dim)
#        self.bnh = nn.BatchNorm1d(hidden_channels*num_layers)
#        # self.attention = Attention(input_dim1, hidden_channels)
#        self.mlp = nn.Sequential(
#            nn.Linear(input_dim1, hidden),
#            nn.BatchNorm1d(hidden),
#            nn.ReLU(),
#            nn.Dropout(0.5),
#            nn.Linear(hidden, hidden//2),
#            nn.BatchNorm1d(hidden//2),
#            nn.ReLU(),
#            nn.Dropout(0.5),
#            nn.Linear(hidden//2, hidden//2),
#            nn.BatchNorm1d(hidden//2),
#            nn.ReLU(),
#            nn.Dropout(0.5),
#            nn.Linear((hidden//2), args.num_classes),
#        )
#
#    def forward(self, data):
#        x, edge_index, batch = data.x, data.edge_index, data.batch
#        # for conv in self.convs:
#        #     x = conv(x, edge_index).relu()
        
#        xs = [x]        
#        for conv in self.convs:
#            xs += [conv(xs[-1], edge_index).tanh()]
#        h = []
#        for i, xx in enumerate(xs):
#            if i== 0:
#                xx = xx.reshape(data.num_graphs, x.shape[1],-1)
#                x = torch.stack([t.triu().flatten()[t.triu().flatten().nonzero(as_tuple=True)] for t in xx])
#                x = self.bn(x)
#            else:
#                # xx = xx.reshape(data.num_graphs, x.shape[1],-1)
#                xx = self.aggr(xx,batch)
#                # h.append(torch.stack([t.flatten() for t in xx]))
#                h.append(xx)
#        
#        h = torch.cat(h,dim=1)
#        h = self.bnh(h)
#        # x = torch.stack(h, dim=0)
#        x = torch.cat((x,h),dim=1)
#        x = self.mlp(x)
#        return x



#### ATTENTION JUST BETWEEN INPUTS AND OUTPUTS OF THE GNN INSTEAD OF BETWEEN NODES
#class GNNWithPartialAttention(MessagePassing):
#    def __init__(self, in_channels, out_channels, heads=1, alpha=0.5, **kwargs):
#        super(GNNWithPartialAttention, self).__init__(aggr='mean', **kwargs)
#        self.in_channels = in_channels
#        self.out_channels = out_channels
#        self.heads = heads
#        self.alpha = alpha
#
#        self.lin = nn.Linear(in_channels, heads * out_channels)
#        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * out_channels))
#
#        nn.init.xavier_uniform_(self.lin.weight)
#        nn.init.xavier_uniform_(self.att)
#
#    def forward(self, x, edge_index):
#        x = self.lin(x).view(-1, self.heads, self.out_channels)
#
#        return self.propagate(edge_index, x=x)
#
#    def message(self, edge_index_i, x_i, x_j):
#        x_i = x_i.view(-1, self.heads, self.out_channels)
#        x_j = x_j.view(-1, self.heads, self.out_channels)
#
#        if self.alpha < 1.0:
#            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
#            alpha = torch.softmax(alpha, dim=1)
#
#            # Dropout-like behavior
#            mask = torch.rand_like(alpha) < self.alpha
#            alpha = alpha * mask / self.alpha
#        else:
#            alpha = torch.ones_like(x_i)
#
#        return x_j * alpha.view(-1, self.heads, 1)


#class ResidualGNNsWithPartialAttention(nn.Module):
#    def __init__(self, args, train_dataset, hidden_channels, hidden, num_layers, k=0.6, #attention_heads=1, alpha=0.5):
#        super().__init__()
#        self.convs = ModuleList()
#        self.aggr = global_mean_pool
#        self.hidden_channels = hidden_channels
#        num_features = train_dataset.num_features
#        self.attention_heads = attention_heads
#        self.alpha = alpha
        
        # Initialize GNN layers with partial attention
#        for _ in range(num_layers):
#            self.convs.append(GNNWithPartialAttention(num_features, hidden_channels, #heads=self.attention_heads, alpha=self.alpha))
#
#        input_dim1 = int(((num_features * num_features)/2)- (num_features/2)+#(hidden_channels*num_layers))
#        input_dim = int(((num_features * num_features)/2)- (num_features/2))
#        self.bn = BatchNorm1d(input_dim)
#        self.bnh = BatchNorm1d(hidden_channels*num_layers)
#        
#        self.mlp = nn.Sequential(
#            nn.Linear(input_dim1, hidden),
#            nn.BatchNorm1d(hidden),
#            nn.ReLU(),
#            nn.Dropout(0.5),
#            nn.Linear(hidden, hidden//2),
#            nn.BatchNorm1d(hidden//2),
#            nn.ReLU(),
#            nn.Dropout(0.5),
#            nn.Linear(hidden//2, hidden//2),
#            nn.BatchNorm1d(hidden//2),
#            nn.ReLU(),
#            nn.Dropout(0.5),
#            nn.Linear((hidden//2), args.num_classes),
#        )
#
#    def forward(self, data):
#        x, edge_index, batch = data.x, data.edge_index, data.batch
#        xs = [x]        
#        for conv in self.convs:
#            xs += [conv(xs[-1], edge_index)]
#
#        h = []
#        for i, xx in enumerate(xs):
#            if i== 0:
#                xx = xx.reshape(data.num_graphs, x.shape[1],-1)
#                x = torch.stack([t.triu().flatten()[t.triu().flatten().nonzero(as_tuple=True)] for #t in xx])
#                x = self.bn(x)
#            else:
#                xx = self.aggr(xx, batch)
#                h.append(xx)
#        
#        h = torch.cat(h, dim=1)
#        h = self.bnh(h)
#        x = torch.cat((x, h), dim=1)
#        x = self.mlp(x)
#        return x


##### RESIDUAL NETWORK WITH ATTENTION BETWEEN INPUT AND GNN 
class ResidualGNNsWithInputAttention(nn.Module):
    def __init__(self, args, train_dataset, hidden_channels, hidden, num_layers, GNN, k=0.6, attention_heads=1, alpha=0.5, x=None):
        super().__init__()
        self.convs = ModuleList()
        self.aggr = global_mean_pool
        self.hidden_channels = hidden_channels
        num_features = train_dataset.num_features
        self.attention_heads = attention_heads
        self.alpha = alpha
        
        if args.model=="ChebConv":
            if num_layers>0:
                self.convs.append(GNN(num_features, hidden_channels,K=5))
                for i in range(0, num_layers - 1):
                    self.convs.append(GNN(hidden_channels, hidden_channels,K=5))
        else:
            if num_layers>0:
                self.convs.append(GNN(num_features, hidden_channels))
                for i in range(0, num_layers - 1):
                    self.convs.append(GNN(hidden_channels, hidden_channels))

        input_dim1 = int(((num_features * num_features)/2)- (num_features/2)+(hidden_channels*num_layers))
        input_dim = int(((num_features * num_features)/2)- (num_features/2))
        self.bn = BatchNorm1d(input_dim)
        self.bnh = BatchNorm1d(hidden_channels*num_layers)
        
        self.att_lin = nn.Linear(hidden_channels*num_layers, 1)
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim1, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden, hidden//2),
            nn.BatchNorm1d(hidden//2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden//2, hidden//2),
            nn.BatchNorm1d(hidden//2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear((hidden//2), args.num_classes),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        xs = [x]        
        for conv in self.convs:
            xs += [conv(xs[-1], edge_index)]

        h = []
        for i, xx in enumerate(xs):
            if i== 0:
                xx = xx.reshape(data.num_graphs, x.shape[1],-1)
                x = torch.stack([t.triu().flatten()[t.triu().flatten().nonzero(as_tuple=True)] for t in xx])
                x = self.bn(x)
            else:
                xx = self.aggr(xx, batch)
                h.append(xx)
        
        h = torch.cat(h, dim=1)
        h = self.bnh(h)
        

        # Calculate attention between input features and final output
        x = x.view(x.size(0), -1, 1)
        att_weights = torch.sigmoid(self.att_lin(x))
        
        x = att_weights * x + (1 - att_weights) * h * self.alpha

        x = torch.cat((x, h), dim=1)
        x = self.mlp(x)
        return x



#### EDGE LEVEL ATTENTION 

class EdgeLevelAttention(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1):
        super(EdgeLevelAttention, self).__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads

        # Linear transformations for computing attention scores
        self.lin_src = nn.Linear(in_channels, heads * out_channels)
        self.lin_dst = nn.Linear(in_channels, heads * out_channels)
        self.att = nn.Linear(2 * out_channels, heads)

        # Initialization
        nn.init.xavier_uniform_(self.lin_src.weight)
        nn.init.xavier_uniform_(self.lin_dst.weight)
        nn.init.xavier_uniform_(self.att.weight)

    def forward(self, x, edge_index):
        # x: Node feature matrix
        # edge_index: Edge indices

        # Linear transformation for source and destination nodes
        x_src = self.lin_src(x)
        x_dst = self.lin_dst(x)

        # Add self-loops to edge indices
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        return self.propagate(edge_index, x=(x_src, x_dst))

    def message(self, x_j, x_i):
        # x_i: Source node features
        # x_j: Destination node features

        # Compute attention scores
        alpha = self.att(torch.cat([x_i, x_j], dim=-1))
        alpha = torch.softmax(alpha, dim=1)

        # Weight the source node features by attention scores
        return x_i * alpha.view(-1, self.heads, 1)



class ResidualGNNsWithEdgeLevelAttention(nn.Module):
    def __init__(self, args, train_dataset, hidden_channels, hidden, num_layers, k=0.6, heads=1):
        super().__init__()
        self.convs = ModuleList()
        self.aggr = global_mean_pool
        self.hidden_channels = hidden_channels
        num_features = train_dataset.num_features
        self.attention_heads = heads
        
        # Initialize GNN layers with edge-level attention
        for _ in range(num_layers):
            self.convs.append(EdgeLevelAttention(num_features, hidden_channels, heads=self.attention_heads))

        input_dim1 = int(((num_features * num_features)/2)- (num_features/2)+(hidden_channels*num_layers))
        input_dim = int(((num_features * num_features)/2)- (num_features/2))
        self.bn = BatchNorm1d(input_dim)
        self.bnh = BatchNorm1d(hidden_channels*num_layers)
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim1, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden, hidden//2),
            nn.BatchNorm1d(hidden//2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden//2, hidden//2),
            nn.BatchNorm1d(hidden//2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear((hidden//2), args.num_classes),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        xs = [x]        
        for conv in self.convs:
            xs += [conv(xs[-1], edge_index)]

        h = []
        for i, xx in enumerate(xs):
            if i== 0:
                xx = xx.reshape(data.num_graphs, x.shape[1],-1)
                x = torch.stack([t.triu().flatten()[t.triu().flatten().nonzero(as_tuple=True)] for t in xx])
                x = self.bn(x)
            else:
                xx = self.aggr(xx, batch)
                h.append(xx)
        
        h = torch.cat(h, dim=1)
        h = self.bnh(h)

        x = torch.cat((x, h), dim=1)
        x = self.mlp(x)
        return x




#### RESIDUAL WITH SELF ATTENTION 


import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, in_features, heads=1):
        super(SelfAttention, self).__init__()
        self.in_features = in_features
        self.heads = heads
        
        # Linear transformations for computing attention scores
        self.lin_q = nn.Linear(in_features, in_features)
        self.lin_k = nn.Linear(in_features, in_features)
        self.lin_v = nn.Linear(in_features, in_features)
        self.att = nn.Linear(in_features, heads)

    def forward(self, x):
        # Linear transformations
        q = self.lin_q(x)
        k = self.lin_k(x)
        v = self.lin_v(x)

        # Split heads
        q = q.view(-1, self.heads, self.in_features // self.heads)
        k = k.view(-1, self.heads, self.in_features // self.heads)
        v = v.view(-1, self.heads, self.in_features // self.heads)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.in_features // self.heads)**0.5
        scores = torch.softmax(scores, dim=-1)

        # Apply attention to values
        att = torch.matmul(scores, v)
        att = att.view(-1, self.in_features)

        return att



class ResidualGNNsWithSelfAttention(nn.Module):
    def __init__(self, args, train_dataset, hidden_channels, hidden, num_layers, GNN, k=0.6, heads=1):
        super().__init__()
        self.convs = ModuleList()
        self.aggr = global_mean_pool
        self.hidden_channels = hidden_channels
        num_features = train_dataset.num_features
        self.attention_heads = heads
        self.heads = heads
        
        # Initialize GNN layers
        if args.model=="ChebConv":
            if num_layers>0:
                self.convs.append(GNN(num_features, hidden_channels,K=5))
                for i in range(0, num_layers - 1):
                    self.convs.append(GNN(hidden_channels, hidden_channels,K=5))
        else:
            if num_layers>0:
                self.convs.append(GNN(num_features, hidden_channels))
                for i in range(0, num_layers - 1):
                    self.convs.append(GNN(hidden_channels, hidden_channels))

        input_dim1 = int(((num_features * num_features)/2)- (num_features/2)+(hidden_channels*num_layers))
        input_dim = int(((num_features * num_features)/2)- (num_features/2))
        self.bn = BatchNorm1d(input_dim)
        self.bnh = BatchNorm1d(hidden_channels*num_layers)
        
        self.attention = SelfAttention(hidden_channels, heads=self.heads)
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim1, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden, hidden//2),
            nn.BatchNorm1d(hidden//2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden//2, hidden//2),
            nn.BatchNorm1d(hidden//2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear((hidden//2), args.num_classes),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        xs = [x]        
        for conv in self.convs:
            xs += [conv(xs[-1], edge_index)]

        h = []
        for i, xx in enumerate(xs):
            if i== 0:
                xx = xx.reshape(data.num_graphs, x.shape[1],-1)
                x = torch.stack([t.triu().flatten()[t.triu().flatten().nonzero(as_tuple=True)] for t in xx])
                x = self.bn(x)
            else:
                xx = self.aggr(xx, batch)
                h.append(xx)
        
        h = torch.cat(h, dim=1)
        h = self.bnh(h)

        # Apply self-attention
        h = self.attention(h)

        x = torch.cat((x, h), dim=1)
        x = self.mlp(x)
        return x



#OTHER 
from torch_geometric.nn import global_add_pool

class ResidualGNNs2(torch.nn.Module):
    def __init__(self, args, train_dataset, hidden_channels, hidden, num_layers, GNN, k=0.6, use_attention=True):
        super().__init__()
        self.convs = ModuleList()
        self.aggr = global_add_pool
        self.hidden_channels = hidden_channels
        num_features = train_dataset.num_features
        self.use_attention = use_attention
        if args.model == "ChebConv":
            if num_layers > 0:
                self.convs.append(GNN(num_features, hidden_channels, K=5))
                for _ in range(0, num_layers - 1):
                    self.convs.append(GNN(hidden_channels, hidden_channels, K=5))
        else:
            if num_layers > 0:
                self.convs.append(GNN(num_features, hidden_channels))
                for _ in range(0, num_layers - 1):
                    self.convs.append(GNN(hidden_channels, hidden_channels))
        
        input_dim1 = int(((num_features * num_features)/2)- (num_features/2)+(hidden_channels*num_layers))
        input_dim = int(((num_features * num_features)/2)- (num_features/2))
        self.bn = nn.BatchNorm1d(input_dim)
        self.bnh = nn.BatchNorm1d(hidden_channels*num_layers)

        if self.use_attention:
            self.attention = nn.Linear(hidden_channels*num_layers, 1)
        
        self.mlp = nn.Sequential(
            nn.Linear(499501, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden, hidden//2),
            nn.BatchNorm1d(hidden//2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden//2, hidden//2),
            nn.BatchNorm1d(hidden//2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear((hidden//2), args.num_classes),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        xs = [x]        
        for conv in self.convs:
            xs += [conv(xs[-1], edge_index).tanh()]
        h = []
        for i, xx in enumerate(xs):
            if i == 0:
                xx = xx.reshape(data.num_graphs, x.shape[1], -1)
                x = torch.stack([t.triu().flatten()[t.triu().flatten().nonzero(as_tuple=True)] for t in xx])
                x = self.bn(x)
            else:
                xx = self.aggr(xx, batch)
                h.append(xx)
        
        h = torch.cat(h, dim=1)
        h = self.bnh(h)
        if self.use_attention:
            #original applying always the attention
            att_weights = self.attention(h).softmax(dim=1)
            #h = torch.sum(h * att_weights, dim=1)

            
            probability=0.6
            mask = torch.rand_like(att_weights) < probability
            mask = mask.float()  # Convert to float
            
            att_weights = att_weights * mask  # Apply the random mask
            h = torch.sum(h * att_weights, dim=1)

        h = h.unsqueeze(1) 
        x = torch.cat((x, h), dim=1)
        x = self.mlp(x)
        return x
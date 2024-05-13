import torch
import torch.nn as nn
import torch.nn.functional as F
""" 
現状, lastpoolerは, padding_mask=1の文字がずっと続いた後に0の文字が続くということを前提としている。


"""
class MeanPooler(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input, padding_mask):
        """
        Parameters
        ----------
        input: torch.tensor of torch.float [length, batch_size, feature_size]
        padding_mask: torch.tensor of torch.bool [length, batch_size]
            Pisitions where padding_mask is True is regarded as pad_token and ignored.
            (input_tokens == pad_token).transpose(0, 1)を想定
        """
        padding_mask = ~padding_mask.unsqueeze(-1)
        return torch.sum(input*padding_mask, dim=0)/torch.sum(padding_mask, dim=0)

class StartPooler(nn.Module):
    def __init__(self):
        # index_selectを使うことも可能?
        super().__init__()
    def forward(self, input):
        return input[0]
        
class MaxPooler(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input: torch.Tensor, padding_mask):
        masked_max_input = input.masked_fill(padding_mask.unsqueeze(-1), -torch.inf)
        return torch.max(masked_max_input, dim=0)[0]

class MeanStartMaxPooler(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input: torch.Tensor, padding_mask: torch.Tensor):
        padding_mask = ~padding_mask.unsqueeze(-1)
        masked_max_input = input.masked_fill(~padding_mask, -torch.inf)
        return torch.cat([
            torch.sum(input*padding_mask, dim=0)/torch.sum(padding_mask, dim=0),
            input[self.slice],
            torch.max(masked_max_input, dim=0)[0]], dim=-1)
    
class MeanStartEndMaxPooler(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input: torch.Tensor, padding_mask: torch.Tensor, end_mask: torch.Tensor):
        """
        padding_mask: (src == pad_token).transpose(0, 1)
        last_mask: (src == end_token).transpose(0, 1)のようなものを想定        
        """
        padding_mask = padding_mask.unsqueeze(-1)
        masked_max_input = input.masked_fill(padding_mask, -torch.inf)
        padding_mask = ~padding_mask
        end_mask = end_mask.unsqueeze(-1)
        return torch.cat([
            torch.sum(input*padding_mask, dim=0)/torch.sum(padding_mask, dim=0),
            input[0], 
            torch.sum(input*end_mask, dim=0),
            torch.max(masked_max_input, dim=0)[0]], dim=-1)
    
class MeanStdStartEndMaxMinPooler(nn.Module):
    def forward(self, input: torch.Tensor, padding_mask: torch.Tensor, end_mask: torch.Tensor):
        padding_mask = padding_mask.unsqueeze(-1)
        masked_max_input = input.masked_fill(padding_mask, -torch.inf)
        masked_min_input = input.masked_fill(padding_mask, torch.inf)
        padding_mask = ~padding_mask
        end_mask = end_mask.unsqueeze(-1)
        mean = torch.sum(input*padding_mask, dim=0)/torch.sum(padding_mask, dim=0)
        std = torch.sum(((input - mean.unsqueeze(0))**2)*padding_mask, dim=0)/torch.sum(padding_mask, dim=0)
        return torch.cat([
            mean,
            std,
            input[0], 
            torch.sum(input*end_mask, dim=0),
            torch.max(masked_max_input, dim=0)[0],
            torch.min(masked_min_input, dim=0)[0]], dim=-1)

class NoAffinePooler(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.mean_norm = nn.LayerNorm(input_size[-1], elementwise_affine=False)
        self.start_norm = nn.LayerNorm(input_size[-1], elementwise_affine=False)
        self.max_norm = nn.LayerNorm(input_size[-1], elementwise_affine=False)
        self.output_size = input_size[1:]
        self.output_size[-1] = self.output_size[-1]*3
    def forward(self, input, padding_mask):
        padding_mask = ~padding_mask
        masked_max_input = input + torch.log(padding_mask).unsqueeze(-1)
        return torch.cat([
            self.mean_norm(torch.sum(input*padding_mask.unsqueeze(-1), dim=0)/torch.sum(padding_mask, dim=0).unsqueeze(-1)),
            self.start_norm(input[0]),
            self.max_norm(torch.max(masked_max_input, dim=0)[0])], dim=-1)

class NemotoPooler(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.output_size = input_size[1:]
        self.output_size[-1] = self.output_size[-1]*3
    def forward(self, input):
        mx = torch.max(input,dim=0)[0]
        ave = torch.mean(input,dim=0)
        first = input[0]
        return torch.cat([mx,ave,first],dim=1)

class GraphPooler(nn.Module):
    def __init__(self, node_size, edge_size):
        """
        [WARNING] padding_mask is different from the above pooling functions
        """
        super().__init__()
        self.node_norm = nn.LayerNorm(node_size, elementwise_affine=False)
        self.edge_norm = nn.LayerNorm(edge_size, elementwise_affine=False)

    def forward(self, node: torch.Tensor, edge: torch.Tensor, padding_mask: torch.Tensor):
        """
        Parameters
        ----------
        node: (float)[n_node, batch_size, node_size]
        edge: (float)[batch_size, n_node, n_node, edge_size]
        padding_mask: (bool)[batch_size, n_node]
            padding_mask = node == pad_token
        """
        n_node, _, _ = node.shape
        node_padding_mask = padding_mask.T.unsqueeze(-1) # [N, B, 1]
        node = torch.sum(torch.masked_fill(node, node_padding_mask, 0), dim=0) / (n_node - torch.sum(node_padding_mask, dim=0)) # [B, Fn]
        edge_padding_mask = (padding_mask.unsqueeze(-1) | padding_mask.unsqueeze(-2)).unsqueeze(-1) # [B, N, N, 1]
        edge = torch.sum(torch.masked_fill(edge, edge_padding_mask, 0), dim=(1, 2)) / \
              (n_node**2 - torch.sum(edge_padding_mask, dim=(1,2))) # [B, Fe]
        return torch.cat([self.node_norm(node), self.edge_norm(edge)], dim=-1) # [B, Fn+Fe]


class GraphMeanMaxPooler(nn.Module):
    def __init__(self, node_size, edge_size, eps=1e-5, layer_norm=True):
        super().__init__()
        self.node_size = node_size
        self.edge_size = edge_size
        self.eps = eps
        self.layer_norm = layer_norm
    
    def forward(self, node: torch.Tensor, edge: torch.Tensor, padding_mask: torch.Tensor):
        """
        Parameters
        ----------
        node: (float)[n_node, batch_size, node_size]
        edge: (float)[batch_size, n_node, n_node, edge_size]
        padding_mask: (bool)[batch_size, n_node]
            padding_mask = node == pad_token
        """
        n_node, batch_size, _ = node.shape
        node_padding_mask = padding_mask.T.unsqueeze(-1) # [N, B, 1]
        node_mean = torch.masked_fill(node, node_padding_mask, 0) # [N, B, Fn]
        node_mean = torch.sum(node_mean, dim=0) / (n_node - torch.sum(node_padding_mask, dim=0)) # [B, Fn]
        node_max = torch.masked_fill(node, node_padding_mask, 0) # [N, B, Fn]
        node_max = torch.max(node_max, dim=0).values # [B, Fn]

        edge_padding_mask = (padding_mask.unsqueeze(-1) | padding_mask.unsqueeze(-2)).unsqueeze(-1) # [B, N, N, 1]
        edge_mean = torch.masked_fill(edge, edge_padding_mask, 0) # [B, N, N, Fe]
        edge_mean = torch.sum(edge_mean, dim=(1, 2)) / \
              (n_node**2 - torch.sum(edge_padding_mask, dim=(1,2))) # [B, Fe]
        edge_max = torch.masked_fill(edge, edge_padding_mask, -torch.inf) # [B, N, N, Fe]
        edge_max = torch.max(edge_max.view(batch_size, n_node*n_node, -1), dim=1).values # [B, Fe]
        if self.layer_norm:
            return torch.cat([
                F.layer_norm(node_mean, (self.node_size,), eps=self.eps),
                F.layer_norm(node_max, (self.node_size,), eps=self.eps),
                F.layer_norm(edge_mean, (self.edge_size,), eps=self.eps),
                F.layer_norm(edge_max, (self.edge_size,), eps=self.eps)],
                dim=-1) # [B, Fn*2+Fe*2]
        else:
            return torch.cat([node_mean, node_max, edge_mean, edge_max], dim=-1)
        


class GraphStartMeanMaxPooler(nn.Module):
    def __init__(self, node_size, edge_size, eps=1e-5):
        super().__init__()
        self.node_size = node_size
        self.edge_size = edge_size
        self.eps = eps
    
    def forward(self, node: torch.Tensor, edge: torch.Tensor, padding_mask: torch.Tensor):
        """
        Parameters
        ----------
        node: (float)[n_node, batch_size, node_size]
        edge: (float)[batch_size, n_node, n_node, edge_size]
        padding_mask: (bool)[batch_size, n_node]
            padding_mask = node == pad_token
        """
        n_node, batch_size, _ = node.shape
        node_padding_mask = padding_mask.T.unsqueeze(-1) # [N, B, 1]
        node_start = node[0]
        node_mean = torch.masked_fill(node, node_padding_mask, 0) # [N, B, Fn]
        node_mean = torch.sum(node_mean, dim=0) / (n_node - torch.sum(node_padding_mask, dim=0)) # [B, Fn]
        node_max = torch.masked_fill(node, node_padding_mask, 0) # [N, B, Fn]
        node_max = torch.max(node_max, dim=0).values # [B, Fn]

        edge_padding_mask = (padding_mask.unsqueeze(-1) | padding_mask.unsqueeze(-2)).unsqueeze(-1) # [B, N, N, 1]
        edge_mean = torch.masked_fill(edge, edge_padding_mask, 0) # [B, N, N, Fe]
        edge_start = torch.sum(edge_mean[:,0], dim=1) / (n_node - torch.sum(node_padding_mask, dim=0)) # [B, Fe]
        edge_mean = torch.sum(edge_mean, dim=(1, 2)) / \
              (n_node**2 - torch.sum(edge_padding_mask, dim=(1,2))) # [B, Fe]
        edge_max = torch.masked_fill(edge, edge_padding_mask, -torch.inf) # [B, N, N, Fe]
        edge_max = torch.max(edge_max.view(batch_size, n_node*n_node, -1), dim=1).values # [B, Fe]

        return torch.cat([
            F.layer_norm(node_start, (self.node_size,), eps=self.eps),
            F.layer_norm(node_mean, (self.node_size,), eps=self.eps),
            F.layer_norm(node_max, (self.node_size,), eps=self.eps),
            F.layer_norm(edge_start, (self.edge_size,), eps=self.eps),
            F.layer_norm(edge_mean, (self.edge_size,), eps=self.eps),
            F.layer_norm(edge_max, (self.edge_size,), eps=self.eps)],
            dim=-1)

class GraphPooler2(nn.Module):
    def __init__(self, node_size, edge_size, 
            node_start=False, node_mean=False, node_max=False,
            edge_start=False, edge_mean=False, edge_max=False,
            eps=1e-5):
        super().__init__()
        self.node_size = node_size
        self.edge_size = edge_size
        self.node_start = node_start
        self.node_mean = node_mean
        self.node_max = node_max
        self.edge_start = edge_start
        self.edge_mean = edge_mean
        self.edge_max = edge_max
        self.eps = eps
    
    def forward(self, node: torch.Tensor, edge: torch.Tensor, padding_mask: torch.Tensor):
        """
        Parameters
        ----------
        node: (float)[n_node, batch_size, node_size]
        edge: (float)[batch_size, n_node, n_node, edge_size]
        padding_mask: (bool)[batch_size, n_node]
            padding_mask = node == pad_token
        """
        n_node, batch_size, _ = node.shape
        pooleds = []
        node_padding_mask = padding_mask.T.unsqueeze(-1) # [N, B, 1]
        if self.node_start:
            node_start = node[0]
            pooleds.append(F.layer_norm(node_start, (self.node_size,), eps=self.eps))
        if self.node_mean:
            node_mean = torch.masked_fill(node, node_padding_mask, 0)
            node_mean = torch.sum(node_mean, dim=0) / (n_node - torch.sum(node_padding_mask, dim=0)) # [B, Fn]
            pooleds.append(F.layer_norm(node_mean, (self.node_size,), eps=self.eps))
        if self.node_max:
            node_max = torch.masked_fill(node, node_padding_mask, 0) # [N, B, Fn]
            node_max = torch.max(node_max, dim=0).values # [B, Fn]
            pooleds.append(F.layer_norm(node_max, (self.node_size,), eps=self.eps))

        edge_padding_mask = (padding_mask.unsqueeze(-1) | padding_mask.unsqueeze(-2)).unsqueeze(-1) # [B, N, N, 1]
        if self.edge_start:
            edge_mean = torch.masked_fill(edge, edge_padding_mask, 0) # [B, N, N, Fe]
            edge_start = torch.sum(edge_mean[:,0], dim=1) / (n_node - torch.sum(node_padding_mask, dim=0)) # [B, Fe]
            pooleds.append(F.layer_norm(edge_start, (self.edge_size,), eps=self.eps))
        if self.edge_mean:
            if not self.edge_start:
                edge_mean = torch.masked_fill(edge, edge_padding_mask, 0) # [B, N, N, Fe]
            edge_mean = torch.sum(edge_mean, dim=(1, 2)) / \
                (n_node**2 - torch.sum(edge_padding_mask, dim=(1,2))) # [B, Fe]
            pooleds.append(F.layer_norm(edge_mean, (self.edge_size, ), eps=self.eps))
        if self.edge_max:
            edge_max = torch.masked_fill(edge, edge_padding_mask, -torch.inf) # [B, N, N, Fe]
            edge_max = torch.max(edge_max.view(batch_size, n_node*n_node, -1), dim=1).values # [B, Fe]
            pooleds.append(F.layer_norm(edge_max, (self.edge_size, ), eps=self.eps))
        return torch.cat(pooleds, dim=-1) # [B, F]





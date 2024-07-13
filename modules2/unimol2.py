"""
240119作成
Unimolのdecoder

その後 いつか不明(~240628)だが, UnimolTask用に利用

TODO: rngのseedをload_state_dictに保存できるようにする
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .. import register_module

def fill_attn_mask(attn_repr, padding_mask, bsz, n_node, fill_val=float("-inf")):
    attn_repr = attn_repr.view(bsz, -1, n_node, n_node)
    attn_repr.masked_fill_(
        padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
        fill_val,
    )
    attn_repr = attn_repr.view(-1, n_node, n_node)
    return attn_repr

@register_module
class UnimolMasker(nn.Module):
    def __init__(self, mask_token, voc_size, special_tokens, seed):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.mask_token = mask_token
        self.voc_size = voc_size
        rand_weight = np.ones(voc_size, dtype=float)
        rand_weight[special_tokens] = 0
        rand_weight /= np.sum(rand_weight)
        self.rand_weight = nn.Parameter(torch.tensor(rand_weight), requires_grad=False)

    def forward(self, input: torch.Tensor, mask_mask: torch.Tensor, rand_mask: torch.Tensor):
        input[mask_mask] = self.mask_token
        if torch.any(rand_mask):
            input[rand_mask] = torch.multinomial(self.rand_weight, torch.sum(rand_mask).item(), 
                replacement=True).to(input.dtype)
        return input

@register_module
class UnimolBoolMasker(nn.Module):
    def __init__(self, seed, mask_token=False): 
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.mask_token = mask_token
    
    def forward(self, input: torch.Tensor, mask_mask: torch.Tensor, rand_mask: torch.Tensor):
        device = input.device
        input[mask_mask] = self.mask_token
        if torch.any(rand_mask):
            input[rand_mask] = (torch.rand_like(input[rand_mask], device=device, dtype=float) < 0.5).to(input.dtype)
        return input

@register_module
class UnimolMaskMaker(nn.Module):
    def __init__(self, p_target, p_unmask, p_rand, atom_pad_token, seed, noise):
        super().__init__()
        self.p_target = p_target
        self.p_unmask = p_unmask
        self.p_rand = p_rand
        self.atom_pad_token = atom_pad_token
        self.rng = np.random.default_rng(seed)
        self.noise = noise

    def forward(self, atoms: torch.Tensor, coordinates: torch.Tensor):
        """
        Parameters
        ----------
        atoms(long)[B, L]:
        coordinates(float)[B, L, 3]

        Returns
        -------

        B: batch_size
        L: length
        apo: apair_emb_size
        """
        batch_size, length = atoms.shape
        device = atoms.device

        # calc mask
        ns_target = torch.sum(atoms != self.atom_pad_token, dim=-1)
        target_mask = torch.full((batch_size, length), fill_value=False, 
            dtype=torch.bool, device=device)
        for b in range(batch_size):
            n_target = ns_target[b].item()
            n_mask = int(n_target*self.p_target+self.rng.random())
            mask_idx = self.rng.choice(n_target, n_mask)
            target_mask[b][mask_idx] = True
        decision = torch.tensor(self.rng.random((batch_size, length)), device=device)
        rand_mask = target_mask & (self.p_unmask <= decision) & (decision < self.p_unmask+self.p_rand)
        mask_mask = target_mask & (self.p_unmask+self.p_rand <= decision)

        rand_mask2 = rand_mask.unsqueeze(1) & rand_mask.unsqueeze(2)
        mask_mask2 = mask_mask.unsqueeze(1) & mask_mask.unsqueeze(2)

        coordinates[rand_mask|mask_mask] += torch.tensor(self.rng.uniform(low=-self.noise, high=self.noise, 
            size=(torch.sum(rand_mask|mask_mask).item(), 3)), device=device)

        return coordinates, mask_mask, rand_mask, mask_mask2, rand_mask2


@register_module
class UnimolCoordHead(nn.Module):
    def __init__(self, input_size, pad_token):
        super().__init__()
        self.ln = nn.LayerNorm(input_size)
        self.pad_token = pad_token
        self.proj = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.GELU(),
            nn.Linear(input_size, 1)
        )
    def forward(self, x: torch.Tensor, coord: torch.Tensor, nodes: torch.Tensor):
        """
        Parameters
        ----------
        x: [B, L, L, D]
            Difference of apairs
        coord: [B, L, 3]
        nodes: [B, L]

        Returns
        -------
        coord: [B, L, 3]
        
        """
        x = self.ln(x) # [B, L, L, D]
        padding_mask = nodes.eq(self.pad_token)
        x.masked_fill_(padding_mask.unsqueeze(-1).unsqueeze(1), 0)
        padding_mask = padding_mask.to(torch.float)
        atom_num = torch.sum((1-padding_mask).type_as(x), dim=1).view(-1, 1, 1, 1)
        attn_probs = self.proj(x) # [B, L, L, D]
        delta_pos = coord.unsqueeze(1) - coord.unsqueeze(2)
        coord_update = delta_pos / atom_num * attn_probs
        pair_coord_mask = (1-padding_mask).unsqueeze(1)*(1-padding_mask).unsqueeze(2)
        coord_update = coord_update * pair_coord_mask.unsqueeze(-1)
        coord_update = torch.sum(coord_update, dim=2)
        return coord + coord_update

@register_module
class UnimolDistanceHead(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.GELU(),
            nn.LayerNorm(input_size),
            nn.Linear(input_size, 1))
    
    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x: [B, L, L, input_size(nhead)]


        
        """
        x = self.proj(x).squeeze(-1)
        x = (x+x.transpose(-1, -2))*0.5
        return x

@register_module
class UnimolMLMLoss(nn.Module):
    def __init__(self, pad_token):
        super().__init__()
        self.pad_token = pad_token

    def forward(self, input, target, mask_mask):
        """
        input: [L, B, V]
        target: [B, L]
        mask_mask: [B, L]
        
        """
        return F.cross_entropy(input.transpose(0, 1)[mask_mask], target[mask_mask],
            ignore_index=self.pad_token, reduction='mean')
        
@register_module
class UnimolCoordLoss(nn.Module):
    def __init__(self):
        super().__init__()
        pass
    def forward(self, input, target, mask_mask):
        """
        input: [B, L, 3]
        target: [B, L, 3]
        mask_mask: [B, L]
        """
        return F.smooth_l1_loss(input[mask_mask].view(-1, 3), target[mask_mask].view(-1, 3))

@register_module
class UnimolDistanceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target_coord, mask_mask, padding_mask):
        """
        input: [B, L, L]
        target_coord: [B, L, 3]
        mask_mask: [B, L]
        padding_mask: [B, L]
        """
        target = target_coord.unsqueeze(1) - target_coord.unsqueeze(2) # [b, l, l, 3]
        target = torch.sqrt((target**2).sum(dim=-1)) # [b, l, l]
        

        masked_distance = input[mask_mask, :]
        masked_distance_target = target[
            mask_mask
        ]
        # padding distance
        nb_masked_tokens = mask_mask.sum(dim=-1)
        masked_src_tokens = torch.logical_not(padding_mask)
        masked_src_tokens_expanded = torch.repeat_interleave(masked_src_tokens, nb_masked_tokens, dim=0)
        # normalize … 不要では?
        # masked_distance_target = ( masked_distance_target.float() - self.dist_mean) / self.dist_std
        masked_dist_loss = F.smooth_l1_loss(
            masked_distance[masked_src_tokens_expanded].view(-1).float(),
            masked_distance_target[masked_src_tokens_expanded].view(-1),
            reduction="mean",
            beta=1.0,
        )
        return masked_dist_loss


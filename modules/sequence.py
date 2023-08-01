"""
Created on 2023.7.8
病理画像と同じmodelsになるようにインターフェイスを変更

@author: YoshikaiY
version: 12

"""

import numpy as np
import torch
import torch.nn as nn
from addict import Dict
from ..utils import check_leftargs
from ..models import function_name2func, init_config2func, Tunnel, Module, function_config2func
from .seq_components import PositionalEncoding, pooler_type2class
from ..models2 import SelfAttentionLayer

class TeacherForcerModule(nn.Module):
    name = 'teacherforcer'
    def __init__(self, logger, sizes, input, output, length_dim, **kwargs):
        super().__init__()
        check_leftargs(self, logger, kwargs)
        self.input = input
        self.input_slices = [slice(None)]*length_dim+[slice(None, -1)]
        self.target_slices = [slice(None)]*length_dim+[slice(1, None)]
        self.output_input = output['input']
        self.output_target = output['target']
        size = sizes[self.input].copy()
        if isinstance(size[length_dim], int):
            size[length_dim] -= 1
        else:
            size[length_dim] = size[length_dim] + '-1'
        sizes[self.output_input] = size.copy()
        sizes[self.output_target] = size.copy()

    def forward(self, batch, mode):
        input = batch[self.input]
        batch[self.output_input] = input[self.input_slices]
        batch[self.output_target] = input[self.target_slices]
        return batch

class PoolerModule(nn.Module):
    name = 'pooler'
    def __init__(self, logger, sizes, input, output, pooler, **kwargs):
        """
        input:
            input: str
            input_padding_mask: str
        """
        super().__init__()
        check_leftargs(self, logger, kwargs)
        self.input = input['input']
        self.input_padding_mask = input['padding_mask']
        self.output = output
        self.pooler = pooler_type2class[pooler](input_size=sizes[self.input])
        sizes[self.output] = self.pooler.output_size
    def forward(self, batch, mode):
        batch[self.output] = self.pooler(input_=batch[self.input], 
            padding_mask=1-batch[self.input_padding_mask].T.to(torch.int))

class VAE(nn.Module):
    name = 'vae'
    def __init__(self, logger, sizes, input, output, modes=None, var_coef=1.0, eval_vae=False, **kwargs):
        """
        logger: logging.Logger
        sizes: {param_name:str : [size:int]}
        input:
            mu: str
            var: str
        output: str
        var_coef: float
        eval_vae: bool        
        
        """
        super().__init__()
        check_leftargs(self, logger, kwargs)
        self.modes = modes
        self.input_mu = input['mu']
        self.input_var = input['var']
        self.output = output
        self.var_coef = var_coef
        self.eval_vae = eval_vae
        self.latent_size = sizes[self.input_mu]
        sizes[self.output] = sizes[self.input_mu]
        
    def forward(self, batch, mode):
        if self.modes is not None and mode not in self.modes:
            return
        if mode == 'generate':
            batch[self.output] = {torch.randn((batch['batch_size'], *self.latent_size), device=batch['device'])}
        else:
            mu = batch[self.input_mu]
            var = batch[self.input_var]
            batch['mu'] = mu
            batch['var'] = var

            if mode == 'train' or self.eval_vae:
                latent = mu + torch.randn(*mu.shape, device=mu.device)*torch.sqrt(var)*self.var_coef
            else:
                latent = mu
            batch[self.output] = latent

class AttentionEncoder(nn.Module):
    name = 'attention_encoder'
    def __init__(self, logger, sizes, input, output_memory, output_padding_mask, pe_dropout, layer, 
            num_layers, max_len, pad_token, voc_size, init, modes=None, **kwargs):
        """
        logger: logging.Logger
        sizes: {param_name(str): size([int])}
        input: str
        output_memory: str
        output_padding_mask: str
        pe_dropout: float
        layer: dict
            input for SelfAttentionLayer
        num_layers: int
        init: dict
        """
        super().__init__()
        check_leftargs(self, logger, kwargs)
        self.modes = modes
        self.input = input
        self.output_memory = output_memory
        self.output_padding_mask = output_padding_mask
        d_model = layer.mha.embed_dim
        # embedding
        self.embedding = nn.Embedding(num_embeddings=voc_size, embedding_dim=d_model, padding_idx=pad_token)
        self.pe = PositionalEncoding(d_model=d_model, dropout=pe_dropout, max_len=max_len)
        
        # encoder
        encoder_layer = SelfAttentionLayer(logger=logger, **layer)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer,
            num_layers=num_layers, norm=None)
        self.pad_token = pad_token
        sizes[self.output_memory] = ['length', 'batch_size', d_model]
        sizes[self.output_padding_mask] = ['batch_size', 'length']

        # weight init
        for layer in self.encoder.layers:
            for name, param in layer.state_dict().items():
                init_config2func(init[name])(param)
    def forward(self, batch, mode):
        if self.modes is not None and mode not in self.modes: 
            return
        input_seq = batch[self.input]
        src_padding_mask = (input_seq == self.pad_token)
        input_seq = input_seq.transpose(0, 1).contiguous()
        input_emb = self.pe(self.embedding(input_seq))
        memory = self.encoder(input_emb, mask=None, src_key_padding_mask=src_padding_mask)
        batch[self.output_memory] = memory
        batch[self.output_padding_mask] = src_padding_mask

"""
このクラスは最小限の機能のみで, あくまで各クラスを優先する (memoryありなど多いため)
・dec2probaをforward, cell_forwardに移行した。
"""
class LatentSequenceDecoder(nn.Module):
    def __init__(self, logger, output, start_token, end_token, voc_size, modes=None, **kwargs):
        super().__init__()
        check_leftargs(self, logger, kwargs)
        self.modes = modes
        self.output = output
        self._device_param = nn.Parameter(torch.zeros((0, )))
        self.start_token = start_token
        self.end_token = end_token
        self.voc_size = voc_size

    def greedy_decode(self, batch, decode_len, output='greedy'):
        state = self.initial_state(batch)
        batch_size = self.get_batch_size(batch)
        cur_input = torch.full(size=(batch_size, ), fill_value=self.start_token,
            dtype=torch.long, device=self.device)
        decodeds = []
        for position in range(decode_len):
            cur_proba, state = self.cell_forward(state, cur_input, batch, position)
            cur_input = torch.argmax(cur_proba, dim=-1).squeeze(0)
            decodeds.append(cur_input)
        batch[output] = torch.stack(decodeds, dim=1)
    
    def distribution_decode(self, batch, decode_len, generator=None, output='distribution'):
        state = self.initial_state(batch)
        batch_size = self.get_batch_size(batch)
        cur_input = torch.full(size=(batch_size, ), fill_value=self.start_token,
            dtype=torch.long, device=self.device)
        decodeds = []
        for position in range(decode_len):
            cur_proba, state = self.cell_forward(state, cur_input, batch, position)
            cur_proba = torch.softmax(cur_proba.squeeze(0), dim=-1) # [batch_size, n_voc]
            cur_input = torch.multinomial(cur_proba, num_samples=1, generator=generator)[:,0]
            decodeds.append(cur_input)
        batch[output] = torch.stack(decodeds, dim=1)
        
    def beam_search(self, batch, decode_len, beam_size, out_beam_size=None, output='beam'):
        batch_size = self.get_batch_size(batch)
        device = self.device
        beam_batch = self.expand_beam(batch, beam_size)
        beam_batch_size = batch_size*beam_size
        state = self.initial_state(beam_batch)
        proba = torch.zeros(size=(batch_size, beam_size), dtype=torch.float, device=device)
        decodeds = torch.zeros((0, batch_size, beam_size), dtype=torch.long, device=device, ) # [length, batch_size, beam_width]
        is_ended = torch.full((batch_size, beam_size), fill_value=False,
            dtype=torch.bool, device=device)
        cur_input = torch.full(size=(beam_batch_size, ), fill_value=self.start_token,
            dtype=torch.long, device=device)
        for position in range(decode_len):

            cur_proba, state = self.cell_forward(state, cur_input, beam_batch, position)
            state = self.zip_state(state).view(batch_size, beam_size, -1) # [batch_size, beam_size, state_size]
            cur_proba = cur_proba.view(batch_size, beam_size, -1) # [batch_size, beam_size, voc_size]
            cur_proba = cur_proba - torch.log(torch.exp(cur_proba).sum(dim=-1, keepdim=True))
            cur_proba[is_ended] = -torch.inf
            cur_proba[:,:,self.end_token][is_ended] = 0
            proba = proba.unsqueeze(2)+cur_proba
            if position == 0:
                proba = proba[:, 0].contiguous()
            proba = proba.view(batch_size, -1) # [batch_size, beam_width*voc_size]
            proba, topk_beam_voc = proba.topk(k=beam_size, dim=-1) # [batch_size, beam_width]
            topk_voc = topk_beam_voc % self.voc_size
            topk_beam = torch.div(topk_beam_voc, self.voc_size, rounding_mode='floor')
            decodeds = torch.cat([decodeds.gather(dim=-1, index=topk_beam.expand(*decodeds.shape)),
                topk_voc.unsqueeze(0)])
            cur_input = topk_voc.view(-1)
            is_ended[topk_voc == self.end_token] = True
            state = state.gather(dim=-2, index=topk_beam.unsqueeze(-1).expand(*state.shape)).reshape(beam_batch_size, -1)
            state = self.unzip_state(state)
        decodeds = decodeds.transpose(0, 1)
        if out_beam_size is None:
            batch[output] = decodeds[:,:,0]
        else:
            batch[output] = decodeds[:,:,:out_beam_size]
    def forward(self, batch, mode):
        if self.modes is not None and mode not in self.modes:
            return
        self.train_forward(batch)
        if mode == 'evaluate':
            batch['forced'] = torch.argmax(batch[self.output], axis=-1)
            decode_len = batch['decode_len'] if 'decode_len' in batch \
                else self.get_decode_len(batch)
            batch = self.greedy_decode(batch, decode_len)
    def train_forward(self, batch):
        raise NotImplementedError
    def initial_state(self, state):
        raise NotImplementedError
    def zip_state(self, state):
        raise NotImplementedError
    def unzip_state(self, state):
        raise NotImplementedError
    def expand_beam(self, batch, beam_size):
        raise NotImplementedError
    def cell_forward(self, state, cur_input, batch, position):
        raise NotImplementedError
    def get_batch_size(self, batch):
        raise NotImplementedError
    def get_decode_len(self, batch):
        raise NotImplementedError
    @property
    def device(self):
        return self._device_param.device

class AttentionDecoder(LatentSequenceDecoder):
    name = 'attention_decoder'
    def __init__(self, logger, sizes, input, output, pad_token, start_token, end_token,
            voc_size, pe_dropout, layer, num_layers, init, dec2proba, max_len, load_square_mask='keep', **kwargs):
        """
        input:
            latent: str
            seq: str
        
        """
        super().__init__(logger=logger, output=output, start_token=start_token, end_token=end_token,
            voc_size=voc_size, **kwargs)
        square_mask = nn.Transformer.generate_square_subsequent_mask(max_len)
        self.register_buffer('square_subsequent_mask', square_mask)
        d_model = layer.mha.embed_dim
        self.input_latent = input['latent']
        self.input_seq = input['seq']
        self.d_model = d_model
        self._device_param = nn.Parameter(torch.zeros((0, )))
        sizes[self.output] = ['batch_size', 'length', self.voc_size]

        # embedding
        self.embedding = nn.Embedding(num_embeddings=voc_size,
            embedding_dim=d_model, padding_idx=pad_token)
        self.pe = PositionalEncoding(d_model=d_model, dropout=pe_dropout, max_len=max_len)
        # decoder
        decoder_layer = SelfAttentionLayer(logger=logger, **layer)
        self.decoder = nn.TransformerEncoder(encoder_layer=decoder_layer, num_layers=num_layers)
        self.dec2proba = Tunnel(logger=logger, layers=dec2proba, input_size=['batch_size', 'length', self.d_model])
        
        # weight init
        for layer in self.decoder.layers:
            for param_name in init:
                init_config2func(init[param_name])(layer.state_dict()[param_name])

        # define operation in load_state_dict
        if load_square_mask == 'keep':
            def pre_hook(model, state_dict, prefix, local_metadata, strict,
                    missing_keys, upexpected_keys, error_msgs):
                state_dict[prefix+"square_subsequent_mask"] = model.square_subsequent_mask
        elif load_square_mask == 'load':
            def pre_hook(model, state_dict, prefix, local_metadata, strict,
                    missing_keys, upexpected_keys, error_msgs):
                if prefix+"square_subsequent_mask" in state_dict:
                    model.register_buffer('square_subsequent_mask', state_dict[prefix+"square_subsequent_mask"])
                else:
                    state_dict[prefix+"square_subsequent_mask"] = model.square_subsequent_mask
        elif load_square_mask == 'larger':
            def pre_hook(model, state_dict, prefix, local_metadata, strict,
                    missing_keys, upexpected_keys, error_msgs):
                if prefix+"square_subsequent_mask" in state_dict and \
                    len(model.square_subsequent_mask) < len(state_dict[prefix+"square_subsequent_mask"]):
                    model.register_buffer('square_subsequent_mask', state_dict[prefix+"square_subsequent_mask"])
                else:
                    state_dict[prefix+"square_subsequent_mask"] = model.square_subsequent_mask
        else:
            raise ValueError(f"Unsupported type of config.load_square_mask: {load_square_mask}")
        self._register_load_state_dict_pre_hook(pre_hook, with_module=True)
    def initial_state(self, batch):
        return [torch.zeros(size=(0, len(batch[self.input_latent]), self.d_model), dtype=torch.float,
            device=self.device) for i_layer in range(self.decoder.num_layers)]
    def zip_state(self, state):
        return torch.cat(state, dim=-1).transpose(0, 1).contiguous().reshape(state[0].shape[1], -1)
    def unzip_state(self, state):
        batch_size, state_size = state.shape # [batch_size, length*num_layers*d_model]
        length = int(state_size / (self.decoder.num_layers*self.d_model))
        state = state.view((batch_size, length, self.decoder.num_layers, self.d_model))
        state = state.permute(2, 1, 0, 3)
        return list(state)
    def expand_beam(self, batch, beam_size):
        beam_batch = {
            self.input_latent: torch.repeat_interleave(batch[self.input_latent], beam_size, dim=0)
        }
        return beam_batch
    def get_batch_size(self, batch):
        return batch[self.input_seq].shape[0]
    def get_decode_len(self, batch):
        return batch[self.input_seq].shape[1]

    def train_forward(self, batch):
        latent = batch[self.input_latent]
        input_seq = batch[self.input_seq]
        batch_size, max_len = input_seq.shape
        input_seq = input_seq.transpose(0, 1).contiguous()
        input_emb = self.pe(self.embedding(input_seq)) # [length, batch_size, d_model]
        input_emb += latent # [length, batch_size, d_model]
        
        input_mask = self.square_subsequent_mask[:max_len, :max_len]
        output = self.decoder(src=input_emb, mask=input_mask, src_key_padding_mask=None) # [length, batch_size, d_model]
        output = self.dec2proba(output)
        batch[self.output] = output.transpose(0, 1).contiguous()

    def cell_forward(self, state, cur_input, batch, position):
        cur_input_emb = self.embedding(cur_input)
        cur_input_emb = self.pe.encode_position(cur_input_emb, position)+batch[self.input_latent]
        attn_mask = self.square_subsequent_mask[position+1:position+2, :position+1]
        cur_input_emb = cur_input_emb.unsqueeze(0)
        cur_output = cur_input_emb
        for i_layer, layer in enumerate(self.decoder.layers):
            prev_y = state[i_layer]
            cur_y = layer.norm1(cur_output)
            y = torch.cat([prev_y, cur_y], dim=0)
            state[i_layer] = y
            cur_attn = layer.self_attn(cur_y, y, y, attn_mask=attn_mask,
                        key_padding_mask=None, need_weights=False)[0]
            cur_output = cur_output + layer.dropout1(cur_attn)
            cur_output = cur_output + layer._ff_block(layer.norm2(cur_output))
        cur_proba = self.dec2proba(cur_output)
        return cur_proba, state
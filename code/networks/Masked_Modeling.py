import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

class PositionEmbeddingLearned1D(nn.Module):
    def __init__(self, dic_len: int, d_model: int):
        super().__init__()
        # Create an embedding layer for seq_length positions
        self.position_embeddings = nn.Embedding(dic_len, d_model)

    def forward(self, x):
        # x is a tensor of shape [seq_length, batch_size, embedding_dim]
        # Generate a sequence of position indices (0 to seq_length-1)
        position_ids = torch.arange(x.shape[0], dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(1).expand(-1, x.shape[1])

        # Retrieve the position embeddings
        position_embeddings = self.position_embeddings(position_ids)

        # Add the position embeddings to the input embeddings
        return position_embeddings

class Masked_Modeling(nn.Module):
    def __init__(self):
        super(Masked_Modeling, self).__init__()
        # input is 10*N*512
        dim = 448
        self.encoder = TransformerEncoder(TransformerEncoderLayer(d_model=dim, nhead=8, dim_feedforward=1024), num_layers=2)
        # self.positional_embedding = nn.Embedding(10, 512)
        self.positional_embedding =  PositionEmbeddingLearned1D(dic_len=10, d_model=dim)
        self.xy_projection = nn.Linear(2, dim)
    
    def random_mask(self, x, mask_num = 1):
        mask = torch.zeros_like(x).cuda()
        mask_idx = torch.randperm(x.size(0))[:mask_num]
        mask[mask_idx, :, :] = 1
        x = x * (1 - mask)
        return x, mask, mask_idx
    
    def forward(self, x, return_mask=False, xy_embed=None, mask_num=4):
        if xy_embed is None:
            pos = self.positional_embedding(x)  
        else:
            pos = self.xy_projection(xy_embed).transpose(0,1)
            # print(pos.shape, x.shape)
        x_, mask, mask_idx = self.random_mask(x, mask_num=mask_num)
        output = self.encoder(x_, pos=pos)
        masked_gt = x[mask_idx,:,:]
        masked_pred = output[mask_idx,:,:]
        if return_mask:
            return masked_pred, masked_gt, output, mask
        else:
            return masked_pred, masked_gt, output
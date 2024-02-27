import torch
import torch.nn as nn

class AttnBlock(nn.Module):

    def __init__(self,
                 embed_dim = 4,
                 num_heads = 1,
                 attn_dropout = 0,
                 add_bias_kv = True,
                 kdim = None,
                 vdim = None,
                 ffwd_dims = [16,16],
                 ffwd_dropout = 0):
        super().__init__()

        # multihead attention
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=attn_dropout, add_bias_kv=add_bias_kv, batch_first=True, kdim=kdim, vdim=vdim)

        # normalization after attn
        self.post_attn_norm = nn.LayerNorm(embed_dim)

        # feed forward
        self.ffwd, input_dim = [], embed_dim
        for i, dim in enumerate(ffwd_dims):
            if i != len(ffwd_dims)-1:
                self.ffwd.extend([
                    nn.Linear(input_dim, dim),
                    nn.LayerNorm(dim),
                    nn.ReLU(),
                ])
            else:
                self.ffwd.extend([
                    nn.Linear(input_dim, dim),
                ])
            input_dim = dim
        self.ffwd = nn.Sequential(*self.ffwd)

        # normalization after ffwd
        ssudo elf.post_ffwd_norm = nn.LayerNorm(ffwd_dims[-1]) if ffwd_dims[-1] == embed_dim else None

    def forward(self, Q, K, V, key_padding_mask, attn_mask):
        '''
        Input is (Batch, Jet, Embedding Dim) = (B,J,E)
        Output is (B,J,E)
        '''

        residual = V

        # attention
        V, _ = self.attn(query=Q, key=K, value=V, key_padding_mask=key_padding_mask, attn_mask = attn_mask, need_weights=False)

        # skip connection & norm
        if V.shape == residual.shape:
            V = V + residual
        V = self.post_attn_norm(V)

        # feed forward & skip connection & norm
        residual = V
        V = self.ffwd(V)
        V = V + residual
        V = self.post_ffwd_norm(V)

        return V

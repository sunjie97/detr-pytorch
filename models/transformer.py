import torch
import torch.nn as nn 


class TransformerEncoderLayer(nn.Module):

    def __init__(self, embed_dim, num_heads, hidden_dim=2048, drop=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=drop)

        self.mlp = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, hidden_dim),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, embed_dim)
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.drop1 = nn.Dropout(drop)
        self.drop2 = nn.Dropout(drop)

    @staticmethod
    def with_pos_embed(x, pos_embed=None):
        return x if pos_embed is None else x + pos_embed 

    def forward(self, x, key_padding_mask=None, pos_embed=None):
        """
        Args:
            x: [hw, bs, embed_dim]
            key_padding_mask: [bs, hw]
            pos: [hw, bs, 256]
        """
        q = k = self.with_pos_embed(x, pos_embed)
        attn = self.attn(q, k, value=x, key_padding_mask=key_padding_mask)[0]
        x = self.norm1(x + self.drop1(attn))

        attn = self.mlp(x)
        x = self.norm2(x + self.drop2(attn))

        return x 


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers):
        super().__init__()

        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])

    def forward(self, x, key_padding_mask=None, pos_embed=None):
        """
        Args:
            x: [hw, bs, embed_dim]
            key_padding_mask: [bs, hw]
            pos: [hw, bs, 256]
        """
        for layer in self.layers:
            x = layer(x, key_padding_mask, pos_embed)

        return x 


class TransformerDecoderLayer(nn.Module):

    def __init__(self, embed_dim, num_heads, hidden_dim=2048, drop=0.1):
        super().__init__()

        self.attn1 = nn.MultiheadAttention(embed_dim, num_heads, dropout=drop)
        self.attn2 = nn.MultiheadAttention(embed_dim, num_heads, dropout=drop)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, embed_dim)
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.drop1 = nn.Dropout(drop)
        self.drop2 = nn.Dropout(drop)
        self.drop3 = nn.Dropout(drop)

    @staticmethod
    def with_pos_embed(x, pos_embed=None):
        return x if pos_embed is None else x + pos_embed 

    def forward(self, tgt, memory, memory_key_padding_mask=None, query_embed=None, pos_embed=None):
        """
        Args:
            tgt:                      ??????shape???query_embed?????????????????????
            memory:                   encoder?????????
            memory_key_padding_mask:  ???????????? encoder ????????? <pad> ???????????????
            query_embed:              nn.Embedding ??????????????? shape ??? [num_queries, embed_dim] ?????????
            pos_embed:                ????????????
        """
        q = k = self.with_pos_embed(tgt, query_embed)
        attn = self.attn1(q, k, value=tgt)[0]
        tgt = self.norm1(tgt + self.drop1(attn))

        q = self.with_pos_embed(tgt, query_embed)
        k = self.with_pos_embed(memory, pos_embed)
        attn = self.attn2(q, k, value=memory, key_padding_mask=memory_key_padding_mask)[0]
        tgt = self.norm2(tgt + self.drop2(attn))

        attn = self.mlp(tgt)
        tgt = self.norm3(tgt + self.drop3(attn))

        return tgt 


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers):
        super().__init__()

        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])

    def forward(self, tgt, memory, memory_key_padding_mask=None, query_embed=None, pos_embed=None):
        outs = []

        for layer in self.layers:
            tgt = layer(tgt, memory, memory_key_padding_mask, query_embed, pos_embed)
            outs.append(tgt)

        return torch.stack(outs)


class Transformer(nn.Module):

    def __init__(self, embed_dim=512, num_heads=8, num_encoder_layer=6, num_decoder_layer=6, hidden_dim=2048, drop=0.1):
        super().__init__()

        self.embed_dim = embed_dim

        encoder_layer = TransformerEncoderLayer(embed_dim, num_heads, hidden_dim, drop)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layer)

        decoder_layer = TransformerDecoderLayer(embed_dim, num_heads, hidden_dim, drop)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layer)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, key_padding_mask, query_embed, pos_embed):
        """
        Args:
            x: [bs, embed_dim, 7, 7]
            mask: [2, 7, 7]
            query_embed: [num_queries, embed_dim]
            pos_embed: [2, 256, 7, 7]
        """
        b, c, h, w = x.shape 
        x = x.flatten(2).permute(2, 0, 1)  # [hw, bs, embed_dim]
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)  # [hw, bs, embed_dim]
        query_embed = query_embed.unsqueeze(1).repeat(1, b, 1)  # [num_queries, bs, embed_dim]
        key_padding_mask = key_padding_mask.flatten(1)  # [bs, hw]

        tgt = torch.zeros_like(query_embed)  # [num_queries, bs, embed_dim]

        memory = self.encoder(x, key_padding_mask=key_padding_mask, pos_embed=pos_embed)
        outs = self.decoder(tgt, memory, memory_key_padding_mask=key_padding_mask, query_embed=query_embed, pos_embed=pos_embed)

        # [num_layers, b, num_queries, embed_dim]  [b, c, h, w]
        return outs.transpose(1, 2), memory.permute(1, 2, 0).view(b, c, h, w)


def build_transformer(args):

    return Transformer(
        embed_dim=args.embed_dim,
        num_heads=args.nheads,
        num_encoder_layer=args.encoder_layers,
        num_decoder_layer=args.decoder_layers,
        hidden_dim=args.hidden_dim,
        drop=args.dropout
    )

"""
attn_mask: ?????????????????????????????????
key_padding_mask: ??????token?????????????????????????????? <pad> ??????????????????????????????self attention????????????key_padding_mask????????????????????????????????????
"""

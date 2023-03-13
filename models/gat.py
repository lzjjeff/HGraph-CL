import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
    reference: https://github.com/xptree/DeepInf
    """
    def __init__(self, att_head, in_dim, out_dim, dp_gnn, leaky_alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dp_gnn = dp_gnn

        self.att_head = att_head
        self.W = nn.Parameter(torch.Tensor(self.att_head, self.in_dim, self.out_dim))
        self.b = nn.Parameter(torch.Tensor(self.out_dim))

        self.w_src = nn.Parameter(torch.Tensor(self.att_head, self.out_dim, 1))
        self.w_dst = nn.Parameter(torch.Tensor(self.att_head, self.out_dim, 1))
        self.leaky_alpha = leaky_alpha
        self.init_gnn_param()

        assert self.in_dim == self.out_dim*self.att_head
        self.H = nn.Linear(self.in_dim, self.in_dim)
        init.xavier_normal_(self.H.weight)
#         self.layernorm = nn.LayerNorm(out_dim)

    def init_gnn_param(self):
        init.xavier_uniform_(self.W.data)
        init.zeros_(self.b.data)
        init.xavier_uniform_(self.w_src.data)
        init.xavier_uniform_(self.w_dst.data)

    def forward(self, feat_in, adj=None):
        batch, N, in_dim = feat_in.size()
        assert in_dim == self.in_dim

        feat_in_ = feat_in.unsqueeze(1)
        h = torch.matmul(feat_in_, self.W)

        attn_src = torch.matmul(torch.tanh(h), self.w_src)
        attn_dst = torch.matmul(torch.tanh(h), self.w_dst)
        attn = attn_src.expand(-1, -1, -1, N) + attn_dst.expand(-1, -1, -1, N).permute(0, 1, 3, 2)
        attn = F.leaky_relu(attn, self.leaky_alpha, inplace=True)

        mask = 1 - adj.unsqueeze(1)
        attn.data.masked_fill_(mask.bool(), -999)

        attn = F.softmax(attn, dim=-1)
        feat_out = torch.matmul(attn, h) + self.b

        feat_out = feat_out.transpose(1, 2).contiguous().view(batch, N, -1)
        feat_out = F.elu(feat_out)

        gate = torch.sigmoid(self.H(feat_in))
        feat_out = gate * feat_out + (1 - gate) * feat_in
#         feat_out = self.layernorm(feat_out + feat_in)

        feat_out = F.dropout(feat_out, self.dp_gnn, training=self.training)

        return feat_out, attn

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_dim) + ' -> ' + str(self.out_dim*self.att_head) + ')'


class GAT(nn.Module):
    def __init__(self, input_dim=128, gnn_dim=32, num_layers=2, num_heads=4, dropout=0.1, leaky_alpha=0.2):
        super(GAT, self).__init__()
        self.input_dim = input_dim
        self.gnn_dim = gnn_dim

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.gnn_layer_stack = nn.ModuleList()
        for i in range(self.num_layers):
            in_dim = self.gnn_dim * self.num_heads if i != 0 else self.input_dim
            self.gnn_layer_stack.append(
                GraphAttentionLayer(self.num_heads, in_dim, self.gnn_dim, dropout, leaky_alpha)
            )

    def forward(self, seq_h, adj):
        batch, max_len, _ = seq_h.size()

        for i, gnn_layer in enumerate(self.gnn_layer_stack):
            seq_h, attn = gnn_layer(seq_h, adj)

        return seq_h, attn


import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionAggregator(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionAggregator, self).__init__()
        self.attn = nn.Linear(hidden_size, 1, bias=False)

    def get_attn(self, reps, mask=None):
        attn_scores = self.attn(reps).squeeze(2)
        if mask is not None:
            attn_scores = attn_scores + mask
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(2)  # (batch_size, len, 1)
        attn_out = torch.sum(reps * attn_weights, dim=1)  # (batch_size, hidden_dim)

        return attn_out, attn_weights

    def forward(self, reps, mask=None):
        attn_out, attn_weights = self.get_attn(reps, mask)

        return attn_out, attn_weights  # (batch_size, hidden_dim), (batch_size, len, 1)


class MeanAggregator(nn.Module):
    def __init__(self):
        super(MeanAggregator, self).__init__()

    def forward(self, reps):
        return reps.mean(1)


class MaxAggregator(nn.Module):
    def __init__(self):
        super(MaxAggregator, self).__init__()

    def forward(self, reps):
        return torch.max(reps, dim=1)
    

class MultimodalGraphReadout(nn.Module):
    def __init__(self, m_dim, readout_t, readout_v, readout_a):
        super(MultimodalGraphReadout, self).__init__()
        self.readout_t = readout_t
        self.readout_v = readout_v
        self.readout_a = readout_a
        self.project_m = nn.Linear(m_dim, m_dim)

    def forward(self, hs_gnn, mask):
        hs_t_, hs_v_, hs_a_ = torch.split(hs_gnn, hs_gnn.size(1)//3, dim=1)
        reps_t_, _ = self.readout_t(hs_t_, mask)
        reps_v_, _ = self.readout_v(hs_v_, mask)
        reps_a_, _ = self.readout_a(hs_a_, mask)
        reps_m = F.relu(self.project_m(torch.cat([reps_t_, reps_v_, reps_a_], dim=-1)))
        
        return reps_m
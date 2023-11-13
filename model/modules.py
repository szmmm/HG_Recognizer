import torch.nn as nn
import torch
import math
import torch.nn.functional as F
import numpy as np


class PositionalEncoding(nn.Module):
    """Implement the PE function."""

    def __init__(self, ft_size, time_len, joint_num, domain):
        super(PositionalEncoding, self).__init__()
        self.joint_num = joint_num
        self.time_len = time_len
        self.domain = domain

        if domain == "temporal" or domain == "mask_t":
            # temporal embedding
            pos_list = list(range(self.joint_num * self.time_len))

        elif domain == "spatial" or domain == "mask_s":
            # spatial embedding
            pos_list = []
            for t in range(self.time_len):
                for j_id in range(self.joint_num):
                    pos_list.append(j_id)
        else:
            raise Exception("Attention Domain Not Supported")

        position = torch.from_numpy(np.array(pos_list)).unsqueeze(1).float()

        # Compute the positional encodings in log space.
        pe = torch.zeros(self.time_len * self.joint_num, ft_size)

        div_term = torch.exp(torch.arange(0, ft_size, 2).float() *
                             -(math.log(10000.0) / ft_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


class LayerNorm(nn.Module):
    """Construct a layer norm module (See citation for details)."""

    def __init__(self, ft_dim, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(ft_dim))
        self.b_2 = nn.Parameter(torch.zeros(ft_dim))
        self.eps = eps

    def forward(self, x):
        # [batch, time, ft_dim]
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class MultiHeadedAttention(nn.Module):
    def __init__(self, h_num, h_dim, input_dim, frame_num, joint_num, dp_rate, domain):
        """Take in model size and number of heads."""
        super(MultiHeadedAttention, self).__init__()
        self.h_dim = h_dim  # head dimension
        self.h_num = h_num  # head num
        self.attn = None  # calculate_att weight
        self.domain = domain  # spatial or temporal
        self.frame_num = frame_num
        self.joint_num = joint_num

        self.register_buffer('t_att', self.get_att_weights()[0])
        self.register_buffer('s_att', self.get_att_weights()[1])

        self.key_map = nn.Sequential(
            nn.Linear(input_dim, self.h_dim * self.h_num),
            nn.Dropout(dp_rate),
        )

        self.query_map = nn.Sequential(
            nn.Linear(input_dim, self.h_dim * self.h_num),
            nn.Dropout(dp_rate),
        )

        self.value_map = nn.Sequential(
            nn.Linear(input_dim, self.h_dim * self.h_num),
            nn.ReLU(),
            nn.Dropout(dp_rate),
        )

    def get_att_weights(self):
        t_att = torch.ones(self.frame_num * self.joint_num, self.frame_num * self.joint_num)
        filtered_area = torch.zeros(self.joint_num, self.joint_num)

        for i in range(self.frame_num):
            row_begin = i * self.joint_num
            column_begin = row_begin
            row_num = self.joint_num
            column_num = row_num

            t_att[row_begin: row_begin + row_num, column_begin: column_begin + column_num] *= filtered_area  # Sec 3.4

        identity = torch.eye(self.frame_num * self.joint_num)
        s_att = (1 - t_att)
        t_att = (t_att + identity)
        return t_att, s_att

    def attention(self, query, key, value):
        """Compute Scaled Dot Product Attention"""
        # [batch, time, ft_dim]
        h_dim = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(h_dim)
        if self.domain is not None:
            # section 3.4 spatial temporal mask operation
            if self.domain == "temporal":
                scores *= self.t_att  # set weight to 0 to block gradient
                scores += (1 - self.t_att) * (-9e15)  # set weight to -inf to remove effect in Softmax
            elif self.domain == "spatial":
                scores *= self.s_att  # set weight to 0 to block gradient
                scores += (1 - self.s_att) * (-9e15)  # set weight to -inf to remove effect in Softmax

        # apply weight_mask to block information passage between inner-joint

        p_attn = F.softmax(scores, dim=-1)

        return torch.matmul(p_attn, value), p_attn

    def forward(self, x):
        """Implements Figure 2"""
        batch_num = x.size(0)  # [batch, t, dim]
        # 1) Do all the linear projections in batch from input_dim => h x h_dim

        query = self.query_map(x).view(batch_num, -1, self.h_num, self.h_dim).transpose(1, 2)
        key = self.key_map(x).view(batch_num, -1, self.h_num, self.h_dim).transpose(1, 2)
        value = self.value_map(x).view(batch_num, -1, self.h_num, self.h_dim).transpose(1, 2)

        # 2) Apply attention on all the projected vectors in batch
        x, self.attn = self.attention(query, key, value)  # [batch, h_num, T, h_dim]

        # 3) Concatenate using a view
        x = x.transpose(1, 2).contiguous() \
            .view(batch_num, -1, self.h_dim * self.h_num)  # [batch, T, h_dim * h_num]

        return x, self.attn


class STALayer(nn.Module):
    """Encoder is made up of self-attn and feed forward (defined below)
        input_size : the dim of input
        output_size: the dim of output
        h_num: att head num
        h_dim: dim of each att head
        frame_num: input frame number
        domain: do att on spatial domain or temporal domain
    """

    def __init__(self, input_size, output_size, h_num, h_dim, dp_rate, frame_num, joint_num, domain):
        super(STALayer, self).__init__()

        self.pe = PositionalEncoding(input_size, frame_num, joint_num, domain)
        # h_num, h_dim, input_dim, dp_rate,domain
        self.attn = MultiHeadedAttention(h_num, h_dim, input_size, frame_num, joint_num, dp_rate, domain)

        self.ft_map = nn.Sequential(
            nn.Linear(h_num * h_dim, output_size),
            nn.ReLU(),
            LayerNorm(output_size),
            nn.Dropout(dp_rate),
        )
        self.attn_wt = None
        self.init_parameters()

    def forward(self, x):
        x = self.pe(x)  # add PE
        x, self.attn_wt = self.attn(x)  # pass attention model
        x = self.ft_map(x)  # apply a linear layer like Transformer
        return x, self.attn_wt

    def init_parameters(self):
        model_list = [self.attn, self.ft_map]
        for model in model_list:
            for p in model.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

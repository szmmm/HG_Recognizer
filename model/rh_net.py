from model.modules import *
import seaborn as sn
import os
import torch.nn as nn


class STADualNet(nn.Module):
    """ Spatial-Temporal Attention Network with Two Heads

    input_size : the dim of input
    output_size: the dim of output
    h_num: att head num
    h_dim: dim of each att head
    time_len: input frame number
    domain: do att on spatial domain or temporal domain
    plot_att: set True to save attention graphs to model dir

    """

    def __init__(self, num_classes, num_states, joint_num, input_dim, window_size, dp_rate, plot_att=False):
        super().__init__()

        h_dim = 64
        h_num = 8
        self.window_size = window_size
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.num_state = num_states
        self.joints = joint_num
        self.plot_att = plot_att

        self.input_map = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            LayerNorm(256),
            nn.Dropout(0.2),
        )

        self.s_att_w = None
        self.t_att_w = None
        self.att_3 = None
        # input_size, h_num, h_dim, dp_rate, time_len, domain
        self.s_att = STALayer(input_size=256, output_size=256, h_num=h_num, h_dim=h_dim, dp_rate=dp_rate,
                              frame_num=self.window_size, joint_num=self.joints, domain="spatial")
        self.t_att = STALayer(input_size=256, output_size=256, h_num=h_num, h_dim=h_dim, dp_rate=dp_rate,
                              frame_num=self.window_size, joint_num=self.joints, domain="temporal")

        # second spatial self-attention layer for gesture state classification
        self.t_att_2 = STALayer(input_size=256, output_size=256, h_num=h_num, h_dim=h_dim, dp_rate=dp_rate,
                                frame_num=self.window_size, joint_num=self.joints, domain="temporal")

        self.stt = nn.Linear(256 + num_classes, self.num_state)

        self.cls = nn.Linear(256, self.num_classes)

    def forward(self, x):
        # input shape: [batch_size, time_len, joint_num, dimension]

        time_len = x.shape[1]
        joint_num = x.shape[2]
        dimension = x.shape[-1]

        # reshape x into (space * time) vectors
        x = x.reshape(-1, time_len * joint_num, dimension)

        # input map
        x = self.input_map(x)

        # spatial
        x, self.s_att_w = self.s_att(x)

        # temporal
        x1, self.t_att_w = self.t_att(x)

        # x2 = x1
        # average over time x joint for classification
        x1 = torch.mean(x1, dim=1)
        cls = self.cls(x1)

        x2, self.att_3 = self.t_att_2(x)

        # average over joints for state prediction
        x2 = x2.reshape(-1, time_len, joint_num, x2.shape[-1])  # reshaping to compute MCE Loss
        x2 = torch.mean(x2, dim=2)  # [batch_size, time_len, dimension]

        # concat class prediction to final input for state prediction
        cls_expand = cls.unsqueeze(1).expand(-1, x2.shape[1], -1)
        x2 = torch.cat((x2, cls_expand), dim=2)

        x2 = self.stt(x2)
        state = x2.transpose(1, 2)  # reshape output for loss calculation

        return cls, state

    def get_window_size(self):
        return self.window_size

    def get_class_num(self):
        return self.num_classes

    def get_input_dim(self):
        return self.input_dim

    def get_joint_num(self):
        return self.joints

    def save_att_to(self, path):
        s_att_dir = f'{path}/s_att'
        # spatial att weights
        for i in range(self.s_att_w.shape[0]):
            for j in range(self.s_att_w.shape[1]):
                fig = sn.heatmap(self.s_att_w[i, j].cpu())
                path = f'{s_att_dir}/b{i}_h{j}.png'
                if not os.path.exists(path):
                    os.makedirs(path)
                fig.savefig(path)

        # temporal att weights
        t_att_dir = f'{path}/t_att'
        for i in range(self.t_att_w.shape[0]):
            for j in range(self.t_att_w.shape[1]):
                fig = sn.heatmap(self.s_att_w[i, j])
                path = f'{t_att_dir}/b{i}_h{j}.png'
                if not os.path.exists(path):
                    os.makedirs(path)
                fig.savefig(path)

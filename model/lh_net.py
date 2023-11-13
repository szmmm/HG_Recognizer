from model.modules import *
import seaborn as sn
import os
import torch.nn as nn


class STANet(nn.Module):
    """ Spatial-Temporal Attention Network

    input_size : the dim of input
    output_size: the dim of output
    h_num: att head num
    h_dim: dim of each att head
    time_len: input frame number
    domain: do att on spatial domain or temporal domain
    plot_att: set True to save attention graphs to model dir

    """

    def __init__(self, num_classes, joint_num, input_dim, window_size, dp_rate, plot_att=False):
        super().__init__()

        self.h_dim = 32
        self.h_num = 8
        self.window_size = window_size
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.joints = joint_num
        self.plot_att = plot_att
        self.dp_rate = dp_rate

        self.input_map = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            LayerNorm(128),
            nn.Dropout(0.2),
        )

        self.s_att_w = None
        self.t_att_w = None
        # input_size, h_num, h_dim, dp_rate, time_len, domain
        self.s_att = STALayer(input_size=128, output_size=128, h_num=self.h_num, h_dim=self.h_dim, dp_rate=self.dp_rate,
                              frame_num=self.window_size, joint_num=self.joints, domain="spatial")
        self.t_att = STALayer(input_size=128, output_size=128, h_num=self.h_num, h_dim=self.h_dim, dp_rate=self.dp_rate,
                              frame_num=self.window_size, joint_num=self.joints, domain="temporal")

        self.cls = nn.Linear(128, self.num_classes)

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
        x, self.t_att_w = self.t_att(x)

        # x2 = x1
        # average over time x joint for classification
        x = torch.mean(x, dim=1)
        cls = self.cls(x)

        return cls

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

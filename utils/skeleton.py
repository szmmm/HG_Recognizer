import os
import sys
import numpy as np
import pickle
import torch
from random import randint, shuffle
from torch.utils.data import DataLoader, Dataset


def selected_frame(data, num_frame):
    """This function uniformly samples data to num_frame frames.
    Not suitable for online recognition model
    """
    frame, joint, dim = data.shape
    if frame == num_frame:
        return data
    interval = frame / num_frame
    uniform_list = [int(i * interval) for i in range(num_frame)]
    return data[uniform_list, :]


def data_aug(data, cls, state):
    def scale(dt):
        for point in range(dt.shape[0]):
            ratio = 0.1
            factor = np.random.uniform(1 - ratio, 1 + ratio)
            dt[point][:][:3] *= factor  # scale each joint
        return dt

    def noise(dt):
        ratio = 0.1
        # select 4 joints
        all_joint = list(range(1, dt.shape[2]))
        shuffle(all_joint)
        selected_joint = all_joint[:5]  # 5 noisy joints

        for point in range(dt.shape[0]):
            for j_id in selected_joint:
                # noise_offset = np.random.uniform(low, high, 7)
                factor = np.random.uniform(1 - ratio, 1 + ratio)
                for t in range(dt.shape[1]):
                    dt[point][t][j_id] *= factor
        return dt

    skeleton_aug = data[:]
    cls_aug = cls[:]
    state_aug = state[:]

    x = np.random.randint(5)
    # skeleton_aug = np.append(skeleton_aug, noise(data[x::5]), axis=0)
    skeleton_aug = np.append(skeleton_aug, scale(data[x::5]), axis=0)
    cls_aug = np.append(cls_aug, cls[x::5], axis=0)
    state_aug = np.append(state_aug, state[x::5], axis=0)

    return skeleton_aug, cls_aug, state_aug
    # return noise(data), cls, state


class SkeletonData(Dataset):
    def __init__(self, data, cls_labels, state_labels, mode="train", use_data_aug=False):
        if use_data_aug and mode == "train":
            data, cls_labels, state_labels = data_aug(data, cls_labels, state_labels)
            self.data = torch.tensor(data, dtype=torch.float)
            self.cls_labels = torch.tensor(cls_labels, dtype=torch.float)
            self.state_labels = torch.tensor(state_labels, dtype=torch.float)
            self.mode = mode
        else:
            self.data = torch.tensor(data, dtype=torch.float)
            self.cls_labels = torch.tensor(cls_labels, dtype=torch.float)
            self.state_labels = torch.tensor(state_labels, dtype=torch.float)
            self.mode = mode

    def __len__(self):
        return len(self.cls_labels)

    def __getitem__(self, item):
        skeleton_data = self.data[item]
        cls_label = int(self.cls_labels[item])
        stt_label = self.state_labels[item]

        return skeleton_data, cls_label, stt_label

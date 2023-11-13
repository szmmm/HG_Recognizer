import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import torch
from server import generate_label
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

classes = ['Scale', 'Duplicate', 'Delete', 'Pen', 'Cube', 'Cylinder', 'Sphere', 'Spray', 'Cut', 'Palette', 'Null']


def plot_heat_map(m):
    assert len(m.shape) == 2

    df = pd.DataFrame(m)
    plt.figure(figsize=(10, 8))
    fig = sn.heatmap(df, annot=True, cmap="viridis", fmt='.2f')
    plt.xlabel('Prediction', fontsize=16)
    plt.ylabel('Target', fontsize=16)
    plt.tick_params(labelsize=16)
    plt.xticks(np.arange(len(classes)) + 0.5, classes)
    plt.yticks(np.arange(len(classes)) + 0.5, classes)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    for text in fig.texts:
        if text.get_text() == '0.00':
            text.set_text('0')
    plt.show()


def plot_confusion_matrix(rh_score, lh_score, rh_label, lh_label, state_cm):
    print(rh_label[:len(lh_label)])
    print(lh_label)
    # print(get_cm(rh_score, rh_label))
    # print(get_cm(lh_score, lh_label))
    print(rh_score[:len(lh_label)])
    print(lh_score)

    lh_score = torch.cat((lh_score, 3 * torch.ones(len(rh_score) - len(lh_score)).cuda()), 0)
    lh_label = torch.cat((lh_label, 3 * torch.ones(len(rh_label) - len(lh_label))), 0)
    print(lh_label)
    print(lh_score.size())
    print(rh_score.size())
    print(lh_label.size())
    print(rh_label.size())

    out = []
    lab = []
    for i in range(len(rh_score)):
        final = generate_label(rh_score[i].cpu(), lh_score[i].cpu())
        out.append(final)
        lab.append(generate_label(rh_label[i], lh_label[i]))

    print(out)
    print(lab)
    m = confusion_matrix(lab, out, normalize='true')
    # m = np.ones(11, 11)

    plot_heat_map(m)
    plot_heat_map(state_cm)

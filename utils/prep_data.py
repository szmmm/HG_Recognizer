import numpy as np
import pandas as pd
import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from utils.skeleton import *


def load_data(data_path, num_frame, use_quaternion=True):
    data_list = []
    label_list = []
    state_list = []
    null_state = 4  # state label for static gestures
    for file in tqdm(sorted(glob.glob(data_path))):
        csv_data = read_data_from_csv(file, use_quaternion)  # split csv into data points with label
        data_len, col = csv_data.shape

        if data_len % num_frame != 0:
            discard_frame = data_len % num_frame
            csv_data = csv_data[discard_frame:]

        csv_data = np.split(csv_data, data_len // num_frame, axis=0)

        for data_point in csv_data:

            if 'scale' in file:
                label_list.append(0)
            elif 'duplicate' in file or 'delete' in file:
                label_list.append(1)
            elif 'pen' in file:
                label_list.append(2)
            elif 'cube' in file:
                label_list.append(3)
            elif 'cylinder' in file:
                label_list.append(4)
            elif 'sphere' in file:
                label_list.append(5)
            elif 'spray' in file:
                label_list.append(6)
            elif 'cut' in file:
                label_list.append(7)
            elif 'palette' in file:
                label_list.append(8)
            elif 'null' in file:
                label_list.append(9)  # null class
            else:
                continue

            if 'state' in list(data_point.columns):
                state_list.append(data_point['state'])
                data_point = data_point.drop('state', axis='columns')
            else:
                state_list.append(null_state*np.ones(data_point.shape[0]))  # gesture with no state labels#

            data_point = np.array(np.split(data_point, 11, axis=1))
            data_point = np.swapaxes(data_point, 0, 1)
            data_list.append(data_point)

    data_array = np.stack(data_list, axis=0)
    label_array = np.array(label_list)
    state_array = np.stack(state_list, axis=0)

    return data_array, label_array, state_array


def load_lh_data(data_path, num_frame, use_quaternion=True):
    data_list = []
    label_list = []
    state_list = []
    null_state = 4  # state label for static gestures
    for file in tqdm(sorted(glob.glob(data_path))):
        csv_data = read_data_from_csv(file, use_quaternion)  # split csv into data points with label
        data_len, col = csv_data.shape

        if data_len <= num_frame:
            continue

        if data_len % num_frame != 0:
            discard_frame = data_len % num_frame
            csv_data = csv_data[discard_frame:]

        csv_data = np.split(csv_data, data_len // num_frame, axis=0)

        for data_point in csv_data:

            if 'scale' in file:
                label_list.append(0)
            elif 'duplicate' in file:
                label_list.append(1)
            elif 'delete' in file:
                label_list.append(2)
            elif 'null' in file:
                label_list.append(3)  # null class
            else:
                continue

            if 'state' in list(data_point.columns):
                state_list.append(data_point['state'])
                data_point = data_point.drop('state', axis='columns')
            else:
                state_list.append(null_state*np.ones(data_point.shape[0]))  # gesture with no state labels#

            data_point = np.array(np.split(data_point, 11, axis=1))
            data_point = np.swapaxes(data_point, 0, 1)
            data_list.append(data_point)

    data_array = np.stack(data_list, axis=0)
    label_array = np.array(label_list)
    state_array = np.stack(state_list, axis=0)

    return data_array, label_array, state_array


def read_data_from_csv(data_path, use_quaternion=True):
    gesture_clip = pd.read_csv(data_path)

    # ignore camera and timestamp, zero all wrs position
    for header in list(gesture_clip.columns):
        if 'wrs' in header:
            if not use_quaternion:
                if 'wrs_qw' not in header and 'wrs_qx' not in header and 'wrs_qy' not in header:
                    gesture_clip = gesture_clip.drop(header, axis='columns')
                gesture_clip['wrs_qx'] = 0
                gesture_clip['wrs_qy'] = 0

                # convert qw to angle
                # assert (gesture_clip['wrs_qw'] < -1).any()
                gesture_clip['wrs_qw'] = np.arccos(np.minimum(1, gesture_clip['wrs_qw'])) * 2
            elif 'q' not in header:
                gesture_clip[header] = 0

        # elif '4' not in header and 'state' not in header and '3' not in header:
        #     gesture_clip = gesture_clip.drop(header, axis='columns')
        elif '4' not in header and '3' not in header and 'state' not in header:
            gesture_clip = gesture_clip.drop(header, axis='columns')

    return gesture_clip


def change_state(data_path):
    gesture_clip = pd.read_csv(data_path)
    transition_1 = 0
    transition_2 = 0

    # ignore camera and timestamp, zero all wrs position
    if "cut" in data_path or "spray" in data_path:
        for header in list(gesture_clip.columns):
            if 'state' in header:
                state_column = gesture_clip[header].to_numpy()

                for i in range(state_column.shape[0] - 1):
                    if state_column[i] == 1 and state_column[i+1] == 0:
                        transition_1 = i
                    elif state_column[i] == 0 and state_column[i+1] == 1:
                        transition_2 = i

                first = transition_1//2
                second = (transition_2 - transition_1)//4
                third = transition_2 - second
                fourth = (state_column.shape[0] - transition_2) // 2

                state_column[first:transition_1] = -1
                state_column[transition_1:(transition_1 + second)] = -2
                state_column[third:transition_2] = -2
                state_column[transition_2:(transition_2+fourth)] = -1

                for i in range(state_column.shape[0]):
                    if state_column[i] == 1:
                        state_column[i] = 3
                    elif state_column[i] == -1:
                        state_column[i] = 2
                    elif state_column[i] == -2:
                        state_column[i] = 1
    else:
        for header in list(gesture_clip.columns):
            if 'state' in header:
                gesture_clip[header] = 4


    new_path = data_path.replace('training_data', 'training_data_multistate')
    new_dir = new_path.split("gesture_")[0]
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    gesture_clip.to_csv(new_path, index=False)


def selected_frame(data, num_frame):
    """This function uniformly samples data to num_frame frames.
    Not suitable for online recognition model
    """
    frame, dim = data.shape
    if frame == num_frame:
        return data
    interval = frame / num_frame
    uniform_list = [int(i * interval) for i in range(num_frame)]
    return data.iloc[uniform_list]

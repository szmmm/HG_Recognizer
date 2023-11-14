import math
import torch.nn.functional as F
import tqdm
from utils.prep_data import load_lh_data
from sklearn.model_selection import train_test_split
from utils.skeleton import *
from utils import parser
import numpy as np
from datetime import datetime
import time
import os
import logging
from model.rh_net import STADualNet
from model.lh_net import STANet
from model.temp_calibration import ModelWithTemperature
from train import train_model, get_cm, get_logger
from torch.utils.data import DataLoader

# ensure reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def init_lh_data_loader(args, logger, t_data, v_data=None):
    if v_data is None:
        data_set, label_set, state_set = load_lh_data(data_path=t_data, num_frame=args.frame_size)
        logger.info("Splitting data into train and test sets...")
        train_data, val_data, train_label, val_label, train_state, val_state = \
            train_test_split(data_set, label_set, state_set, test_size=0.1, stratify=label_set)
    else:
        logger.info("Segmenting training data from csv...")
        train_data, train_label, train_state = load_lh_data(data_path=t_data, num_frame=args.frame_size)
        logger.info("\nSegmenting test data from csv...")
        val_data, val_label, val_state = load_lh_data(data_path=v_data, num_frame=args.frame_size)

    aug = True
    if aug:
        logger.info("\nUsing data noise augmentation")

    train_dataset = SkeletonData(train_data, train_label, train_state, mode='train', use_data_aug=aug)

    test_dataset = SkeletonData(val_data, val_label, val_state, mode='test')

    train_data_loader = DataLoader(train_dataset,
                                   batch_size=args.batch_size, shuffle=True,
                                   num_workers=args.workers, pin_memory=False)

    val_data_loader = DataLoader(test_dataset,
                                 batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.workers, pin_memory=False)

    return train_data_loader, val_data_loader


def initialise_lh_model(args, model):
    class_num = 4  # including null class
    joint_num = 11
    input_dim = 7
    save_att = False

    init_model = model(num_classes=class_num, joint_num=joint_num, input_dim=input_dim,
                       window_size=args.frame_size, dp_rate=args.dp_rate, plot_att=save_att)
    device = torch.device("cuda")
    init_model = torch.nn.DataParallel(init_model).to(device)

    return init_model


def model_lh_forward(batched_sample, model, criterion):
    device = torch.device("cuda")

    data = batched_sample[0]
    data = data.to(device)

    class_labels = batched_sample[1]
    class_labels = class_labels.type(torch.LongTensor)
    class_labels = class_labels.to(device)

    class_prob = model(data)

    # print(state_prob.shape)

    loss = criterion(class_prob, class_labels)
    # print(loss)
    # accuracy = get_acc(class_prob, class_labels)

    return class_prob, loss


def train_lh_model(arguments, model_type, name, t_data, v_data=None):
    if arguments.training:
        # folder for saving trained model...
        # change this path to the fold where you want to save your pre-trained model
        name = datetime.now().strftime('%d%m%y') + '_' + name
        path = os.getcwd()
        model_fold = f"{path}/{name}"

        if not os.path.exists(model_fold):
            os.makedirs(model_fold)

        # get logger ready
        logger = get_logger(model_fold + "/train.log")

        train_loader, val_loader = init_lh_data_loader(arguments, logger, t_data, v_data)

        logger.info("\nInput Parameters:")
        logger.info(arguments)

        logger.info("\nInitialising model.............")
        model = initialise_lh_model(arguments, model_type)
        optimiser = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=arguments.learning_rate)
        loss_criterion = torch.nn.CrossEntropyLoss().cuda()

        train_data_num = len(train_loader.dataset)
        valid_data_num = len(val_loader.dataset)
        iter_per_epoch = math.ceil(train_data_num / arguments.batch_size)
        logger.info("Current device is: {}"
                    .format(torch.cuda.get_device_name(torch.cuda.current_device())))
        logger.info("training data num: {}".format(train_data_num))
        logger.info("test data num: {}".format(valid_data_num))
        logger.info("batch size: {}".format(arguments.batch_size))
        logger.info("num of workers: {}".format(arguments.workers))

        # parameters recording training log
        max_acc = 0
        min_loss = 100
        no_improve_epoch = 0
        n_iter = 0

        logger.info("\n*** Starting Training ***")
        for epoch in range(arguments.epochs):
            # Training Step
            model.train()
            start_time = time.time()
            train_loss = 0
            for i, sample_batched in enumerate(train_loader):
                n_iter += 1
                label = sample_batched[1]
                if i + 1 > iter_per_epoch:
                    continue
                cls, loss = model_lh_forward(sample_batched, model, loss_criterion)
                train_loss += loss.item()
                model.zero_grad()
                loss.backward()
                # clip_grad_norm_(model.parameters(), 0.1)
                optimiser.step()

                if i == 0:
                    score_list = cls
                    label_list = label
                else:
                    score_list = torch.cat((score_list, cls), 0)
                    label_list = torch.cat((label_list, label), 0)

            train_loss /= (i + 1)
            train_cm, train_acc = get_cm(score_list, label_list)

            logger.info("\nCurrent Epoch: [%2d] time: %4.4f, "
                        "train_loss: %.4f  train_acc: %.4f"
                        % (epoch + 1, time.time() - start_time,
                           train_loss, train_acc))
            start_time = time.time()

            # adjust_learning_rate(model_solver, epoch + 1, args)
            # print(print(model.module.encoder.gcn_network[0].edg_weight))

            # Validation Step
            with torch.no_grad():
                val_loss = 0
                model.eval()
                for i, sample_batched in enumerate(val_loader):
                    label = sample_batched[1]
                    cls, loss = model_lh_forward(sample_batched, model, loss_criterion)
                    val_loss += loss.item()

                    if i == 0:
                        score_list = cls
                        label_list = label

                    else:
                        score_list = torch.cat((score_list, cls), 0)
                        label_list = torch.cat((label_list, label), 0)

                val_loss /= (i + 1)
                val_cm, val_acc = get_cm(score_list, label_list)

                logger.info("Current Epoch: [%2d], "
                            "val_loss: %.4f  val_acc: %.4f"
                            % (epoch + 1, val_loss, val_acc))
                # save best model
                if val_acc > max_acc:
                    # if val_loss < min_loss:
                    no_improve_epoch = 0
                    min_loss = val_loss
                    val_loss = round(val_loss, 3)
                    max_acc = val_acc
                    best_model = "ep_{}_acc_{}_{}.pt".format(epoch + 1, val_acc,
                                                              datetime.now().strftime('%H%M%S'))
                    torch.save(model.module.state_dict(), f'{model_fold}/{best_model}')
                    logger.info(
                        "Performance improve, saving new model. Best accuracy: {}".format(max_acc))
                    logger.info(val_cm.diagonal() / val_cm.sum(axis=1))
                else:
                    no_improve_epoch += 1
                    logger.info("No_improve_epoch: {} Best accuracy {}".format(no_improve_epoch, max_acc))

                if no_improve_epoch == arguments.patience:
                    logger.info("*** Stop Training ***")
                    break

    if arguments.inference:
        print("\n*** Start Inference ***")
        try:
            model_path = f'{model_path}/{best_model}'
        except NameError:
            test_sub = '8'
            model_fold = f"{os.getcwd()}/saved_model_240323_LH_model_20f_scale_1-2_256_0.8"
            best_model = "ep_6_acc_0.995_134636"
            model_path = f'{model_fold}/{best_model}.pt'
            logger = get_logger(model_fold + "train.log")

            test_data, test_label, test_state = load_lh_data(data_path=v_data, num_frame=arguments.frame_size)
            test_dataset = SkeletonData(test_data, test_label, test_state, mode='test')
            val_loader = DataLoader(test_dataset,
                                    batch_size=arguments.batch_size, shuffle=False,
                                    num_workers=arguments.workers, pin_memory=False)
        loss_criterion = torch.nn.CrossEntropyLoss().cuda()
        model = initialise_lh_model(arguments, model_type)
        # model = ModelWithTemperature(model).to(torch.device('cuda'))
        model.module.load_state_dict(torch.load(model_path))

        with torch.no_grad():
            val_loss = 0
            model.eval()
            for i, sample in enumerate(val_loader):
                label = sample[1]
                cls, loss = model_lh_forward(sample, model, loss_criterion)
                val_loss += loss
                if i == 0:
                    score_list = cls
                    label_list = label
                else:
                    score_list = torch.cat((score_list, cls), 0)
                    label_list = torch.cat((label_list, label), 0)

            val_loss = val_loss / float(i + 1)
            val_cm, val_acc = get_cm(score_list, label_list)

            print(f"Test set loss: {val_loss}")
            print("\nClass Accuracy: %.4f"
                  % val_acc)

            # model.model.module.save_att_to(model_fold)
            # confusion_matrix(score_list, label_list, state_score, state_list)
            return torch.argmax(score_list, dim=1), label_list


if __name__ == "__main__":
    arg = parser.parser_args()

    test_subject = '1-2'

    # rh_tr_data = f'/training_data/P*[!{test_subject}]/*/*right_hand*.csv'
    # rh_v_data = f'/training_data/P*[{test_subject}]/*/*right_hand*.csv'
    #
    # lh_tr_data = f'/training_data/P*[!{test_subject}]/*/*left_hand*.csv'
    # lh_v_data = f'/training_data/P*[{test_subject}]/*/*left_hand*.csv'

    train_model(arg, STADualNet, "RH_model_20f_noise_1-2_256", rh_tr_data, rh_v_data)
    train_lh_model(arg, STANet, "LH_model_20f_noise_1-2_256", lh_tr_data, lh_v_data)

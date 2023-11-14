import torch.nn.functional as F
import tqdm
from utils.prep_data import load_data
from sklearn.model_selection import train_test_split
from utils.skeleton import *
from sklearn.metrics import confusion_matrix, accuracy_score
from datetime import datetime
import time
import os
import logging
from model.rh_net import *
from torch.utils.data import DataLoader
from model.temp_calibration import ModelWithTemperature

# ensure reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def init_data_loader(args, logger, t_data, v_data=None):
    if v_data is None:
        data_set, label_set, state_set = load_data(data_path=t_data, num_frame=args.frame_size)
        logger.info("Splitting data into train and test sets...")
        train_data, val_data, train_label, val_label, train_state, val_state = \
            train_test_split(data_set, label_set, state_set, test_size=0.2, stratify=label_set)
    else:
        logger.info("Segmenting training data from csv...")
        train_data, train_label, train_state = load_data(data_path=t_data, num_frame=args.frame_size)
        logger.info("\nSegmenting test data from csv...")
        val_data, val_label, val_state = load_data(data_path=v_data, num_frame=args.frame_size)

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


def initialise_model(args, model):
    class_num = 10  # including null class
    state_num = 5
    joint_num = 11
    input_dim = 7
    save_att = False

    init_model = model(num_classes=class_num, num_states=state_num, joint_num=joint_num, input_dim=input_dim,
                       window_size=args.frame_size, dp_rate=args.dp_rate, plot_att=save_att)
    device = torch.device("cuda")
    init_model = torch.nn.DataParallel(init_model).to(device)

    return init_model


def model_forward(batched_sample, model, criterion1, criterion2):
    device = torch.device("cuda")

    data = batched_sample[0]
    data = data.to(device)

    class_labels = batched_sample[1]
    class_labels = class_labels.type(torch.LongTensor)

    state_labels = batched_sample[2]
    state_labels = state_labels.type(torch.LongTensor)

    class_labels = class_labels.to(device)
    state_labels = state_labels.to(device)

    class_prob, state_prob = model(data)

    loss1 = criterion1(class_prob, class_labels)
    loss2 = criterion2(state_prob, state_labels)

    total_loss = 0.8 * loss1 + 0.2 * loss2

    return class_prob, state_prob, total_loss


def get_cm(scores, label):
    device = torch.device('cpu')
    scores = scores.to(device).data.numpy()
    label = label.to(device).data.numpy()
    outputs = np.argmax(scores, axis=1)
    # print(label)
    # print(outputs)
    m = confusion_matrix(label.flatten(), outputs.flatten(), normalize='true')
    acc = accuracy_score(label.flatten(), outputs.flatten())
    return m, acc


def get_logger(path):
    log = logging.getLogger(path)
    log.setLevel(logging.INFO)

    stream = logging.StreamHandler()
    stream.setLevel(logging.INFO)
    file_handle = logging.FileHandler(path)
    file_handle.setLevel(logging.INFO)

    log.addHandler(stream)
    log.addHandler(file_handle)

    return log


def train_model(arguments, model_type, name, t_data, v_data=None):
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

        train_loader, val_loader = init_data_loader(arguments, logger, t_data, v_data)

        logger.info("\nInput Parameters:")
        logger.info(arguments)

        logger.info("\nInitialising model.............")
        model = initialise_model(arguments, model_type)
        optimiser = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=arguments.learning_rate)
        loss_criterion1 = torch.nn.CrossEntropyLoss().cuda()
        loss_criterion2 = torch.nn.CrossEntropyLoss().cuda()

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
                state_truth = sample_batched[2]
                if i + 1 > iter_per_epoch:
                    continue
                cls, state, loss = model_forward(sample_batched, model, loss_criterion1, loss_criterion2)
                train_loss += loss.item()
                model.zero_grad()
                loss.backward()
                # clip_grad_norm_(model.parameters(), 0.1)
                optimiser.step()

                if i == 0:
                    score_list = cls
                    label_list = label
                    state_score = state
                    state_list = state_truth
                else:
                    score_list = torch.cat((score_list, cls), 0)
                    label_list = torch.cat((label_list, label), 0)
                    state_score = torch.cat((state_score, state), 0)
                    state_list = torch.cat((state_list, state_truth), 0)

            train_loss /= (i + 1)
            train_cm, train_acc = get_cm(score_list, label_list)
            train_state_cm, train_state_acc = get_cm(state_score, state_list)

            logger.info("\nCurrent Epoch: [%2d] time: %4.4f, "
                        "train_loss: %.4f  train_acc: %.4f  train_state_acc: %.4f"
                        % (epoch + 1, time.time() - start_time,
                           train_loss, train_acc, train_state_acc))
            start_time = time.time()

            # adjust_learning_rate(model_solver, epoch + 1, args)
            # print(print(model.module.encoder.gcn_network[0].edg_weight))

            # Validation Step
            with torch.no_grad():
                val_loss = 0
                model.eval()
                for i, sample_batched in enumerate(val_loader):
                    label = sample_batched[1]
                    state_truth = sample_batched[2]
                    cls, state, loss = model_forward(sample_batched, model, loss_criterion1, loss_criterion2)
                    val_loss += loss.item()

                    if i == 0:
                        score_list = cls
                        label_list = label
                        state_score = state
                        state_list = state_truth
                    else:
                        score_list = torch.cat((score_list, cls), 0)
                        label_list = torch.cat((label_list, label), 0)
                        state_score = torch.cat((state_score, state), 0)
                        state_list = torch.cat((state_list, state_truth), 0)

                val_loss /= (i + 1)
                val_cm, val_acc = get_cm(score_list, label_list)
                val_state_cm, val_state_acc = get_cm(state_score, state_list)

                logger.info("Current Epoch: [%2d], "
                            "val_loss: %.4f  val_acc: %.4f val_state_acc: %.4f"
                            % (epoch + 1, val_loss, val_acc, val_state_acc))

                total_acc = val_acc * 0.8 + val_state_acc * 0.2
                # save best model
                if total_acc > max_acc:
                    # if val_loss < min_loss:
                    total_acc = round(total_acc, 3)
                    no_improve_epoch = 0
                    min_loss = val_loss
                    max_acc = total_acc
                    val_loss = round(val_loss, 3)
                    total_acc = round(total_acc, 3)
                    best_model = "ep_{}_acc_{}_{}".format(epoch + 1, total_acc,
                                                          datetime.now().strftime('%H%M%S'))
                    torch.save(model.module.state_dict(), f'{model_fold}/{best_model}.pt')
                    logger.info(
                        "Performance improve, saving new model. Best average accuracy: {}".format(total_acc))
                    # print(val_cm)
                    logger.info(val_cm.diagonal() / val_cm.sum(axis=1))
                    # logger.info(val_state_cm.diagonal()/val_state_cm.sum(axis=1))

                else:
                    no_improve_epoch += 1
                    logger.info("No_improve_epoch: {} Best accuracy {}".format(no_improve_epoch, max_acc))

                if no_improve_epoch == arguments.patience:
                    logger.info("*** Stop Training ***")
                    break

    if arguments.inference:
        print("\n*** Start Inference ***")
        try:
            model_path = f'{model_fold}/{best_model}'
        except NameError:
            model_fold = "saved_models/saved_model_240323_RH_model_20f_scale_1-2_256_0.8"
            best_model = "ep_7_acc_0.948_134202_scaled"
            model_path = f'{model_fold}/{best_model}.pt'
            logger = get_logger(model_fold + "/train.log")

            val_data, val_label, val_state = load_data(data_path=v_data, num_frame=arguments.frame_size)
            test_dataset = SkeletonData(val_data, val_label, val_state, mode='test')
            val_loader = DataLoader(test_dataset,
                                    batch_size=arguments.batch_size, shuffle=False,
                                    num_workers=arguments.workers, pin_memory=False)
        loss_criterion1 = torch.nn.CrossEntropyLoss().cuda()
        loss_criterion2 = torch.nn.CrossEntropyLoss().cuda()
        model = initialise_model(arguments, model_type)
        model = ModelWithTemperature(model).to(torch.device('cuda'))
        model.load_state_dict(torch.load(model_path))

        with torch.no_grad():
            val_loss = 0
            model.eval()
            for i, sample in enumerate(val_loader):
                label = sample[1]
                state_truth = sample[2]
                cls, state, loss = model_forward(sample, model, loss_criterion1, loss_criterion2)
                val_loss += loss
                if i == 0:
                    score_list = cls
                    label_list = label
                    state_score = state
                    state_list = state_truth
                else:
                    score_list = torch.cat((score_list, cls), 0)
                    label_list = torch.cat((label_list, label), 0)
                    state_score = torch.cat((state_score, state), 0)
                    state_list = torch.cat((state_list, state_truth), 0)

            val_loss = val_loss / float(i + 1)
            val_cm, val_acc = get_cm(score_list, label_list)
            val_state_cm, val_state_acc = get_cm(state_score, state_list)

            print(f"Test set loss: {val_loss}")
            print("\nClass Accuracy: %.4f"
                  "\nState Accuracy: %.4f"
                  % (val_acc, val_state_acc))

        return torch.argmax(score_list, dim=1), label_list, val_state_cm

    if arguments.calibration:
        try:
            model_path = f'{model_fold}/{best_model}.pt'
        except NameError:
            model_fold = f"{os.getcwd()}/131123_RH_model_20f_noise_1-2_256"
            best_model = "ep_1_acc_0.887_153402"
            model_path = f'{model_fold}/{best_model}.pt'
            logger = get_logger(model_fold + "/train.log")
            train_loader, val_loader = init_data_loader(arguments, logger, t_data, v_data)
        model = initialise_model(arguments, model_type)
        model.module.load_state_dict(torch.load(model_path))

        epoch = 5
        scaled_model = ModelWithTemperature(model)

        for i in range(epoch):
            scaled_model, before_nll, before_ece, after_nll, after_ece = scaled_model.set_temperature(val_loader)

            logger.info('Before temperature - NLL: %.3f, ECE: %.3f' % (before_nll, before_ece))
            logger.info('After temperature - NLL: %.3f, ECE: %.3f' % (after_nll, after_ece))
            logger.info('Optimal temperature: %.3f' % scaled_model.temperature.item())

        torch.save(scaled_model.state_dict(), f'{model_fold}/{best_model}_scaled.pt')

        print(f"Final Temperature: {scaled_model.temperature.item()}")
        print("\nModel calibrated SUCCESSFULLY!")

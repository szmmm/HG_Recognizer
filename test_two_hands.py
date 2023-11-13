from model.lh_net import *
from train import *
from utils import parser
from utils.prep_data import *
from server import right_hand_inference, left_hand_inference, generate_label, ToolID
import editdistance

temp = True

if __name__ == "__main__":
    print('\n=====================\nRunning Online Recognition Test\n=====================')

    model_fold = "/right_model"
    output_dir = "HG_Recogniser/log"  # logging currently not in use
    best_model = "best_model"
    model_path = f'{model_fold}/{best_model}.pth'

    lh_model_fold = "/left_model"
    lh_best_model = "best_model"
    lh_model_path = f'{lh_model_fold}/{lh_best_model}.pth'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    args = parser.parser_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = STADualNet(num_classes=10, num_states=5, joint_num=11, input_dim=7, window_size=args.frame_size,
                       dp_rate=args.dp_rate)
    model = torch.nn.DataParallel(model)

    if temp:
        model = ModelWithTemperature(model)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        num_class = model.model.module.get_class_num()
        window_size = model.model.module.get_window_size()
        input_dim = model.model.module.get_input_dim()
        joint_num = model.model.module.get_joint_num()

    else:
        model.module.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        num_class = model.module.get_class_num()
        window_size = model.module.get_window_size()
        input_dim = model.module.get_input_dim()
        joint_num = model.module.get_joint_num()

    print("\nRight Hand Model Loaded Successfully!")

    lh_model = STANet(num_classes=4, joint_num=11, input_dim=7, window_size=args.frame_size,
                      dp_rate=args.dp_rate)
    lh_model = torch.nn.DataParallel(lh_model)
    lh_model.module.load_state_dict(torch.load(lh_model_path, map_location=device))
    lh_model.to(device)

    print("\nLeft Hand Model Loaded Successfully!")

    fsm = 0  # initial FSM state
    prev_class = -1
    final_class = -1
    subject_num = 0

    class_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    out_dict = {}
    total_score = 0
    step = 0
    step_size = 0

    logger = get_logger(model_fold + f"/test.log")
    logger.info("RH Model: " + best_model)
    logger.info("LH Model: " + lh_best_model)

    rh_data_path = "C:/online_seq/*/*right_hand*.csv"
    lh_data_path = "C:/online_seq/*/*left_hand*.csv"

    right_results = []
    left_results = []
    prob_all = []
    for file_name in tqdm(sorted(glob.glob(rh_data_path))):
        subject_num += 1
        right_csv = read_data_from_csv(file_name)  # split csv into data points with label
        file_name = file_name.replace("right_hand", "left_hand")
        left_csv = read_data_from_csv(file_name)

        if 'state' in list(right_csv.columns):
            right_csv = right_csv.drop('state', axis='columns')
            left_csv = left_csv.drop('state', axis='columns')

        right_csv = np.array(np.split(right_csv, 11, axis=1))
        right_csv = np.swapaxes(right_csv, 0, 1)
        left_csv = np.array(np.split(left_csv, 11, axis=1))
        left_csv = np.swapaxes(left_csv, 0, 1)

        right_labels = []
        left_labels = []
        prob = []

        for frame_end in range(args.frame_size, right_csv.shape[0]):
            # for frame_end in range(1700, 1900):
            # if step < step_size:
            #     step += 1
            # else:
            #     step = 0
            sample = right_csv[frame_end - args.frame_size:frame_end, :, :]
            lh_sample = left_csv[frame_end - args.frame_size:frame_end, :, :]

            sample = torch.tensor(sample, dtype=torch.float).to(device)
            lh_sample = torch.tensor(lh_sample, dtype=torch.float).to(device)

            rh_class, predict_state, class_prob = right_hand_inference(model, sample)
            lh_class, lh_prob = left_hand_inference(lh_model, lh_sample)

            max_prob = (torch.max(class_prob).item() + torch.max(lh_prob).item()) / 2
            score_str = ','.join(list(map(str, class_prob.tolist())))

            right_labels.append(rh_class)
            left_labels.append(lh_class)
            prob.append(max_prob)

        right_results.append(right_labels)
        left_results.append(left_labels)
        prob_all.append(prob)

    predictions = []
    labels = []

    for i in range(len(right_results)):
        gs = [10]
        for j in range(len(right_results[i])):
            g = generate_label(right_results[i][j], left_results[i][j])
            p = prob_all[i][j]
            # Finite State Machine
            if p > 0.9:
                if g != prev_class:
                    prev_class = g
                    fsm = 0  # initial state
                else:
                    fsm += 1
                    if fsm == 30:  # 30 consecutive identical gestures triggers classification
                        final_class = g
                        fsm = 0
            else:
                final_class = ToolID.Null  # null

            if final_class != ToolID.Null and final_class != gs[-1]:
                gs.append(final_class)
        score = editdistance.eval(gs[1:], class_labels)

        predictions += gs[1:]
        labels += class_labels

        total_score += score
        for output in gs[1:]:
            if output not in out_dict.keys():
                out_dict[output] = 1
            else:
                out_dict[output] += 1
        logger.info("distance: {}".format(score))
        logger.info("prediction: {}".format(gs[1:]))

    logger.info("\nNum of subjects: {}".format(subject_num))
    logger.info("Final distance: {}".format(total_score))
    logger.info("Percentage distance: {}".format(total_score / (len(class_labels) * subject_num)))
    logger.info("Occurrences: {}".format(out_dict))

import socket
from train import *
from model.rh_net import STADualNet
from model.lh_net import STANet
from utils import parser


class ToolID:
    Scale, Duplicate, Delete, \
    Pen, Cube, Cylinder, Sphere, Spray, Cut, Palette, Null = range(11)


class RToolID:
    Scale, DupDel, \
    Pen, Cube, Cylinder, Sphere, Spray, Cut, Palette, Null = range(10)


class LToolID:
    Scale, Duplicate, Delete, Null = range(4)


def right_hand_inference(r_model, d):
    with torch.no_grad():
        r_model.eval()
        c_prob, s_prob = r_model(d.unsqueeze(0))
        c_prob = c_prob.squeeze(0)
        s_prob = s_prob.squeeze(0)

        c_prob = torch.nn.Softmax(dim=0)(c_prob)

        cls = torch.argmax(c_prob).item()
        stt_list = torch.argmax(s_prob, dim=0)

        # majority voting over all frames for states
        stt = stt_list.bincount().argmax().item()

        # print(f"Class: {predict_class}"
        #       f"\nPredicted Scores: {predict_score}")

    return cls, stt, c_prob


def left_hand_inference(l_model, d):
    with torch.no_grad():
        l_model.eval()
        c_prob = l_model(d.unsqueeze(0))
        c_prob = c_prob.squeeze(0)
        c_prob = torch.nn.Softmax(dim=0)(c_prob)
        # print(class_prob)

        cls = torch.argmax(c_prob).item()

        # print(f"Class: {predict_class}"
        #       f"\nPredicted Scores: {predict_score}")

    return cls, c_prob


def generate_label(r_cls, l_cls):
    # Check two hand gestures
    if r_cls == RToolID.DupDel:
        if l_cls == LToolID.Duplicate or l_cls == LToolID.Delete:
            out = l_cls
        else:
            out = ToolID.Null

    elif r_cls == RToolID.Scale:
        if l_cls != r_cls:
            out = ToolID.Null
        else:
            out = r_cls

    # Adjust RToolID to ToolID
    else:
        out = r_cls + 1

    return out


temp = True
left_hand = True

if __name__ == "__main__":
    print('\n=====================\nGesture Recognition Server\n=====================')

    model_fold = "/right_model"
    output_dir = "/log"  # logging currently not in use
    best_model = "best_model"
    model_path = f'{model_fold}/{best_model}.pth'

    lh_model_fold = "/right_model"
    lh_best_model = "best_model"
    lh_model_path = f'{lh_model_fold}/{lh_best_model}.pth'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    args = parser.parser_args()
    # device = torch.device("cpu")
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

    if left_hand:
        lh_model = STANet(num_classes=4, joint_num=11, input_dim=7, window_size=args.frame_size,
                          dp_rate=args.dp_rate)
        lh_model = torch.nn.DataParallel(lh_model)
        lh_model.module.load_state_dict(torch.load(lh_model_path, map_location=device))
        lh_model.to(device)

        print("\nLeft Hand Model Loaded Successfully!")

    # Create a TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Bind the socket to the port
    server_address = ('localhost', 11001)
    print('\nStarting Server on %s Port %s' % server_address)
    sock.bind(server_address)
    ip_address = socket.gethostbyname(socket.gethostname())
    print('IP Address: %s' % ip_address)
    print("To Connect Use: 127.0.0.1")

    # Enum of message types, must match GestureSpotterClient
    class MessageType:
        Invalid, Acknowledge, Goodbye, DataSample, PredictedClass = range(5)


    data_buffer = []
    lh_data_buffer = []

    fsm = 0  # initial FSM state
    final_class = -1
    last_class = -1

    step = 0
    step_size = 10
    # Listen for incoming connections
    sock.listen(1)
    buffer_size = 4096

    while True:
        # Wait for a connection
        print('\nWaiting for a connection...')
        connection, client_address = sock.accept()

        try:
            print('connection from', client_address)

            # Receive the data in small chunks
            while True:
                data = connection.recv(buffer_size)
                # print('Received "%s"' % data)

                if data:
                    splitData = data.decode().split('\t')
                    # Data sent in "{Message Type}\t{Coordinates}\n"
                    if len(splitData) > 1 and len(splitData[0]) == 1:

                        messageType = int(splitData[0])
                        # Message handling cases

                        # Acknowledge
                        if messageType == MessageType.Acknowledge:
                            print('\nReceived ACK message: %s' % splitData[1])
                            connection.sendall(str(int(MessageType.Acknowledge)).encode())

                        # Real-Time Data
                        elif messageType == MessageType.DataSample:
                            data_sample_str = splitData[1].split('\n')[0]  # trim data to avoid warning
                            data_sample = np.fromstring(data_sample_str, sep=',').reshape((-1, input_dim))

                            # Append sample to buffer
                            # [frame, joints, dimension]
                            if len(data_buffer) == 0:
                                data_buffer.append(data_sample)
                            elif data_sample.shape[0] == data_buffer[-1].shape[0]:
                                data_buffer.append(data_sample)

                            if len(data_buffer) >= window_size:
                                # if step < step_size:
                                #     step += 1
                                # else:
                                #     step =

                                # Slice out data frame from buffer
                                # [window_size, joints, dimension]
                                sample = torch.tensor(data_buffer[-window_size:], dtype=torch.float).to(device)

                                # ============= Right Hand ============== #
                                if sample.size(1) == joint_num:  # single hand data
                                    pred_class, pred_stt, class_prob = right_hand_inference(model, sample)
                                    max_prob = torch.max(class_prob).item()
                                    score_str = ','.join(list(map(str, class_prob.tolist())))

                                    # Finite State Machine
                                    if max_prob > 0.8:
                                        if pred_class != last_class:
                                            last_class = pred_class
                                            fsm = 0  # initial state
                                        else:
                                            fsm += 1
                                            if fsm == 10:  # 20 consecutive identical gestures triggers classification
                                                final_class = pred_class
                                                fsm = 0
                                    else:
                                        final_class = int(ToolID.Null)  # null

                                    # Reply with prediction message
                                    prediction_msg = "%s\t%d:%d:%s\n" % (str(int(MessageType.PredictedClass)),
                                                                         final_class, pred_stt, score_str)
                                    # print(prediction_msg)
                                    connection.sendall(prediction_msg.encode())

                                # ============= Two Hands ============== #
                                elif sample.size(1) > joint_num:
                                    sample, lh_sample = torch.split(sample, joint_num, dim=1)
                                    rh_class, pred_stt, class_prob = right_hand_inference(model, sample)
                                    lh_class, lh_prob = left_hand_inference(lh_model, lh_sample)

                                    pred_class = generate_label(rh_class, lh_class)
                                    max_prob = (torch.max(class_prob).item() + torch.max(lh_prob).item()) / 2

                                    score_str = ','.join(list(map(str, class_prob.tolist())))
                                    lh_score_str = ','.join(list(map(str, lh_prob.tolist())))

                                    # Finite State Machine
                                    if max_prob > 0.9:
                                        if pred_class != last_class:
                                            last_class = pred_class
                                            fsm = 0  # initial state
                                        else:
                                            fsm += 1
                                            if fsm == 30:  # 30 consecutive identical gestures triggers classification
                                                final_class = pred_class
                                                fsm = 0
                                    else:
                                        final_class = ToolID.Null  # null

                                    # Reply with prediction message
                                    prediction_msg = "%s\t%d:%d:%d:%d:%s\n" % (str(int(MessageType.PredictedClass)),
                                                                               final_class, lh_class, rh_class,
                                                                               pred_stt, score_str)
                                    connection.sendall(prediction_msg.encode())

                            # else:
                            #     prediction_msg = 'waiting'
                            #     connection.sendall(prediction_msg.encode())
                        # Goodbye
                        elif messageType == MessageType.Goodbye:
                            print("Goodbye message, saving buffer")
                            output_filename = '{}/gesture_log_{}.csv'.format(output_dir,
                                                                             time.strftime("%Y%m%d_%H%M%S"))
                            # np.savetxt(output_filename, data_buffer.reshape((-1, 6*7)),
                            # fmt='%1.4f', delimiter=",")

                            # Reset data buffer
                            data_buffer = []
                            connection.sendall(str(int(MessageType.Acknowledge)).encode())
                            break

                    else:
                        # print('no more data from', client_address)
                        print(f'malformed msg received: {splitData}')

        finally:
            # Clean up the connection
            connection.close()

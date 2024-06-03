import argparse
import sys
import random

import numpy as np
from torch import optim

from aggregation import *
from attack import LIT_attack
from model.Models import *
from model.resnetcifar import *
from clients import ClientsGroup
# from model.WideResNet import WideResNet
from torch.utils.tensorboard import SummaryWriter
from utils import *
import datetime
import logging
import threading
import time
import copy
import socket

import pandas as pd
import torchvision.models as models

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")
    parser.add_argument('--debug', type=bool, default=False, help='True, False')
    parser.add_argument('--byz_type', type=str, default="GS_attack", help="LF_attack,GS_attack,LIT_attack,Scaling_attack,no_attack")
    parser.add_argument('--agg_type', type=str, default="pca_hdbscan_b",
                        help="average, multi_krum, auror, foolsgold, FLDetector, pca_kmeans_a,  pca_agglomer_c,"
                             "pca_agglomer_a, pca_hdbscan_b,pca_hdbscan_c")
    parser.add_argument('-dataset', "--dataset", type=str, default="cifar10", help="mnist,emnist,cifar10")
    parser.add_argument('-E', '--epoch', type=int, default=1, help='local train epoch')
    parser.add_argument('--pca_d', type=int, default=10, help='numer of pca descending dimensions')
    parser.add_argument('-iid', '--IID', type=bool, default=True, help='the way to allocate data to clients')
    parser.add_argument('--beta', type=float, default=9, help='The parameter for the dirichlet distribution 1,3,5,7,9')
    # parser.add_argument('-B', '--batchsize', type=int, default=256, help='local train batch size')

    parser.add_argument('-nc', '--num_of_clients', type=int, default=100, help='numer of the clients')
    parser.add_argument('-nmc', '--num_malicious_client', type=int, default=45, help='numer of the clients')
    parser.add_argument('-cf', '--cfraction', type=float, default=1, help='0 means 1 client')
    parser.add_argument('-lr', "--learning_rate", type=float, default=0.01, help="learning rate")
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    # parser.add_argument('-mn', '--model_name', type=str, default='mnist_cnn', help='the model to train')
    # parser.add_argument('-ncomm', '--round', type=int, default=201, help='number of communications')

    parser.add_argument('--load_model_file', type=str, default=None, help='the model to load as global model')
    parser.add_argument('--load_model_round', type=int, default=0, help='how many rounds have executed')
    # parser.add_argument('--load_model_file', type=str, default=r"G:\FL\By-FL\logs\2022-10-22\23.14.05\global_weight\globalmodel.pth", help='the model to load as global model')
    # parser.add_argument('--load_model_round', type=int, default=101, help='how many rounds have executed')

    args = parser.parse_args()
    args = args.__dict__
    return args


if __name__ == "__main__":

    # -----------------------设置各项参数-----------------------#
    args = get_args()
    # os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # -----------------------配置日志文件，结果保存，以及TensorBoard-----------------------#
    logdir = os.path.join(args['logdir'], str(datetime.datetime.now().strftime("%Y-%m-%d/%H.%M.%S")))
    mkdirs(logdir)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        filename=os.path.join(logdir, 'info.log'),
        format='[%(levelname)s](%(asctime)s) %(message)s',
        datefmt='%Y/%m/%d/ %I:%M:%S %p', level=logging.DEBUG, filemode='w')
    logger = logging.getLogger()

    _host_name = socket.gethostname()

    file = open(logdir + '/readme.md', mode='a', encoding='utf=8')
    file.write('###实验机器：{} \n'.format(_host_name))
    logger.info("***"+str(_host_name)+"***")
    file.write('###实验目的：\n \n \n')
    file.write('###实验参数：\n{}\n \n'.format(args))
    file.write('###实验结果：\n \n \n')
    file.write('###结果分析：\n \n \n')
    file.close()
    # file.write(logdir + '\n' + '本次实验的说明，修改了哪些变量，主要的目的是验证什么')

    # initiate TensorBoard for tracking losses and metrics
    writer = SummaryWriter(log_dir=logdir, filename_suffix="info")
    tb_port = 6007
    tb_host = "127.0.0.1"
    tb_thread = threading.Thread(
        target=launch_tensor_board,
        args=([logdir, tb_port, tb_host]),
        daemon=True
    ).start()
    time.sleep(3.0)

    print("**Basic Setting...")
    logger.info("**Basic Setting...")
    print('  ', args)
    logging.info(args)

    # ------------------------创建clients，分割数据----------------------------#
    print("**Initializing clients data...")
    logger.info("**Initializing clients data...")
    myClients = ClientsGroup(args['dataset'], args['IID'], args['num_of_clients'], args['beta'], args['datadir'], dev)
    testDataLoader = myClients.test_data_loader

    print("  ", myClients.clients_distributions)
    logger.info(str(myClients.clients_distributions))

    # ------------------------初始化模型----------------------------#
    print('**Initializing nets...')
    logger.info("**Initializing nets")
    net = None
    init_img = None
    n_comm_rounds = 100
    batchsize = 256

    # 初始化模型  init_img TensorBoard展示模型
    if args['dataset'] == 'mnist':
        net = Mnist_2NN()
        init_img = torch.zeros((1, 1, 28, 28), device=dev)
        n_comm_rounds = 100
        batchsize = 64

    elif args['dataset'] == 'emnist':
        net = EMnist_CNN()
        init_img = torch.zeros((1, 1, 28, 28), device=dev)
        n_comm_rounds = 100
        batchsize = 256

    elif args['dataset'] == 'cifar10':
        net = ResNet18_cifar10(num_classes=10)
        init_img = torch.zeros((1, 3, 32, 32), device=dev)
        n_comm_rounds = 100
        batchsize = 128

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = torch.nn.DataParallel(net)

    net = net.to(dev)
    writer.add_graph(net, init_img)

    # 定义损失函数
    loss_func = F.cross_entropy
    lr = args['learning_rate']
    opti = optim.Adam(net.parameters(), lr=lr)

    # 得到全局的参数
    global_parameters = {}

    # 确定开始轮次，是从0开始的，还是断了再续的

    start_rounds = 0
    if args['load_model_file']:
        global_parameters = torch.load(args['load_model_file'])
        start_rounds = args['load_model_round']
    else:
        for key, var in net.state_dict().items():
            global_parameters[key] = var.clone()

    # ---------------------------------------以上准备工作已经完成------------------------------------------#
    if args['debug']:
        # 调试简化客户端
        num_in_comm = 10
        clients_all = ['client{}'.format(i) for i in range(0, num_in_comm)]
        random.seed(3)
        malicious_clients = random.sample(clients_all, 3)
    else:
        # 选择参与训练的Clients
        num_in_comm = int(max(args['num_of_clients'] * args['cfraction'], 1))
        clients_all = ['client{}'.format(i) for i in range(0, num_in_comm)]
        random.seed(1)
        malicious_clients = random.sample(clients_all, args['num_malicious_client'])

    benign_clients = []
    for c in clients_all:
        if c not in malicious_clients:
            benign_clients.append(c)


    if args['byz_type'] != 'no_attack':
        print("  malicious_clients: ", malicious_clients)
        logger.info("**malicious_clients: " + str(malicious_clients))
        print('  malicious_clients: %d / %d' % (len(malicious_clients), len(clients_all)))
        logger.info('**malicious_clients: %d / %d' % (len(malicious_clients), len(clients_all)))

    print('**Completed...')
    logger.info("**Completed...")
    print("-" * 100 + '  Start Training  ' + "-" * 100)
    logger.info("-" * 100 + '  Start Training  ' + "-" * 100)
    # -------------------------------------------开始训练----------------------------------------------#

    # FLDetector 用到的参数
    client_grad_list, old_client_grad_list = [], []
    weight_record, grad_record = [], []
    score_record = []
    last_weight, last_global_grad = None, None
    if args['debug']:
        N, K, B = 2, 3, 5
    else:
        N, K, B = 10, 10, 20

    # auror
    malicious_records = []
    malicious_score1 = np.zeros(num_in_comm)
    malicious_score = np.ones((num_in_comm))/2

    if args['byz_type'] == 'no_attack':
        clients_in_comm = benign_clients
    else:
        clients_in_comm = clients_all

    for round in range(start_rounds, n_comm_rounds):
        print("**COMMON ROUND:", str(round))
        logger.info("**COMMON ROUND:" + str(round))
        net = net.to(dev)
        client_params = {}

        '''
            local_parameters  本地更新模型
        '''
        for client in clients_in_comm:
            local_param = myClients.clients_set[client].localUpdate_New(args['epoch'], batchsize, net,
                                                                        loss_func, opti, global_parameters,
                                                                        malicious_clients, args['byz_type'], num_in_comm)

            local_loss, local_acc = test_accuracy(net, local_param, testDataLoader)

            # 将模型参数打平
            local_param_flatten = torch.cat([param.data.clone().view(-1) for key, param in local_param.items()], dim=0)
            client_params[client] = copy.deepcopy(local_param_flatten.cpu())

            # print(torch.cuda.memory_allocated())
            print(
                '[Round: %d %s] accuracy: %f  loss: %f | %s' % (round, client, local_acc, local_loss, args['byz_type']))
            logger.info(
                '[Round: %d %s] accuracy: %f  loss: %f | %s' % (round, client, local_acc, local_loss, args['byz_type']))

        if args['byz_type'] == 'LIT_attack':
            print("LIT_attacking")
            mal_params_net = {}
            client_params = LIT_attack(myClients, client_params, args['epoch'], batchsize, net,
             loss_func, opti, global_parameters, malicious_clients)

            for client in malicious_clients:
                mal_params = client_params[client]
                start_idx = 0
                for key, var in net.state_dict().items():
                    param = mal_params[start_idx:start_idx + len(var.data.view(-1))].reshape(var.data.shape)
                    start_idx = start_idx + len(var.data.view(-1))
                    mal_params_net[key] = copy.deepcopy(param)

                local_loss, local_acc = test_accuracy(net, mal_params_net, testDataLoader)
                print(
                    '[LIT_attack %s] accuracy: %f  loss: %f ' % (client, local_acc, local_loss))
                logger.info(
                    '[LIT_attack %s] accuracy: %f  loss: %f ' % (client, local_acc, local_loss))

        '''
            服务器聚合
        '''

        start = time.time()
        # agg_params, detect_malicious_client = server_agg(client_params, args['agg_type'], len(malicious_clients), args['pca_d'], logger)

        if args['agg_type'] == 'krum':
            print("agg_type: " + "krum")
            logger.info("agg_type: " + "krum")
            agg_params, detect_malicious_client = agg_multi_krum(client_params, len(clients_in_comm) - 1)

        elif args['agg_type'] == 'multi_krum':
            print("agg_type: " + "multi_krum")
            logger.info("agg_type: " + "multi_krum")
            agg_params, detect_malicious_client = agg_multi_krum(client_params, len(malicious_clients))

        elif args['agg_type'] == 'foolsgold':
            print("agg_type: " + "foolsgold")
            logger.info("agg_type: " + "foolsgold")
            agg_params, detect_malicious_client = agg_foolsgold(client_params)


        elif args['agg_type'] == 'auror':
            print("agg_type: " + "auror")
            logger.info("agg_type: " + "auror")

            detect_malicious_client = []
            benign_client_params = {}

            malicious_record = agg_auror(client_params, args['dataset'])
            malicious_records.append(malicious_record)
            if len(malicious_records) > 10:
                del malicious_records[0]
            sum_records = np.sum(malicious_records, axis=0)
            for i, value in enumerate(sum_records):
                if value > len(malicious_records) / 2:
                    detect_malicious_client.append('client{}'.format(i))
                else:
                    benign_client_params['client{}'.format(i)] = client_params['client{}'.format(i)]

            if len(benign_client_params) == 0:
                agg_params = agg_average(client_params)
            else:
                agg_params = agg_average(benign_client_params)




        elif args['agg_type'] == 'pca_agglomer_a':
            print("agg_type: " + "pca_agglomer_a")
            logger.info("agg_type: " + "pca_agglomer_a")

            detect_malicious_client = []
            benign_client_params = {}

            malicious_record = agg_pca_agglomer(client_params, args['pca_d'], round, logdir, args['dataset'])

            # 集成鲁棒 参考auror
            malicious_records.append(malicious_record)
            if len(malicious_records) > 10:
                del malicious_records[0]
            sum_records = np.sum(malicious_records, axis=0)
            print(sum_records)
            for i, value in enumerate(sum_records):
                if value > len(malicious_records) / 2:
                    detect_malicious_client.append('client{}'.format(i))
                else:
                    benign_client_params['client{}'.format(i)] = client_params['client{}'.format(i)]

            if len(benign_client_params) == 0:
                agg_params = agg_average(client_params)
            else:
                agg_params = agg_average(benign_client_params)



        elif args['agg_type'] == 'pca_hdbscan_b':
            print("agg_type: " + "pca_hdbscan_b")
            logger.info("agg_type: " + "pca_hdbscan_b")

            detect_malicious_client = []
            benign_client_params = {}

            malicious_record, grad = agg_pca_hdbscan(client_params, args['pca_d'], round)
            print(malicious_record)
            # malicious_records.append(malicious_record)

            malicious_score1 = malicious_score1 + malicious_record
            print(malicious_score1)
            # 集成鲁棒
            for i, value in enumerate(malicious_score1):
                if value > (round + 1) / 2:
                    detect_malicious_client.append('client{}'.format(i))
                else:
                    benign_client_params['client{}'.format(i)] = client_params['client{}'.format(i)]

            if len(benign_client_params) == 0:
                agg_params = agg_average(client_params)
            else:
                agg_params = agg_average(benign_client_params)



        else:
            print("agg_type: " + "average")
            logger.info("agg_type: " + "average")
            agg_params = agg_average(client_params)
            detect_malicious_client = {}

        agg_time = time.time() - start
        defense_acc, malicious_precision, malicious_recall = computer_defense_acc(detect_malicious_client,
                                                                                  malicious_clients, clients_in_comm)

        # 将聚合后的参数对应到网络中
        start_idx = 0
        for key, var in global_parameters.items():
            param = agg_params[start_idx:start_idx + len(var.data.view(-1))].reshape(var.data.shape)
            start_idx = start_idx + len(var.data.view(-1))
            global_parameters[key] = copy.deepcopy(param)

        '''
            输出各项结果
        '''

        logger.info('[Round: %d] >> Detect_malicious_client: %s' % (round, str(detect_malicious_client)))

        logger.info('[Round: %d] >> Number_malicious_client: %d' % (round, len(detect_malicious_client)))
        print('[Round: %d] >> Number_malicious_client: %d' % (round, len(detect_malicious_client)))

        logger.info('[Round: %d] >> Server Defense accuracy: %f' % (round, defense_acc))
        print('[Round: %d] >> Server Defense accuracy: %f' % (round, defense_acc))
        writer.add_scalar('scalar/defense_acc', defense_acc, round)

        logger.info('[Round: %d] >> Server Detect malicious Precision: %f' % (round, malicious_precision))
        print('[Round: %d] >> Server Detect malicious Precision: %f' % (round, malicious_precision))
        writer.add_scalar('scalar/malicious_precision', malicious_precision, round)

        logger.info('[Round: %d] >> Server Detect malicious Recall: %f' % (round, malicious_recall))
        print('[Round: %d] >> Server Detect malicious Recall: %f' % (round, malicious_recall))
        writer.add_scalar('scalar/malicious_recall', malicious_recall, round)

        logger.info('[Round: %d] >> Time of aggregation: %f s' % (round, agg_time))
        print('[Round: %d] >> Time of aggregation: %f s' % (round, agg_time))

        '''
            训练结束之后，通过测试集来验证方法的泛化性，
            注意:虽然训练时，Server没有得到过任何一条数据，但是联邦学习最终的目的
            还是要在Server端学习到一个鲁棒的模型，所以在做测试的时候，是在Server端进行的
        '''

        global_loss, global_acc = test_accuracy(net, global_parameters, testDataLoader)

        if round > 10:
            print(round)

        if args['byz_type'] == 'LIT_attack' or args['byz_type'] == 'Scaling_attack':
            global_ASR = test_ASR(net, global_parameters, testDataLoader)
            writer.add_scalar('scalar/global_ASR', global_ASR, round)
            print('[Round: %d] >> Global ASR: %f' % (round, global_ASR))
            logger.info('[Round: %d] >> Global ASR: %f' % (round, global_ASR))

        print('[Round: %d] >> Global Model Test accuracy: %f' % (round, global_acc))
        print('[Round: %d] >> Global Model Test loss: %f' % (round, global_loss))
        logger.info('[Round: %d] >> Global Model Test accuracy: %f' % (round, global_acc))
        logger.info('[Round: %d] >> Global Model Test loss: %f' % (round, global_loss))
        writer.add_scalar('scalar/Test_Accuracy', global_acc, round)
        writer.add_scalar('scalar/Test_Loss', global_loss, round)
        writer.add_scalar('scalar/learning_rate', opti.param_groups[0]["lr"], round)


    writer.close()
    sys.exit(0)

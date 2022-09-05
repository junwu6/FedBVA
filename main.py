import copy
import numpy as np
import torch
import time
import os

from sys import exit
from torchvision import datasets, transforms

from models.networks import CNNMnist, CNNCifar
from models.Updates import ServerUpdate, BVDClientUpdate
from models.Test import test_attack, test_blackbox_attack
from models.BVDAttack import get_BVD_adv_examples, get_clean_adv_examples, get_adv_examples
from utils.params import args_parser
from utils.sampling import BVD_iid_sampling, BVD_noniid_sampling, BVD_iid_sampling_uniform
from utils.Timer import timer
from utils.save_result import save_to_csv

# blackbox models for MNIST and Fashion-MNIST
from models.networks import MNIST_Net_A, MNIST_Net_B, MNIST_Net_C, MNIST_Net_D

# blackbox models for CIFAR-10 and CIFAR-100
from networks.resnet import resnet18
from networks.vgg import vgg11_bn
from networks.xception import xception
from networks.mobilenetv2 import mobilenetv2

def run_main(method_name):
    # parse args
    args = args_parser()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    args.device = torch.device("cuda: 0" if use_cuda else "cpu")

    if method_name == "FedAvg":
        args.ds_type = 0
        args.local_adt = False
    elif method_name == "FedAvg_AT":
        args.ds_type = 2
        args.local_adt = False
    elif method_name == "Fed_Bias":
        args.ds_type = 1
        args.local_adt = False
        args.c2 = 100000
    elif method_name == "Fed_Variance":
        args.ds_type = 1
        args.local_adt = False
        args.c2 = -100000
    elif method_name == "Fed_BVA":
        args.ds_type = 1
        args.local_adt = False
        args.c2 = 0.01
    elif method_name == "EAT":
        args.ds_type = 0
        args.local_adt = True
    elif method_name == "EAT+Fed_BVA":
        args.ds_type = 1
        args.local_adt = True
        args.c2 = 0.01

    dataset_train, dataset_test, dict_users, dict_server = None, None, None, None
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor()])
        dataset_train = datasets.MNIST('data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('data/mnist/', train=False, download=True, transform=trans_mnist)
    elif args.dataset == 'fashion-mnist':
        trans_fashion_mnist = transforms.Compose([transforms.ToTensor()])
        dataset_train = datasets.FashionMNIST('data/fashion_mnist/', train=True, download=True, transform=trans_fashion_mnist)
        dataset_test = datasets.FashionMNIST('data/fashion_mnist/', train=False, download=True, transform=trans_fashion_mnist)
    elif args.dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        dataset_train = datasets.CIFAR10(root='./data/cifar10/', train=True, download=True, transform=transform_train)

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        dataset_test = datasets.CIFAR10(root='./data/cifar10/', train=False, download=True, transform=transform_test)
    elif args.dataset == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
        ])
        dataset_train = datasets.CIFAR100(root='data/cifar100/', train=True, download=True, transform=transform_train)

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
        ])
        dataset_test = datasets.CIFAR100(root='data/cifar100/', train=False, download=True, transform=transform_test)
    else:
        exit('Error: unrecognized dataset')

    if args.iid and (args.dataset == 'mnist' or args.dataset == 'fashion-mnist'):
        dict_server, dict_users = BVD_iid_sampling(dataset_train, args)
    elif args.iid and (args.dataset == 'cifar10' or args.dataset == 'cifar100'):
        dict_server, dict_users = BVD_iid_sampling_uniform(dataset_train, args)
    else:
        dict_server, dict_users = BVD_noniid_sampling(dataset_train, args)

    global_model = None
    if args.model == "cnn" and (args.dataset == 'mnist' or args.dataset == 'fashion-mnist'):
        global_model = CNNMnist(args=args).to(args.device)
    elif args.model == "cnn" and (args.dataset == 'cifar10' or args.dataset == 'cifar100'):
        global_model = CNNCifar(args).to(args.device)
    else:
        exit('Error: unrecognized model..')

    # use the same initialization for all experiments
    if not os.path.isfile("save/{}_{}_initialization.pt".format(args.model, args.dataset)):
        torch.save(global_model.state_dict(), "save/{}_{}_initialization.pt".format(args.model, args.dataset))
    else:
        global_model.load_state_dict(torch.load("save/{}_{}_initialization.pt".format(args.model, args.dataset)))
    global_model.train()

    perturbed_data = None
    avg_run_time = []
    avg_server_time, avg_client_time = [], []
    results = []
    for iter in range(args.epochs):
        result_epoch = []
        start = time.time()
        client_models = []
        w_locals, loss_locals = [], []
        num_selected_clients = max(int(args.frac * args.num_clients), 1)
        idxs_users = np.random.choice(range(args.num_clients), num_selected_clients, replace=False)
        client_start = time.time()
        for idx in idxs_users:
            local = BVDClientUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx], perturbed_data=perturbed_data)
            w, loss, client_model = local.train(net=copy.deepcopy(global_model).to(args.device))
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
            client_models.append(client_model)
        avg_client_time.append(time.time() - client_start)
        # update global weights
        server_start = time.time()
        w_glob = ServerUpdate(w_locals)
        global_model.load_state_dict(w_glob)

        if args.ds and args.ds_type==1: # generate the shared data D_s (BVD generated)
            perturbed_data = get_BVD_adv_examples(copy.deepcopy(client_models), dataset=dataset_train, shared_idx=dict_server, args=args)
        elif args.ds and args.ds_type==0: # generate the shared data D_s (clean)
            perturbed_data = get_clean_adv_examples(dataset=dataset_train, shared_idx=dict_server, args=args)
        elif args.ds and args.ds_type==2: # generate the shared data D_s (AT Baseline, using aggregated model to generate)
            perturbed_data = get_adv_examples(copy.deepcopy(global_model), dataset=dataset_train, shared_idx=dict_server, args=args)

        avg_server_time.append(time.time() - server_start)
        avg_run_time.append(time.time() - start)
        timer(start, time.time())

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}, Method {}'.format(iter, loss_avg, method_name))

        if (iter+1) % 10 == 0:
            torch.save(global_model.state_dict(), "save/SGD_{}_iid({})_{}_epoch_{}.pt".format(args.dataset, args.iid, method_name, iter+1))

            print("\n================================ FGSM ===================================")
            result_epoch += test_attack(copy.deepcopy(global_model), dataset_test, args, attack="FGSM")
            print("\n=============================== PGD-10 ===================================")
            result_epoch += test_attack(copy.deepcopy(global_model), dataset_test, args, attack="PGD-10")
            print("\n=============================== PGD-20 ===================================")
            result_epoch += test_attack(copy.deepcopy(global_model), dataset_test, args, attack="PGD-20")
            print("\n")

            print("\n************** Start Testing Blackbox Attacks **************")
            blackbox_models = []
            for im in range(4):
                if args.dataset == 'mnist' or args.dataset == 'fashion-mnist':
                    if im == 0:
                        blackbox_models.append(MNIST_Net_A())
                        blackbox_models[im].load_state_dict(torch.load(args.blackbox_model_path + 'model_A'))
                    elif im == 1:
                        blackbox_models.append(MNIST_Net_B())
                        blackbox_models[im].load_state_dict(torch.load(args.blackbox_model_path + 'model_B'))
                    elif im == 2:
                        blackbox_models.append(MNIST_Net_C())
                        blackbox_models[im].load_state_dict(torch.load(args.blackbox_model_path + 'model_C'))
                    elif im == 3:
                        blackbox_models.append(MNIST_Net_D())
                        blackbox_models[im].load_state_dict(torch.load(args.blackbox_model_path + 'model_D'))
                elif args.dataset == 'cifar10' or args.dataset == 'cifar100':
                    if im == 0:
                        blackbox_models.append(resnet18())
                        blackbox_models[im].load_state_dict(torch.load(args.blackbox_model_path + 'model_A'))
                    elif im == 1:
                        blackbox_models.append(vgg11_bn())
                        blackbox_models[im].load_state_dict(torch.load(args.blackbox_model_path + 'model_B'))
                    elif im == 2:
                        blackbox_models.append(xception())
                        blackbox_models[im].load_state_dict(torch.load(args.blackbox_model_path + 'model_C'))
                    elif im == 3:
                        blackbox_models.append(mobilenetv2())
                        blackbox_models[im].load_state_dict(torch.load(args.blackbox_model_path + 'model_D'))

            print("\n================================ FGSM  ===================================")
            result_epoch += test_blackbox_attack(blackbox_models, global_model, dataset_test, args, attack="FGSM")
            print("\n=============================== PGD-10  ===================================")
            result_epoch += test_blackbox_attack(blackbox_models, global_model, dataset_test, args, attack="PGD-10")
            print("\n=============================== PGD-20  ===================================")
            result_epoch += test_blackbox_attack(blackbox_models, global_model,  dataset_test, args, attack="PGD-20")
            print("\n")
            results.append(result_epoch)
    print("running time:", np.mean(avg_run_time), np.std(avg_run_time))
    print("Server running time:", np.mean(avg_server_time), np.std(avg_server_time))
    print("Client running time:", np.mean(avg_client_time), np.std(avg_client_time))
    save_to_csv(np.array(results), save_name='save/SGD_{}_iid({})_{}'.format(args.dataset, args.iid, method_name))


if __name__ == '__main__':
    methods = ['FedAvg', 'FedAvg_AT', 'Fed_Bias', 'Fed_Variance', 'Fed_BVA', 'EAT', 'EAT+Fed_BVA']

    for method_name in methods:
        run_main(method_name)

import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=100, help="rounds of training") # default 100
    parser.add_argument('--num_clients', type=int, default=100, help="number of clients: K")  # default 100
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C") # default 0.1
    parser.add_argument('--local_ep', type=int, default=50, help="the number of local epochs: E") # default 50
    parser.add_argument('--local_bs', type=int, default=64, help="local batch size: B") # default 64
    parser.add_argument('--bs', type=int, default=1000, help="test batch size") # default 1000
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate") # default 0.01 (SGD), 0.001(Adam)
    parser.add_argument('--momentum', type=float, default=0.9, help="SGD momentum (default: 0.9)") # default 0.9

    # model arguments
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    parser.add_argument('--num_shared', type=int, default=64, help='number of examples shared across clients') # default 64

    # AT type arguments
    parser.add_argument('--ds', type=bool, default=True, help='add asymmetric shared data set D_s')
    parser.add_argument('--ds_type', type=int, default=0, help='type of D_s: 0 is clean, 1 is BVD based, 2 is AT baseline')
    parser.add_argument('--local_adt', type=bool, default=False, help='local adversarial training')
    parser.add_argument('--blackbox_model_path', type=str, default='save/mnist_blackbox/')

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--iid', type=bool, default=True, help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=1, help="number of channels of imges")
    parser.add_argument('--no_cuda', type=bool, default=False, help="GPU or not")
    parser.add_argument('--verbose', type=bool, default=True, help='verbose print')
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
    args = parser.parse_args()
    return args

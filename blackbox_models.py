from __future__ import print_function
import argparse
import time
import torch
import torch.nn.functional as F
import torch.optim as optim

from sys import exit
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms

# blackbox models for MNIST and Fashion-MNIST
from models.networks import MNIST_Net_A, MNIST_Net_B, MNIST_Net_C, MNIST_Net_D

# blackbox models for CIFAR-10 and CIFAR-100
from networks.resnet import resnet18
from networks.vgg import vgg11_bn
from networks.xception import xception
from networks.mobilenetv2 import mobilenetv2


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="mnist", help="dataset")
    parser.add_argument('--model_name', type=str, default="model_A", help="Model name")
    parser.add_argument('--epochs', type=int, default=100, help="rounds of training")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--test_batch_size', type=int, default=1000)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--no_cuda', type=bool, default=False, help="GPU or not")
    parser.add_argument('--save_model', type=bool, default=True, help='Save model')
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
    args = parser.parse_args()
    return args


def timer(start,end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Elapsed time is: {:0>2}:{:0>2}:{:05.2f}\n".format(int(hours),int(minutes),seconds))


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':
    args = args_parser()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda: 0" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor()])
        trainset = datasets.MNIST('data/mnist/', train=True, download=True, transform=trans_mnist)
        testset  = datasets.MNIST('data/mnist/', train=False, download=True, transform=trans_mnist)
    elif args.dataset == 'fashion-mnist':
        trans_fashion_mnist = transforms.Compose([transforms.ToTensor()])
        trainset = datasets.FashionMNIST('data/fashion_mnist/', train=True, download=True, transform=trans_fashion_mnist)
        testset  = datasets.FashionMNIST('data/fashion_mnist/', train=False, download=True, transform=trans_fashion_mnist)
    elif args.dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = datasets.CIFAR10(root='./data/cifar10/', train=True, download=True, transform=transform_train)

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        testset = datasets.CIFAR10(root='./data/cifar10/', train=False, download=True, transform=transform_test)
    elif args.dataset == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
        ])
        trainset = datasets.CIFAR100(root='data/cifar100/', train=True, download=True, transform=transform_train)

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
        ])
        testset = datasets.CIFAR100(root='data/cifar100/', train=False, download=True, transform=transform_test)
    else:
        exit('Error: unrecognized dataset')

    model = None
    if args.dataset == 'mnist' or args.dataset == 'fashion-mnist':
        if args.model_name == 'model_A':
            model = MNIST_Net_A().to(device)
        elif args.model_name == 'model_B':
            model = MNIST_Net_B().to(device)
        elif args.model_name == 'model_C':
            model = MNIST_Net_C().to(device)
        elif args.model_name == 'model_D':
            model = MNIST_Net_D().to(device)
    elif args.dataset == 'cifar10' or args.dataset == 'cifar100':
        if args.model_name == 'model_A':
            model = resnet18().to(device)
        elif args.model_name == 'model_B':
            model = vgg11_bn().to(device)
        elif args.model_name == 'model_C':
            model = xception().to(device)
        elif args.model_name == 'model_D':
            model = mobilenetv2().to(device)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, **kwargs)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True, **kwargs)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        timer(start_time, time.time())

    if args.save_model:
        # change the folder path name if necessary
        torch.save(model.state_dict(), "save/mnist_blackbox/" + args.model_name)

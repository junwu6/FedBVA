import copy
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from models.Attacks import get_perturbed_image


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


def ServerUpdate(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


class BVDClientUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, perturbed_data=None):
        self.args = args
        self.client_attack = args.local_adt # local adversarial training flag
        self.perturbed_data = perturbed_data
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def train(self, net):
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        # optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                if self.client_attack: # local perturbation
                    image_range = [images.min(), images.max()]
                    perturbed_images = get_perturbed_image(copy.deepcopy(net), images, labels, image_range=image_range, epsilon=0.3)

                net.zero_grad()
                if self.client_attack:
                    # # PGD version, use perturbed data only
                    # loss = self.loss_func(net(perturbed_images), labels)

                    # EAT version, use both clean and perturbed data
                    loss = self.loss_func(net(images), labels) + self.loss_func(net(perturbed_images), labels)
                else:
                    loss = self.loss_func(net(images), labels)

                if self.perturbed_data is not None:
                    images, labels = self.perturbed_data[0].to(self.args.device), self.perturbed_data[1].to(self.args.device)
                    net.zero_grad()
                    loss += self.loss_func(net(images), labels)

                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), net

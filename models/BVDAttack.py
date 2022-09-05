from torch.utils.data import DataLoader
from models.Updates import DatasetSplit
import torch.nn.functional as F
import torch
from models.Attacks import fgsm_attack, get_perturbed_image
import copy


def get_BIAS(client_models, data, target):
    data_grads, output_mean = [], []
    for im, model in enumerate(client_models):
        model_copy = copy.deepcopy(model)
        data_clone, target_clone = data.clone(), target.clone()
        data_clone.requires_grad = True
        output = model_copy(data_clone)
        output_mean.append(F.softmax(output, dim=1))

        loss = F.cross_entropy(output, target_clone)
        model_copy.zero_grad()
        loss.backward()
        data_grad = data_clone.grad.data
        data_grads.append(data_grad)
    output_mean = torch.mean(torch.stack(output_mean, 0), 0)
    data_grads_bias = torch.mean(torch.stack(data_grads, 0), 0)
    return data_grads_bias, output_mean.cpu().detach().numpy()


def get_VARIANCE(client_models, data, output_mean):
    batch_size = data.shape[0]
    data_grads_variance = list()
    for im, model in enumerate(client_models):
        data_clone = data.clone()
        data_clone.requires_grad = True
        output = F.softmax(model(data_clone), dim=1)

        data_grads_variance_subset = 0
        data_grads_previous = 0
        for ic in range(output.shape[1]):  # output.shape = [1, 10] for MNIST
            variance_multiplier = torch.log(output_mean[:, ic]) + 1
            for ib in range(batch_size):
                model.zero_grad()
                output[ib, ic].backward(retain_graph=True)
                data_grad = data_clone.grad.data - data_grads_previous
                data_grads_previous = copy.deepcopy(data_clone.grad.data)
                data_grads_variance_subset += data_grad * variance_multiplier.view(batch_size, 1, 1, 1).repeat(1, 1, 28, 28)
        data_grads_variance.append(data_grads_variance_subset)

    data_grads_variance = -1 * torch.mean(torch.stack(data_grads_variance, 0), 0)
    return data_grads_variance


def get_BVD_adv_examples(client_models, dataset, shared_idx, args, epsilon=0.3):
    adv_loader = DataLoader(DatasetSplit(dataset, shared_idx), batch_size=len(shared_idx), shuffle=False)

    perturbed_data, perturbed_target = None, None
    for data, target in adv_loader:
        image_range = [data.min(), data.max()]
        data, target = data.to(args.device), target.to(args.device)
        if args.c2 > 100:
            data_grads, _ = get_BIAS(client_models, data, target)
        elif args.c2 < -100:
            _, output_mean = get_BIAS(client_models, data, target)
            output_mean = torch.from_numpy(output_mean).float().to(args.device)
            data_grads = get_VARIANCE(client_models, data, output_mean)
        else:
            data_grads_bias, output_mean = get_BIAS(client_models, data, target)
            output_mean = torch.from_numpy(output_mean).float().to(args.device)
            data_grads_variance = get_VARIANCE(client_models, data, output_mean)
            data_grads = data_grads_bias + args.c2 * data_grads_variance

        perturbed_data = fgsm_attack(data, epsilon, data_grads, image_range=image_range)
        perturbed_target = target

    return perturbed_data, perturbed_target


def get_adv_examples(model, dataset, shared_idx, args, epsilon=0.3):
    adv_loader = DataLoader(DatasetSplit(dataset, shared_idx), batch_size=len(shared_idx), shuffle=False)

    perturbed_data, perturbed_target = None, None
    for data, target in adv_loader:
        image_range = [data.min(), data.max()]
        data, target = data.to(args.device), target.to(args.device)
        perturbed_data = get_perturbed_image(model, data, target, image_range=image_range, epsilon=epsilon)
        perturbed_target = target
    return perturbed_data, perturbed_target


def get_clean_adv_examples(dataset, shared_idx, args):
    adv_loader = DataLoader(DatasetSplit(dataset, shared_idx), batch_size=len(shared_idx), shuffle=False)

    perturbed_data, perturbed_target = None, None
    for data, target in adv_loader:
        data, target = data.to(args.device), target.to(args.device)
        perturbed_data = data
        perturbed_target = target
    return perturbed_data, perturbed_target


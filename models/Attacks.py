import torch
import torch.nn.functional as F


def fgsm_attack(image, epsilon, data_grad, image_range=None):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, image_range[0], image_range[1])
    return perturbed_image.detach()


def get_perturbed_image(model, data, target, image_range=None, epsilon=0.3):
    data_clone = data.clone()
    data_clone.requires_grad = True
    output = model(data_clone)
    loss = F.cross_entropy(output, target)
    model.zero_grad()
    loss.backward()
    data_grad = data_clone.grad.data
    perturbed_image = fgsm_attack(data, epsilon, data_grad, image_range=image_range)
    return perturbed_image

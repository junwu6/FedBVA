'''
Adapted from https://github.com/Harry24k/adversarial-attacks-pytorch
'''
import torch
import torch.nn as nn


class FGSM(nn.Module):
    def __init__(self, model, eps=0.3):
        super().__init__()
        self.model = model
        self.eps = eps
        self.loss = nn.CrossEntropyLoss()

    def forward(self, images, labels, image_range=None):
        images.requires_grad = True
        outputs = self.model(images)
        cost = self.loss(outputs, labels)
        grad = torch.autograd.grad(cost, images, retain_graph=False, create_graph=False)[0]

        adv_images = images + self.eps * grad.sign()
        adv_images = torch.clamp(adv_images, min=image_range[0], max=image_range[1]).detach()

        return adv_images


class PGD(nn.Module):
    def __init__(self, model, eps=0.3, alpha=0.01, steps=40, random_start=False, targeted=False):
        super().__init__()
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.sign = 1
        if targeted:
            self.sign = -1
        self.loss = nn.CrossEntropyLoss()

    def forward(self, images, labels, image_range=None):
        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1)

        for i in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.model(adv_images)
            cost = self.sign * self.loss(outputs, labels)
            grad = torch.autograd.grad(cost, adv_images, retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=image_range[0], max=image_range[1]).detach()

        return adv_images

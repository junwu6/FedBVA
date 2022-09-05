from torch.utils.data import DataLoader
from utils.Timer import timer
import time
import copy
from models.Test_attacks import FGSM, PGD


def test_attack(net_g, datatest, args, attack="FGSM"):
    net_g.eval()
    data_loader = DataLoader(datatest, batch_size=args.bs, shuffle=False)
    final_acc = []
    for eps in [0, 0.3]:
        start = time.time()
        correct = 0
        evasion_attack = None
        if attack == "FGSM":
            evasion_attack = FGSM(copy.deepcopy(net_g), eps=eps)
        elif attack == "PGD-10" and eps > 0:
            evasion_attack = PGD(copy.deepcopy(net_g), eps=eps, alpha=0.01, steps=10, random_start=True)
        elif attack == "PGD-20" and eps > 0:
            evasion_attack = PGD(copy.deepcopy(net_g), eps=eps, alpha=0.01, steps=20, random_start=True)

        if evasion_attack is not None:
            for idx, (data, target) in enumerate(data_loader):
                data, target = data.to(args.device), target.to(args.device)
                image_range = [data.min(), data.max()]
                perturbed_data = evasion_attack(data, target, image_range=image_range)
                output = net_g(perturbed_data)
                y_pred = output.data.max(1, keepdim=True)[1]
                correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
            acc = correct.detach().cpu().numpy() / float(len(data_loader.dataset))
            print("Epsilon: {}\t Test Accuracy = {}/{} = {:.4f}".format(eps, correct, len(data_loader) * args.bs, acc))
            final_acc.append(acc)
            timer(start, time.time())
    return final_acc


def test_blackbox_attack(blackbox_models, net_g, datatest, args, attack="FGSM"):
    threat_model_names = ['model A', 'model B', 'model C', 'model D', 'model E']
    final_acc = []
    for im in range(len(blackbox_models)):
        model_name = threat_model_names[im]
        net_threat = blackbox_models[im].to(args.device)
        net_threat.eval()

        net_g.eval()
        data_loader = DataLoader(datatest, batch_size=args.bs, shuffle=False)
        for eps in [0.3]:
            start = time.time()
            correct = 0
            evasion_attack = None
            if attack == "FGSM":
                evasion_attack = FGSM(copy.deepcopy(net_threat), eps=eps)
            elif attack == "PGD-10":
                evasion_attack = PGD(copy.deepcopy(net_threat), eps=eps, alpha=0.01, steps=10, random_start=True)
            elif attack == "PGD-20":
                evasion_attack = PGD(copy.deepcopy(net_threat), eps=eps, alpha=0.01, steps=20, random_start=True)

            for idx, (data, target) in enumerate(data_loader):
                data, target = data.to(args.device), target.to(args.device)
                image_range = [data.min(), data.max()]
                perturbed_data = evasion_attack(data, target, image_range=image_range)
                output = net_g(perturbed_data)
                y_pred = output.data.max(1, keepdim=True)[1]
                correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
            acc = correct.detach().cpu().numpy() / float(len(data_loader.dataset))
            print("Thread {}\t Epsilon: {}\t Test Accuracy = {}/{} = {:.4f}".format(model_name, eps, correct, len(data_loader) * args.bs, acc))
            final_acc.append(acc)
            timer(start, time.time())

    return final_acc

import numpy as np
import copy


def BVD_iid_sampling(dataset, args):
    np.random.seed(2020)
    num_clients = args.num_clients
    num_share_data = args.num_shared

    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    dict_server = set(np.random.choice(all_idxs, num_share_data, replace=False))
    num_items = int((len(dataset) - len(dict_server)) / num_clients)

    # Ds and Dk will have no overlaps
    all_idxs = list(set(all_idxs) - dict_server)

    for i in range(num_clients):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_server, dict_users


def BVD_iid_sampling_uniform(dataset, args):
    np.random.seed(2020)
    num_clients = args.num_clients

    dict_users, all_idxs = {}, [i for i in range(len(dataset))]

    # make Ds has one perturb sample per class
    labels = np.array(dataset.train_coarse_labels)
    idxs_labels = np.vstack((all_idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    all_idxs = idxs_labels[0, :]

    split_indices, _ = np.histogram(labels, bins=np.array(args.num_classes))
    split_indices = np.cumsum(split_indices)[:-1]
    split_indices = np.array([0] + list(split_indices))

    # k shared samples per class in D_s
    num_repeat = 3
    split_indices_repeat = copy.deepcopy(split_indices)
    for k in range(1, num_repeat, 1):
        split_indices_repeat = np.concatenate((split_indices_repeat, split_indices + k), axis=0)

    dict_server = set(all_idxs[split_indices_repeat])
    num_items = int((len(dataset) - len(dict_server)) / num_clients)

    # Ds and Dk will have no overlaps
    all_idxs = list(set(all_idxs) - dict_server)

    for i in range(num_clients):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_server, dict_users, labels


def BVD_noniid_sampling(dataset, args):
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(args.num_clients)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.targets.numpy()
    dict_server = set(np.random.choice(idxs, args.num_shared, replace=False))

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(args.num_clients):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
        # make sure Ds and Dk has no overlap
        dict_users[i] = np.array(list(set(dict_users[i]) - dict_server))

    return dict_server, dict_users



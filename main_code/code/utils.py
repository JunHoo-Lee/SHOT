import copy
import torch.nn.functional as F
import torch
from torchmeta.modules import DataParallel as DDP
import torch.nn as nn
import torchvision.transforms as T
import yaml
from torchmeta.utils.data.dataloader import BatchMetaDataLoader
import csv
import os
from model import (
    ConvNet,
    SeparatedConvNet,
    WarpedConvNet,
    BasicBlock,
    BasicBlockWithoutResidual,
    ResNet,
)
from torchmeta.datasets.helpers import (
    miniimagenet,
    tieredimagenet,
    cifar_fs,
    fc100,
    cub,
    vgg_flower,
    aircraft,
    traffic_sign,
    svhn,
    cars,
)
from collections import OrderedDict
from torchmeta.modules import MetaModule


class Augment:
    """
    A stochastic data augmentation module
    Transforms any given data example randomly
    resulting in two correlated views of the same example,
    denoted x ̃i and x ̃j, which we consider as a positive pair.
    """

    def __init__(self, img_size, s=1):
        color_jitter = T.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        # 10% of the image
        blur = T.GaussianBlur((3, 3), (0.1, 2.0))

        self.train_transform = T.Compose(
            [
                T.RandomResizedCrop(size=img_size),
                T.RandomHorizontalFlip(p=0.5),  # with 0.5 probability
                T.RandomApply([color_jitter], p=0.8),
                T.RandomApply([blur], p=0.5),
                T.RandomGrayscale(p=0.2),
                # imagenet stats
                # T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.train_transform1 = T.Compose(
            [
                T.RandomResizedCrop(size=img_size),
                T.RandomHorizontalFlip(p=0.5),  # with 0.5 probability
                T.RandomApply([color_jitter], p=0.8),
                T.RandomApply([blur], p=0.5),
                T.RandomGrayscale(p=0.2),
                # imagenet stats
                # T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.test_transform = T.Compose(
            [
                # T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __call__(self, x):
        return self.train_transform(x), self.train_transform1(x), self.test_transform(x)


def load_dataset(args, mode):
    folder = args.folder
    ways = args.num_ways
    shots = args.num_shots
    test_shots = 15
    download = args.download
    shuffle = True

    if mode == "meta_train":
        args.meta_train = True
        args.meta_val = False
        args.meta_test = False
    elif mode == "meta_valid":
        args.meta_train = False
        args.meta_val = True
        args.meta_test = False
    elif mode == "meta_test":
        args.meta_train = False
        args.meta_val = False
        args.meta_test = True
    else:
        print("there is some error")

    if args.dataset == "miniimagenet":
        dataset = miniimagenet(
            folder=folder,
            shots=shots,
            ways=ways,
            shuffle=shuffle,
            test_shots=test_shots,
            meta_train=args.meta_train,
            meta_val=args.meta_val,
            meta_test=args.meta_test,
            download=download,
        )
    elif args.dataset == "tieredimagenet":
        dataset = tieredimagenet(
            folder=folder,
            shots=shots,
            ways=ways,
            shuffle=shuffle,
            test_shots=test_shots,
            meta_train=args.meta_train,
            meta_val=args.meta_val,
            meta_test=args.meta_test,
            download=download,
        )
    elif args.dataset == "cifar_fs":
        dataset = cifar_fs(
            folder=folder,
            shots=shots,
            ways=ways,
            shuffle=shuffle,
            test_shots=test_shots,
            meta_train=args.meta_train,
            meta_val=args.meta_val,
            meta_test=args.meta_test,
            download=download,
        )
    elif args.dataset == "fc100":
        dataset = fc100(
            folder=folder,
            shots=shots,
            ways=ways,
            shuffle=shuffle,
            test_shots=test_shots,
            meta_train=args.meta_train,
            meta_val=args.meta_val,
            meta_test=args.meta_test,
            download=download,
        )
    elif args.dataset == "cub":
        dataset = cub(
            folder=folder,
            shots=shots,
            ways=ways,
            shuffle=shuffle,
            test_shots=test_shots,
            meta_train=args.meta_train,
            meta_val=args.meta_val,
            meta_test=args.meta_test,
            download=download,
        )
    elif args.dataset == "vgg_flower":
        dataset = vgg_flower(
            folder=folder,
            shots=shots,
            ways=ways,
            shuffle=shuffle,
            test_shots=test_shots,
            meta_train=args.meta_train,
            meta_val=args.meta_val,
            meta_test=args.meta_test,
            download=download,
        )
    elif args.dataset == "aircraft":
        dataset = aircraft(
            folder=folder,
            shots=shots,
            ways=ways,
            shuffle=shuffle,
            test_shots=test_shots,
            meta_train=args.meta_train,
            meta_val=args.meta_val,
            meta_test=args.meta_test,
            download=download,
        )
    elif args.dataset == "traffic_sign":
        dataset = traffic_sign(
            folder=folder,
            shots=shots,
            ways=ways,
            shuffle=shuffle,
            test_shots=test_shots,
            meta_train=args.meta_train,
            meta_val=args.meta_val,
            meta_test=args.meta_test,
            download=download,
        )
    elif args.dataset == "svhn":
        dataset = svhn(
            folder=folder,
            shots=shots,
            ways=ways,
            shuffle=shuffle,
            test_shots=test_shots,
            meta_train=args.meta_train,
            meta_val=args.meta_val,
            meta_test=args.meta_test,
            download=download,
        )
    elif args.dataset == "cars":
        dataset = cars(
            folder=folder,
            shots=shots,
            ways=ways,
            shuffle=shuffle,
            test_shots=test_shots,
            meta_train=args.meta_train,
            meta_val=args.meta_val,
            meta_test=args.meta_test,
            download=download,
        )

    return dataset


def load_model(args):
    if (
        args.dataset == "miniimagenet"
        or args.dataset == "tieredimagenet"
        or args.dataset == "cub"
        or args.dataset == "cars"
    ):
        wh_size = 5
    elif (
        args.dataset == "cifar_fs"
        or args.dataset == "fc100"
        or args.dataset == "vgg_flower"
        or args.dataset == "aircraft"
        or args.dataset == "traffic_sign"
        or args.dataset == "svhn"
    ):
        wh_size = 2
    else:
        raise ValueError("Unknown dataset")

    if args.model == "4conv":
        model = ConvNet(
            in_channels=3,
            out_features=args.num_ways,
            hidden_size=args.hidden_size,
            wh_size=wh_size,
        )
    elif args.model == "4conv_sep":
        model = SeparatedConvNet(
            in_channels=3,
            out_features=args.num_ways,
            hidden_size=args.hidden_size,
            wh_size=wh_size,
        )
    elif args.model == "resnet":
        if args.blocks_type == "a":
            blocks = [BasicBlock, BasicBlock, BasicBlock, BasicBlock]
        elif args.blocks_type == "b":
            blocks = [BasicBlock, BasicBlock, BasicBlock, BasicBlockWithoutResidual]
        elif args.blocks_type == "c":
            blocks = [
                BasicBlock,
                BasicBlock,
                BasicBlockWithoutResidual,
                BasicBlockWithoutResidual,
            ]
        elif args.blocks_type == "d":
            blocks = [
                BasicBlock,
                BasicBlockWithoutResidual,
                BasicBlockWithoutResidual,
                BasicBlockWithoutResidual,
            ]
        elif args.blocks_type == "e":
            blocks = [
                BasicBlockWithoutResidual,
                BasicBlockWithoutResidual,
                BasicBlockWithoutResidual,
                BasicBlockWithoutResidual,
            ]

        model = ResNet(
            blocks=blocks,
            keep_prob=1.0,
            avg_pool=True,
            drop_rate=0.0,
            out_features=args.num_ways,
            wh_size=1,
        )
    return model


def update_parameters(
    args,
    model,
    loss,
    params=None,
    amplitude=1,
    first_order=False,
):
    if not isinstance(model, MetaModule):
        raise ValueError(
            "The model must be an instance of `torchmeta.modules."
            "MetaModule`, got `{0}`".format(type(model))
        )

    if params is None:
        params = OrderedDict(model.meta_named_parameters())
    step_size = None

    grads = torch.autograd.grad(
        loss, params.values(), create_graph=not first_order, allow_unused=True
    )

    updated_params = OrderedDict()
    grads_norm = 0

    if isinstance(step_size, (dict, OrderedDict)):
        for (name, param), grad in zip(params.items(), grads):
            if grad != None and (args.pretrain == False or name != "classifier.weight"):
                updated_params[name] = param - step_size[name] * grad * amplitude
                if step_size[name] > 0:
                    grads_norm += torch.norm(grad) ** 2

    else:
        for (name, param), grad in zip(params.items(), grads):
            if grad != None:
                updated_params[name] = param - (amplitude * args.step_size * grad)
    return updated_params


def get_accuracy(logits, targets):
    """Compute the accuracy (after adaptation) of MAML on the test/query points

    Parameters
    ----------
    logits : `torch.FloatTensor` instance
        Outputs/logits of the model on the query points. This tensor has shape
        `(num_examples, num_classes)`.
    targets : `torch.LongTensor` instance
        A tensor containing the targets of the query points. This tensor has
        shape `(num_examples,)`.

    Returns
    -------
    accuracy : `torch.FloatTensor` instance
        Mean accuracy on the query points
    """
    _, predictions = torch.max(logits, dim=-1)
    return torch.mean(predictions.eq(targets).float())


class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum()
        return b


def save_results(args, results, filename="results.csv"):
    """
    I want to save the results with csv file
    results are dict type
    """

    dataset_list = ["miniimagenet", "cars"]
    mode = "a"
    fieldnames = ["save_dir", "meta_lr", "step_size", "SHOTlr"]
    fieldnames.extend(dataset_list)
    filename = os.path.join("csv", filename)
    if not os.path.exists(filename):
        mode = "w"
    with open(filename, mode) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if f.tell() == 0:
            writer.writeheader()
        merged_results = {**args.__dict__, **results}
        subdict = {key: merged_results[key] for key in fieldnames}
        # results.update(subdict)
        writer.writerow(subdict)


def load_data(args, mode):
    """
    In this function, you should load the dataset, create the
    dataloader, and return the dataloader.
    config needs following attributes:
    - dataset
    - folder
    - num_ways
    - num_shots
    - download
    - batch_size
    - num_workers
    """
    dataset = load_dataset(args, mode)
    dataloader = BatchMetaDataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )
    return dataloader


def setup_model(args):
    """
    In this function, you should load the model, define the optimizer
    and move the model to the appropriate device.
    config needs following attributes:
    - model_name
    - dataset
    - device
    - meta_lr
    """
    model = load_model(args)

    if args.multi_gpu:
        from torchmeta.modules import DataParallel as DDP

        model = DDP(model)

    model.to(args.device)

    head_params = [p for name, p in model.named_parameters() if "classifier" in name]
    body_params = [
        p for name, p in model.named_parameters() if "classifier" not in name
    ]
    meta_optimizer = torch.optim.Adam(
        [
            {"params": head_params, "lr": args.meta_lr},
            {"params": body_params, "lr": args.meta_lr},
        ]
    )
    pretrain_optimizer = torch.optim.Adam(
        [
            {"params": body_params, "lr": args.meta_lr},
        ]
    )

    optimizer = {}
    optimizer["meta_optimizer"] = meta_optimizer
    optimizer["pretrain_optimizer"] = pretrain_optimizer

    return model, optimizer

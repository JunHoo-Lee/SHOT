from typing import OrderedDict
from utils import (
    load_dataset,
    load_model,
    update_parameters,
    get_accuracy,
    save_results,
    load_data,
    setup_model,
)
from algorithm import update, inner_loop_update
import torch
import os
import torch.nn.functional as F
import torch.nn as nn
from torchmeta.utils.data.dataloader import BatchMetaDataLoader
from tqdm import tqdm
import numpy as np
import argparse


def natural_optimizer(args, model, support_input, support_target):
    # pip install https://github.com/YiwenShaoStephen/NGD-SGD.git
    from ngd import NGD

    optimizer = NGD(model.parameters(), lr=args.step_size)

    for i in range(args.inner_update_num):
        optimizer.zero_grad()
        logit = model(support_input, prefix=args.prefix)
        loss = F.cross_entropy(logit, support_target)
        loss.backward()
        optimizer.step()

    return model


def train(args, model, meta_optimizer):
    """
    In this function, you should train the model.
    which means, we should we should load dataloader here, then
    """
    dataloader = load_data(args, mode="meta_train")
    model.train()

    pbar = tqdm(
        dataloader, total=args.train_batches, desc="Training", position=1, leave=False
    )

    for batch_idx, batch in enumerate(dataloader):
        support_inputs, support_targets = batch["train"]
        support_inputs = support_inputs.to(device=args.device)
        support_targets = support_targets.to(device=args.device)

        query_inputs, query_targets = batch["test"]
        query_inputs = query_inputs.to(device=args.device)
        query_targets = query_targets.to(device=args.device)

        accuracy = torch.tensor(0.0).to(args.device)
        outer_loss = torch.tensor(0.0).to(args.device)

        for _, (
            support_input,
            support_target,
            query_input,
            query_target,
        ) in enumerate(
            zip(
                support_inputs,
                support_targets,
                query_inputs,
                query_targets,
            )
        ):
            outer_loss_b, accuracy_b = update(
                args,
                model,
                support_input,
                support_target,
                query_input,
                query_target,
            )
            outer_loss += outer_loss_b
            accuracy += accuracy_b

        accuracy.div_(args.batch_size)
        outer_loss.div_(args.batch_size)

        model.zero_grad()
        meta_optimizer.zero_grad()
        outer_loss.backward()
        meta_optimizer.step()

        """
        TODO: Meta optimizer의 경우 algorithm에 따라서 따로 만들게 해야 함
        Pretrin method의 경우 만들고, 불러오는 위치를 다르게?
        
        """

        postfix = {"loss": outer_loss.item(), "accuracy": accuracy.item()}
        pbar.set_postfix(postfix)
        pbar.update(1)

        if batch_idx + 1 == args.train_batches:
            break
    pbar.close()

    return model, evaluate(model, args), meta_optimizer


def evaluate(model, args, mode="meta_valid"):
    """
    In this function, you should evaluate the model performance on the
    validation or test set.
    """
    dataloader = load_data(args, mode=mode)
    model.eval()
    accuracy_logs = []

    # if mode is test, then desc is "Testing" else "Evaluating"
    if mode == "meta_test":
        desc = "Testing"
        total = args.test_batches
    else:
        desc = "Evaluating"
        total = args.valid_batches

    pbar = tqdm(dataloader, total=total, desc=desc, position=1, leave=False)

    for batch_idx, batch in enumerate(dataloader):
        support_inputs, support_targets = batch["train"]
        support_inputs = support_inputs.to(device=args.device)
        support_targets = support_targets.to(device=args.device)

        query_inputs, query_targets = batch["test"]
        query_inputs = query_inputs.to(device=args.device)
        query_targets = query_targets.to(device=args.device)

        accuracy = torch.tensor(0.0).to(device=args.device)

        for _, (
            support_input,
            support_target,
            query_input,
            query_target,
        ) in enumerate(
            zip(support_inputs, support_targets, query_inputs, query_targets)
        ):
            # inner loop update
            if args.natural_gradient:
                import copy

                adapted_model = copy.deepcopy(model)
                adapted_model = natural_optimizer(
                    args, adapted_model, support_input, support_target
                )
                query_logit = adapted_model(query_input)
                accuracy += get_accuracy(query_logit, query_target)
                del adapted_model

            else:
                model_params = inner_loop_update(
                    args, model, support_input, support_target
                )
                query_logit = model(
                    query_input, params=model_params, prefix=args.prefix
                )
                accuracy += get_accuracy(query_logit, query_target)

        accuracy_logs.append(accuracy.div_(args.batch_size).item())

        postfix = {"accuracy": accuracy_logs[-1]}
        pbar.set_postfix(postfix)
        pbar.update(1)
        if batch_idx + 1 == total:
            break
    pbar.close()

    return sum(accuracy_logs) / len(accuracy_logs)


def save_model(args, model, epoch, best=False):
    """
    In this function, you should save the trained model.
    """
    if args.method == "pretrain":
        fname = "pretrained_epochs_{}.pt".format(epoch * 1)

    else:
        fname = "epochs_{}.pt".format(epoch * 1)

    save_dir = os.path.join(args.output_folder, args.save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    filename = os.path.join(save_dir, fname)
    if best:
        if args.method == "pretrain":
            filename = os.path.join(save_dir, "pretrained_best_model.pt")
        else:
            filename = os.path.join(save_dir, "best_model.pt")

    with open(filename, "wb") as f:
        state_dict = model.state_dict()
        torch.save(state_dict, f)


def load_saved_model(
    args,
    model,
    epoch=0,
    best=False,
):
    """
    In this function, you should load the trained model.
    """
    # Set the default filename
    filename = os.path.join(args.output_folder, args.save_dir, f"epochs_{epoch}.pt")

    # Check if pretrained model should be loaded
    if args.load_pretrained:
        filename = os.path.join(
            args.output_folder, args.save_dir, "pretrained_best_model.pt"
        )
    else:
        bestname = os.path.join(args.output_folder, args.save_dir, "best_model.pt")
        lastname = os.path.join(args.output_folder, args.save_dir, f"epochs_{290}.pt")

        # Check if best model should be loaded
        if os.path.isfile(lastname):
            best = True
            args.test = True
            print("last output exists!, so load best model")

        if best:
            filename = bestname

    try:
        with open(filename, "rb") as f:
            print("you are now loading a model", filename)
            state_dict = torch.load(f, map_location=args.device)
            model.load_state_dict(state_dict)
            if args.resume:
                args.test = True
    except FileNotFoundError:
        print("nod model!")
        pass

    return model, args


def parse_args():
    """
    Main parameters to change:
    num_shots: number of support examples per class
    dataset: miniimagenet, tieredimagenet, cub, cars, omniglot, cifar_fs, aircraft
    method: maml, maml_approx, maml_lil, maml_lil_approx
    save_name: name of the saved model
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder",
        type=str,
        default="../data",
        help="path to the data folder",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="./output",
        help="path to the output folder",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="miniimagenet",
        help="other datasets: miniimagenet, tieredimagenet, \
        cub, cars, omniglot, cifar_fs, aircraft",
    )
    parser.add_argument(
        "--blocks_type",
        type=str,
        default="a",
        help="a, maml_approx, maml_lil, maml_lil_approx",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="maml",
        help="maml, maml_approx, maml_lil, maml_lil_approx",
    )
    parser.add_argument(
        "--num_ways",
        type=int,
        default=5,
        help="number of classes in a classification task",
    )
    parser.add_argument(
        "--MORE",
        action="store_true",
        help="resume from previous state",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume from previous state",
    )
    parser.add_argument(
        "--num_shots",
        type=int,
        default=1,
        help="number of support examples per class",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="number of tasks in a batch",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="number of epochs to train the model",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="number of workers for the dataloader",
    )

    parser.add_argument(
        "--teacher_lr_scale",
        type=int,
        default=2,
        help="scale of teacher step",
    )
    parser.add_argument(
        "--inner_update_num",
        type=int,
        default=3,
        help="number of inner loop updates",
    )
    parser.add_argument(
        "--step_size",
        type=float,
        default=0.5,
        help="step size for the inner loop updates",
    )
    parser.add_argument(
        "--meta_lr",
        type=float,
        default=0.001,
        help="meta learning rate",
    )

    parser.add_argument(
        "--load_pretrained",
        action="store_true",
        help="should I load pretrained..",
    )
    parser.add_argument(
        "--SHOT",
        action="store_true",
        help="whether to use SHOT",
    )
    parser.add_argument(
        "--SHOTlr",
        type=float,
        default=0.1,
        help="step size for the one-step update",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="4conv",
        help="4conv, resnet",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="cuda or cpu",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=64,
        help="number of filters for the conv nets",
    )
    parser.add_argument(
        "--train-batches",
        type=int,
        default=100,
        help="Number of batches the model is trained over (i.e., validation save steps) (default: 100).",
    )
    parser.add_argument(
        "--valid-batches",
        type=int,
        default=25,
        help="Number of batches the model is validated over (default: 25).",
    )
    parser.add_argument(
        "--test-batches",
        type=int,
        default=2500,
        help="Number of batches the model is tested over (default: 2500).",
    )

    parser.add_argument(
        "--csv_name",
        type=str,
        default="lr_SHOT_FINAL.csv",
        help="whether to use first order approximation",
    )

    parser.add_argument(
        "--first_order",
        action="store_true",
        help="whether to use first order approximation",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="maml",
        help="directory to save the model",
    )
    parser.add_argument(
        "--multi_gpu",
        action="store_true",
        help="whether to use multiple GPUs",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="whether to download the dataset",
    )
    parser.add_argument(
        "--augtarget",
        action="store_true",
        help="augment target",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="whether to test the model",
    )
    parser.add_argument(
        "--natural_gradient",
        action="store_true",
        help="whether to use natural gradient",
    )

    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="path to the output folder",
    )

    return parser.parse_args()


def test(args, model):
    original_dataset = args.dataset
    model, _ = load_saved_model(args, model, best=True)
    result = {}
    # dataset_list = ["miniimagenet", "tieredimagenet", "cub", "cars"]
    dataset_list = ["miniimagenet", "cars"]
    for dataset in dataset_list:
        args.dataset = dataset
        result[dataset] = evaluate(model, args, mode="meta_test")
    # restore the original dataset
    args.dataset = original_dataset
    return result


def main(args):
    """
    In this function, you should call the above functions in the
    appropriate order.
    """
    print(args)

    model, optimizer = setup_model(args)
    model, _ = load_saved_model(args, model, best=False)
    meta_optimizer = optimizer["meta_optimizer"]
    args.load_pretrained = False

    max_accuracy = 0.0
    max_epoch = 0
    if args.multi_gpu:
        args.prefix = "module."

    if args.test:
        args.epochs = 0
    elif args.method == "pretrain":
        args.epochs = 30
    pbar = tqdm(total=args.epochs, desc="Epochs", position=0)

    for epoch in range(args.epochs):
        model, valid_accuracy, meta_optimizer = train(args, model, meta_optimizer)

        if valid_accuracy > max_accuracy:
            max_accuracy = valid_accuracy
            max_epoch = epoch
            save_model(args, model, epoch, best=True)
        postfix = {
            "valid_accuracy": valid_accuracy,
            "max_accuracy": max_accuracy,
            "max_epoch": max_epoch,
        }
        if epoch % 10 == 0:
            save_model(args, model, epoch)
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()
    if args.method != "pretrain":
        results = test(args, model)
        print(results)
        save_results(args, results, args.csv_name)


if __name__ == "__main__":
    args = parse_args()
    main(args)

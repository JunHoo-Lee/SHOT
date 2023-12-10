from typing import OrderedDict
import torch
import os
import torch.nn.functional as F
import torch.nn as nn
from torchmeta.utils.data.dataloader import BatchMetaDataLoader
from utils import (
    update_parameters,
    get_accuracy,
    Augment,
)

from tqdm import tqdm
import numpy as np
import argparse


def inner_loop_update(
    args,
    model,
    support_set,
    support_set_labels,
    approx=False,
    teacher=False,
    projection=False,
):
    """
    In this function, you should perform the inner loop update, which
    updates the model's parameters on the support set.
    config needs following attributes:
    - inner_update_num
    - first_order
    """
    params = None

    if approx:
        inner_update_num = 1
        amplitude = args.inner_update_num
    elif teacher:
        inner_update_num = args.inner_update_num * args.teacher_lr_scale
        amplitude = 1 / args.teacher_lr_scale
    else:
        inner_update_num = args.inner_update_num
        amplitude = 1

    for _ in range(inner_update_num):
        if projection:
            if args.multi_gpu:
                support_logits = model.module.student_forward(
                    support_set, params=params, prefix=args.prefix
                )
            else:
                support_logits = model.student_forward(
                    support_set, params=params, prefix=args.prefix
                )
        else:
            support_logits = model(support_set, params=params, prefix=args.prefix)

        inner_loss = F.cross_entropy(support_logits, support_set_labels)
        model.zero_grad()
        params = update_parameters(
            args,
            model,
            inner_loss,
            params=params,
            amplitude=amplitude,
            first_order=args.first_order,
        )

    return params


def update(
    args,
    model,
    support_input,
    support_target,
    query_input,
    query_target,
):
    """
    In this function, we will update the
    """
    method = args.method
    if method == "maml":
        return maml_update(
            args, model, support_input, support_target, query_input, query_target
        )
    elif method == "pretrain":
        return pretrain_update(
            args,
            model,
            support_input,
            support_target,
            query_input,
            query_target,
        )
    else:
        """
        An general algorithm
        """

        raise NotImplementedError


# TODO Add some scripts on comparing large step andsmall step


def pretrain_update(
    args, model, support_input, support_target, query_input, query_target
):
    """
    An algorithm for pretrain
    """
    # set initial loss to 0 which has located in model

    loss = torch.tensor(0.0).to(args.device)
    if args.augtarget:
        augment = Augment(84)
        query_input1, query_input2, _ = augment(query_input)

    if args.augtarget:
        query_input_s = query_input2
        query_input_t = query_input1
    else:
        query_input_s = query_input
        query_input_t = query_input
    if args.MORE:
        query_logit_2 = model(
            query_input_s,
            params=inner_loop_update(
                args,
                model,
                support_input,
                support_target,
            ),
            prefix=args.prefix,
        )
        # one-step approximation
        if args.multi_gpu:
            kl_target = model.module.student_forward(
                query_input_t,
                params=inner_loop_update(
                    args,
                    model,
                    support_input,
                    support_target,
                    teacher=True,
                    projection=True,
                ),
                prefix=args.prefix,
            )
        else:
            kl_target = model(
                query_input_t,
                params=inner_loop_update(
                    args,
                    model,
                    support_input,
                    support_target,
                    teacher=True,
                    projection=True,
                ),
                prefix=args.prefix,
            )
        teacher_prob = F.softmax(kl_target, dim=-1)
        accuracy = get_accuracy(kl_target, query_target)
        # loss += args.SHOTlr * F.cross_entropy(query_logit_2, teacher_prob.detach())
        loss += args.SHOTlr * nn.KLDivLoss(query_logit_2, teacher_prob.detach())

    if args.SHOT:
        if args.multi_gpu:
            query_logit_t = model.module.student_forward(
                query_input_t,
                params=inner_loop_update(
                    args, model, support_input, support_target, projection=True
                ),
                prefix=args.prefix,
            )
        else:
            query_logit_t = model.student_forward(
                query_input_t,
                params=inner_loop_update(
                    args, model, support_input, support_target, projection=True
                ),
                prefix=args.prefix,
            )

        # Compute logits and loss for student model
        query_params = inner_loop_update(
            args,
            model,
            support_input,
            support_target,
            approx=True,
        )
        query_logit_s = model(query_input_s, params=query_params, prefix=args.prefix)
        loss = args.SHOTlr * F.cross_entropy(
            query_logit_s, F.softmax(query_logit_t, dim=-1).detach()
        )
        accuracy = get_accuracy(query_logit_t, query_target)

    return loss, accuracy


# define lambda, if args.multi_gpu: model.module.student_forward else model.student_forward

# lambda_ = lambda x: x.module.student_forward if args.multi_gpu else x.student_forward


def maml_update(args, model, support_input, support_target, query_input, query_target):
    """
    An algorithm for maml
    """

    model_params = inner_loop_update(args, model, support_input, support_target)
    query_logit = model(query_input, params=model_params, prefix=args.prefix)
    outer_loss = F.cross_entropy(query_logit, query_target)
    accuracy = get_accuracy(query_logit, query_target)

    if args.SHOT:
        approx_params = inner_loop_update(
            args,
            model,
            support_input,
            support_target,
            approx=True,
        )
        query_prob = F.softmax(query_logit, dim=1)
        one_step_query_logits = model(
            query_input, params=approx_params, prefix=args.prefix
        )
        one_step_loss_ = F.cross_entropy(one_step_query_logits, query_prob.detach())
        student_val = F.log_softmax(one_step_query_logits, dim=1)
        one_step_loss = F.kl_div(
            student_val, query_prob.detach(), reduction="batchmean"
        )
        outer_loss += args.SHOTlr * one_step_loss
    if args.MORE:
        teacher_params = inner_loop_update(
            args,
            model,
            support_input,
            support_target,
            teacher=True,
        )
        teacher_query_logit = model(
            query_input, params=teacher_params, prefix=args.prefix
        )
        teacher_prob = F.softmax(teacher_query_logit, dim=1)
        outer_loss += args.SHOTlr * F.kl_div(
            F.log_softmax(query_logit, dim=1),
            teacher_prob.detach(),
            reduction="batchmean",
        )
    return outer_loss, accuracy

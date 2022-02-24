# coding=utf-8
# Created by bugface (https://github.com/bugface)
# First created at 2/16/22


def biaffine_loss_calculation(preds, labels, masks, loss_func, num_labels):
    active_idx = masks.view(-1) == 1
    preds = preds.reshape(-1, preds.shape[-1])
    preds_masked = preds[active_idx]
    labels_masked = labels[active_idx]

    loss = loss_func(preds_masked, labels_masked)

    return loss


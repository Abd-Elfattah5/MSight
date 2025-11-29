import numpy as np
import torch
import torch.nn as nn

def dice_loss(pred, target, smooth=1e-7):
    # Assume pred is binary (0 or 1), no sigmoid needed
    pred = pred.flatten(1)
    target = target.flatten(1)

    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)

    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

def iou_loss(pred, target, smooth=1e-7):
    # Assume pred is binary (0 or 1), no sigmoid needed
    pred = pred.flatten(1)
    target = target.flatten(1)

    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1) - intersection

    iou = (intersection + smooth) / (union + smooth)
    return 1 - iou.mean()

def compute_sensitivity(outputs, masks, threshold=0.5, smooth=1e-7):
    preds = outputs  # Assume binary input
    preds = preds.flatten(1).float()
    masks = masks.flatten(1).float()
    
    tp = (preds * masks).sum(dim=1)
    ap = masks.sum(dim=1)
    
    sensitivity = (tp + smooth) / (ap + smooth)
    return sensitivity.mean()

def compute_specificity(outputs, masks, threshold=0.5, smooth=1e-7):
    preds = outputs
    preds = preds.flatten(1).float()
    masks = masks.flatten(1).float()
    
    tn = ((1 - preds) * (1 - masks)).sum(dim=1)
    an = (1 - masks).sum(dim=1)
    
    specificity = (tn + smooth) / (an + smooth)
    return specificity.mean()

def compute_precision(outputs, masks, threshold=0.5, smooth=1e-7):
    preds = outputs
    preds = preds.flatten(1).float()
    masks = masks.flatten(1).float()
    
    tp = (preds * masks).sum(dim=1)
    fp = (preds * (1 - masks)).sum(dim=1)
    
    precision = (tp + smooth) / (tp + fp + smooth)
    return precision.mean()

bce_loss = nn.BCELoss()

def compute_losses(outputs, masks):
    loss_dice = dice_loss(outputs, masks)
    loss_iou = iou_loss(outputs, masks)
    loss_bce = bce_loss(outputs, masks)

    total_loss = 0.1 * loss_bce + 0.4 * loss_dice + 0.5 * loss_iou
    return total_loss, loss_bce, loss_dice, loss_iou
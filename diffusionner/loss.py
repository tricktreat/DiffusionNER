from abc import ABC

import torch
from torch import nn as nn
from torch.nn import functional as F
from .matcher import HungarianMatcher

class Loss(ABC):
    def compute(self, *args, **kwargs):
        pass

class DiffusionNERLoss(Loss):
    def __init__(self, entity_type_count, device, model, optimizer, scheduler, max_grad_norm, nil_weight, match_class_weight, match_boundary_weight, loss_class_weight, loss_boundary_weight, match_boundary_type, type_loss, solver):
        self._model = model
        self._optimizer = optimizer
        self._scheduler = scheduler
        
        # losses = ['labels', 'boundary', 'cardinality']
        losses = ['labels', 'boundary']
        self.weight_dict = {'loss_ce': loss_class_weight, 'loss_boundary': loss_boundary_weight}
        self.criterion = Criterion(entity_type_count, self.weight_dict, nil_weight, losses, type_loss = type_loss, match_class_weight = match_class_weight, match_boundary_weight = match_boundary_weight, match_boundary_type = match_boundary_type, solver = solver)
        self.criterion.to(device)
        self._max_grad_norm = max_grad_norm

    def del_attrs(self):
        del self._optimizer 
        del self._scheduler

    def compute(self, output, gt_types, gt_spans, entity_masks, epoch, batch = None):
        # set_trace()

        gt_types_wo_nil = gt_types.masked_select(entity_masks)
        
        if len(gt_types_wo_nil) == 0:
            return 0.1

        sizes = [i.sum() for i in entity_masks]
        entity_masks = entity_masks.unsqueeze(2).repeat(1, 1, 2)
        spans_wo_nil = gt_spans.masked_select(entity_masks).view(-1, 2)

        targets = {"labels": gt_types_wo_nil, "gt_left":spans_wo_nil[:, 0], "gt_right":spans_wo_nil[:, 1], "sizes":sizes}

        train_loss = []
        indices = None

        pred_logits, pred_left, pred_right, pred_left, pred_right = output["pred_logits"], output["pred_spans"][:, :, 0], output["pred_spans"][:, :, 1], output["pred_left"], output["pred_right"]

        outputs = {"pred_logits":pred_logits, "pred_left":pred_left, "pred_right":pred_right, "pred_left":pred_left, "pred_right":pred_right, "token_mask": batch["token_masks"]}
        loss_dict, indices = self.criterion(outputs, targets, epoch, indices = indices)
        
        train_loss = sum(loss_dict[k] * self.weight_dict[k] for k in loss_dict.keys())

        train_loss.backward()
        # find unused parameters
        # for name, param in (self._model.named_parameters()):
        #     if param.grad is None:
        #         print(name)
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._max_grad_norm)
        self._optimizer.step()
        self._scheduler.step()
        self._model.zero_grad()
        return train_loss.item()


class Criterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, entity_type_count, weight_dict, nil_weight, losses, type_loss, match_class_weight, match_boundary_weight, match_boundary_type, solver):
        """ Create the criterion.
        Parameters:
            entity_type_count: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            nil_weight: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.entity_type_count = entity_type_count
        self.matcher = HungarianMatcher(cost_class = match_class_weight, cost_span = match_boundary_weight, match_boundary_type = match_boundary_type, solver = solver)
        self.weight_dict = weight_dict
        self.nil_weight = nil_weight
        self.losses = losses
        empty_weight = torch.ones(self.entity_type_count)
        empty_weight[0] = self.nil_weight
        self.register_buffer('empty_weight', empty_weight)
        self.type_loss = type_loss

    def loss_labels(self, outputs, targets, indices, num_spans):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs

        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)

        labels = targets["labels"].split(targets["sizes"], dim=-1)

        target_classes_o = torch.cat([t[J] for t, (_, J) in zip(labels, indices)])
        target_classes = torch.full(src_logits.shape[:2], 0,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        empty_weight = self.empty_weight.clone()

        if self.nil_weight == -1:
            empty_weight[0] = num_spans / (src_logits.size(0) * src_logits.size(1) - num_spans)
        if self.type_loss == "celoss":
            src_logits = src_logits.view(-1, src_logits.size(2))
            target_classes = target_classes.view(-1)
            loss_ce = F.cross_entropy(src_logits, target_classes, empty_weight, reduction='none')
        if self.type_loss == "bceloss":
            src_logits = src_logits.view(-1, src_logits.size(2))
            target_classes = target_classes.view(-1)
            target_classes_onehot = torch.zeros([target_classes.size(0), src_logits.size(1)], dtype=torch.float32).to(device=target_classes.device)
            target_classes_onehot.scatter_(1, target_classes.unsqueeze(1), 1)
            src_logits_p = F.sigmoid(src_logits)
            loss_ce = F.binary_cross_entropy(src_logits_p, target_classes_onehot, reduction='none')
        losses = {'loss_ce': loss_ce.mean()}

        return losses

    def loss_boundary(self, outputs, targets, indices, num_spans):
        idx = self._get_src_permutation_idx(indices)
        src_spans_left = outputs['pred_left'][idx]
        src_spans_right = outputs['pred_right'][idx]
        token_masks = outputs['token_mask'].unsqueeze(1).expand(-1, outputs['pred_right'].size(1), -1)
        token_masks = token_masks[idx]

        gt_left = targets["gt_left"].split(targets["sizes"], dim=0)
        target_spans_left = torch.cat([t[i] for t, (_, i) in zip(gt_left , indices)], dim=0)
        gt_right = targets["gt_right"].split(targets["sizes"], dim=0)
        target_spans_right = torch.cat([t[i] for t, (_, i) in zip(gt_right , indices)], dim=0)

        left_onehot = torch.zeros([target_spans_left.size(0), src_spans_left.size(1)], dtype=torch.float32).to(device=target_spans_left.device)
        left_onehot.scatter_(1, target_spans_left.unsqueeze(1), 1)
    
        right_onehot = torch.zeros([target_spans_right.size(0), src_spans_right.size(1)], dtype=torch.float32).to(device=target_spans_right.device)
        right_onehot.scatter_(1, target_spans_right.unsqueeze(1), 1)

        left_nll_loss = F.binary_cross_entropy(src_spans_left, left_onehot, reduction='none')
        right_nll_loss = F.binary_cross_entropy(src_spans_right, right_onehot, reduction='none')

        # NIL object boundary
        loss_boundary = (left_nll_loss + right_nll_loss) * token_masks

        losses = {}
        losses['loss_boundary'] = loss_boundary.sum() / num_spans
        # losses['loss_boundary'] = loss_boundary.mean(1).sum() / num_spans
        

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_spans, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boundary': self.loss_boundary,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_spans, **kwargs)

    # @torchsnooper.snoop()
    def forward(self, outputs, targets, epoch, indices = None):
        # Retrieve the matching between the outputs of the last layer and the targets

        if indices is None:
            indices = self.matcher(outputs, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_spans = sum(targets["sizes"])
        num_spans = torch.as_tensor([num_spans], dtype=torch.float, device=next(iter(outputs.values())).device)

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_spans))
        return losses, indices
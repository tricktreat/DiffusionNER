import argparse
import json
import math
import os

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from transformers import AdamW, AutoTokenizer, AutoConfig

from diffusionner import models
from diffusionner import sampling
from diffusionner import util
from diffusionner.entities import Dataset
from diffusionner.evaluator import Evaluator
from diffusionner.input_reader import JsonInputReader, BaseInputReader
from diffusionner.loss import DiffusionNERLoss, Loss
from tqdm import tqdm
from diffusionner.trainer import BaseTrainer
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))


def get_linear_schedule_with_warmup_two_stage(optimizer, num_warmup_steps_stage_one, num_training_steps_stage_one, num_warmup_steps_stage_two, num_training_steps_stage_two, stage_one_lr_scale, last_epoch=-1):
    def lr_lambda(current_step: int):
        if current_step < num_training_steps_stage_one:
            if current_step < num_warmup_steps_stage_one:
                return float(current_step) / float(max(1, num_warmup_steps_stage_one))
            progress = float(current_step - num_warmup_steps_stage_one) / float(max(1, num_training_steps_stage_one - num_warmup_steps_stage_one))
            return max(
                0.0, float(num_training_steps_stage_one - current_step) / float(max(1, num_training_steps_stage_one - num_warmup_steps_stage_one)) * stage_one_lr_scale
            )
        else:
            current_step = current_step - num_training_steps_stage_one
            if current_step < num_warmup_steps_stage_two:
                return float(current_step) / float(max(1, num_warmup_steps_stage_two))
            progress = float(current_step - num_warmup_steps_stage_two) / float(max(1, num_training_steps_stage_two - num_warmup_steps_stage_two))
            # return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(0.5) * 2.0 * progress)))
            return max(
                0.0, float(num_training_steps_stage_two - current_step) / float(max(1, num_training_steps_stage_two - num_warmup_steps_stage_two))
            )
    return LambdaLR(optimizer, lr_lambda, last_epoch)

class DiffusionNERTrainer(BaseTrainer):
    """ Joint entity and relation extraction training and evaluation """

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)

        # byte-pair encoding
        self._tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path,
                                                        # local_files_only = True,
                                                        do_lower_case=args.lowercase,
                                                        cache_dir=args.cache_path,
                                                        use_fast = False)
        # path to export predictions to
        self._predictions_path = os.path.join(self._log_path, 'predictions_%s_epoch_%s.json')

        # path to export relation extraction examples to
        self._examples_path = os.path.join(self._log_path, 'examples_%s_%s_epoch_%s.html')

        self._logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

    def load_model(self, input_reader, is_eval = False):
        args = self.args
        # create model
        model_class = models.get_model(args.model_type)
            
        config = AutoConfig.from_pretrained(args.model_path, cache_dir=args.cache_path)
        model = model_class.from_pretrained(args.model_path,
                                            ignore_mismatched_sizes=True,
                                            # local_files_only = True,
                                            config = config,
                                            # Prompt4NER model parameters
                                            entity_type_count=input_reader.entity_type_count,
                                            lstm_layers = args.lstm_layers,
                                            span_attn_layers = args.span_attn_layers,
                                            timesteps = args.timesteps,
                                            beta_schedule = args.beta_schedule,
                                            sampling_timesteps = args.sampling_timesteps,
                                            num_proposals = args.num_proposals,
                                            scale = args.scale,
                                            extand_noise_spans = args.extand_noise_spans,
                                            span_renewal = args.span_renewal,
                                            step_ensemble = args.step_ensemble,
                                            prop_drop = args.prop_drop,
                                            soi_pooling = args.soi_pooling,
                                            pos_type =  args.pos_type,
                                            step_embed_type = args.step_embed_type,
                                            sample_dist_type = args.sample_dist_type,
                                            split_epoch = args.split_epoch,
                                            pool_type = args.pool_type,
                                            wo_self_attn = args.wo_self_attn,
                                            wo_cross_attn = args.wo_cross_attn)
        return model

    def train(self, train_path: str, valid_path: str, types_path: str, input_reader_cls: BaseInputReader):
        args = self.args
        train_label, valid_label, test_label = 'train', 'valid', "test"

        if self.record:
            self._logger.info("Datasets: %s, %s" % (train_path, valid_path))
            self._logger.info("Model type: %s" % args.model_type)

            # create log csv files
            self._init_train_logging(train_label)
            self._init_eval_logging(valid_label)
            
            if args.eval_test:
                self._init_eval_logging(test_label)

        # read datasets
        input_reader = input_reader_cls(
            types_path, 
            self._tokenizer, 
            self._logger,
            repeat_gt_entities = args.repeat_gt_entities)
        
        dataset_map = {train_label: train_path, valid_label: valid_path}
        if args.eval_test:
            dataset_map[test_label] = valid_path.replace("dev", "test")
        input_reader.read(dataset_map)

        if self.local_rank < 1:
            self._log_datasets(input_reader)

        world_size = 1
        if args.local_rank != -1:
            world_size = dist.get_world_size()

        train_dataset = input_reader.get_dataset(train_label)
        train_sample_count = train_dataset.document_count
        updates_epoch = math.ceil(train_sample_count / (args.train_batch_size * world_size))
        updates_total = updates_epoch * args.epochs
        updates_total_stage_one = updates_epoch * args.split_epoch
        updates_total_stage_two = updates_epoch * (args.epochs - args.split_epoch)

        validation_dataset = input_reader.get_dataset(valid_label)
        if args.eval_test:
            test_dataset = input_reader.get_dataset(test_label)

        if self.record:
            self._logger.info("Updates per epoch: %s" % updates_epoch)
            self._logger.info("Updates total: %s" % updates_total)

        model = self.load_model(input_reader, is_eval = False)
        self._logger.info(model)

        model.to(self._device)
        if args.local_rank != -1:
            model = DDP(model, device_ids=[args.local_rank], find_unused_parameters=False)

        # create optimizer
        optimizer_params = self._get_optimizer_params(model)
        optimizer = AdamW(optimizer_params, lr=args.lr, weight_decay=args.weight_decay, correct_bias=False)
        # create scheduler
        scheduler = get_linear_schedule_with_warmup_two_stage(optimizer,
                                                    num_warmup_steps_stage_one = args.lr_warmup * updates_total_stage_one,
                                                    num_training_steps_stage_one = updates_total_stage_one,
                                                    num_warmup_steps_stage_two = args.lr_warmup * updates_total_stage_two,
                                                    num_training_steps_stage_two = updates_total_stage_two,
                                                    stage_one_lr_scale = args.stage_one_lr_scale)
        compute_loss = DiffusionNERLoss(input_reader.entity_type_count, self._device, model, optimizer, scheduler, args.max_grad_norm, args.nil_weight, args.match_class_weight, args.match_boundary_weight, args.loss_class_weight, args.loss_boundary_weight, args.match_boundary_type, args.type_loss, solver = args.match_solver)

        # eval validation set
        if args.init_eval and self.record:
            self._eval(model, validation_dataset, input_reader, 0, updates_epoch)
            if args.eval_test:
                self._eval(model, test_dataset, input_reader, 0, updates_epoch)


        # train
        best_f1 = 0
        test_f1 = None
        best_epoch = 0
        for epoch in range(args.epochs):
            # train epoch
            self._train_epoch(model, compute_loss, optimizer, train_dataset, updates_epoch, epoch)

            # eval validation sets
            if (not args.final_eval or (epoch == args.epochs - 1)) and self.record and ((epoch%args.eval_every_epochs)==0 or (epoch == args.epochs - 1)):
                f1 = self._eval(model, validation_dataset, input_reader, epoch + 1, updates_epoch)
                if args.eval_test:
                    test = self._eval(model, test_dataset, input_reader, epoch + 1, updates_epoch)
                # self._save_best(model, self._tokenizer,optimizer if args.save_optimizer else None,f1[2],epoch * updates_epoch, "best")
                if best_f1 < f1[2]:
                    self._logger.info(f"Best F1 score update, from {best_f1} to {f1[2]}")
                    best_f1 = f1[2]
                    if args.eval_test:
                        test_f1 = test[2]
                    best_epoch = epoch + 1
                    extra = dict(epoch=epoch, updates_epoch=updates_epoch, epoch_iteration=0)
            if self.record and ((epoch%args.eval_every_epochs)==0 or (epoch == args.epochs - 1)):
                if args.save_path_include_iteration:
                    self._save_model(self._save_path, model, self._tokenizer, epoch,
                            optimizer=optimizer if args.save_optimizer else None, extra=extra,
                            include_iteration=args.save_path_include_iteration, name='model')
                self._logger.info(f"Best Dev-F1 score: {best_f1}, achieved at Epoch: {best_epoch}, Test-F1: {test_f1}")

        # save final model
        extra = dict(epoch=args.epochs, updates_epoch=updates_epoch, epoch_iteration=0)
        global_iteration = args.epochs * updates_epoch
        if self.record:
            self._save_model(self._save_path, model, self._tokenizer, global_iteration,
                            optimizer=optimizer if args.save_optimizer else None, extra=extra,
                            include_iteration=False, name='final_model')
            self._logger.info("Logged in: %s" % self._log_path)
            self._logger.info("Saved in: %s" % self._save_path)
            self._close_summary_writer()

    def eval(self, dataset_path: str, types_path: str, input_reader_cls: BaseInputReader):
        args = self.args
        dataset_label = 'test'

        self._logger.info("Dataset: %s" % dataset_path)
        self._logger.info("Model: %s" % args.model_type)

        # create log csv files
        self._init_eval_logging(dataset_label)
        
        # read datasets
        input_reader = input_reader_cls(
            types_path, 
            self._tokenizer, 
            self._logger,
            repeat_gt_entities = args.repeat_gt_entities)
            
        input_reader.read({dataset_label: dataset_path})
        self._log_datasets(input_reader)

        model = self.load_model(input_reader, is_eval = True)

        model.to(self._device)

        # evaluate
        self._eval(model, input_reader.get_dataset(dataset_label), input_reader)

        self._logger.info("Logged in: %s" % self._log_path)
        self._close_summary_writer()

    def _train_epoch(self, model: torch.nn.Module, compute_loss: Loss, optimizer: Optimizer, dataset,
                     updates_epoch: int, epoch: int):
        args = self.args
        self._logger.info("Train epoch: %s" % epoch)

        # create data loader
        dataset.switch_mode(Dataset.TRAIN_MODE)

        world_size = 1
        if args.local_rank != -1:
            world_size = dist.get_world_size()

        train_sampler = None
        shuffle = False
        if isinstance(dataset, Dataset):
            if len(dataset) < 100000:
                shuffle = True
            if args.local_rank != -1:
                train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas = world_size,rank = args.local_rank, shuffle = shuffle)
                shuffle = False

        data_loader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=shuffle, drop_last=False,
                                    num_workers=args.sampling_processes, collate_fn=sampling.collate_fn_padding,  sampler=train_sampler)
                                    

        model.zero_grad()

        iteration = 0
        total = math.ceil(dataset.document_count / (args.train_batch_size * world_size))
        for batch in tqdm(data_loader, total=total, desc='Train epoch %s' % epoch):
            if epoch == 0 and iteration == 0:
                for k, v in batch.items():
                    torch.set_printoptions(profile='full')
                    if v is None:
                        continue
                    if isinstance(v, dict):
                        for sub_k, sub_v in v.items():
                            extended_k = k + ' -> ' + sub_k
                            self._logger.info(extended_k)
                            self._logger.info(sub_v[:2].size())
                    else:
                        if isinstance(v, torch.Tensor) and v[:2].numel()> 5120:
                            torch.set_printoptions(profile='default')
                        self._logger.info(k)
                        # if sum(v.size()[1:]) > 
                        self._logger.info(v[:2])
                torch.set_printoptions(profile='default')
            model.train()
            batch = util.to_device(batch, self._device)

            # forward step
            outputs = model(
                encodings=batch['encodings'], 
                context_masks=batch['context_masks'], 
                seg_encoding = batch['seg_encoding'], 
                context2token_masks = batch['context2token_masks'], 
                token_masks = batch['token_masks'],
                entity_spans = batch['gt_spans'],
                entity_types = batch['gt_types'],
                entity_masks = batch['entity_masks'],
                meta_doc = batch['meta_doc'], 
                epoch = epoch)

            # compute loss and optimize parameters
            batch_loss = compute_loss.compute(outputs, gt_types=batch['gt_types'], gt_spans = batch['gt_spans'], entity_masks=batch['entity_masks'], epoch = epoch, batch = batch)

            # logging
            iteration += 1
            global_iteration = epoch * updates_epoch + iteration

            if global_iteration % args.train_log_iter == 0 and self.local_rank < 1:
                self._log_train(optimizer, batch_loss, epoch, iteration, global_iteration, dataset.label)

        return iteration

    def _eval(self, model: torch.nn.Module, dataset, input_reader: JsonInputReader,
              epoch: int = 0, updates_epoch: int = 0, iteration: int = 0):
        args = self.args
        self._logger.info("Evaluate: %s" % dataset.label)

        # if isinstance(model, DataParallel):
        #     # currently no multi GPU support during evaluation
        #     model = model.module

        # create evaluator
        evaluator = Evaluator(dataset, input_reader, self._tokenizer, self._logger, args.no_overlapping, args.no_partial_overlapping, args.no_duplicate, self._predictions_path,
                              self._examples_path, args.example_count, epoch, dataset.label,  cls_threshold = args.cls_threshold, boundary_threshold = args.boundary_threshold, entity_threshold = args.entity_threshold, save_prediction = args.store_predictions)

        # create data loader
        dataset.switch_mode(Dataset.EVAL_MODE)

        world_size = 1
        eval_sampler = None
        # assert len(gt) == len(pred)
        # if args.local_rank != -1:
        #     world_size = dist.get_world_size()
        #     eval_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas = world_size,rank = args.local_rank)



        if isinstance(dataset, Dataset):
            data_loader = DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False, drop_last=False,
                                 num_workers=args.sampling_processes, collate_fn=sampling.collate_fn_padding, sampler=eval_sampler)
        else:
            data_loader = DataLoader(dataset, batch_size=args.eval_batch_size, drop_last=False, collate_fn=sampling.collate_fn_padding, sampler=eval_sampler)

        with torch.no_grad():
            model.eval()

            # iterate batches
            total = math.ceil(dataset.document_count / (args.eval_batch_size * world_size))
            for batch in tqdm(data_loader, total=total, desc='Evaluate epoch %s' % epoch):
                # move batch to selected device
                batch = util.to_device(batch, self._device)

                # run model (forward pass)
                outputs = model(
                    encodings=batch['encodings'], 
                    context_masks=batch['context_masks'], 
                    seg_encoding = batch['seg_encoding'], 
                    context2token_masks=batch['context2token_masks'], 
                    token_masks=batch['token_masks'],
                    meta_doc = batch['meta_doc'])

                # evaluate batch
                evaluator.eval_batch(outputs, batch)
        global_iteration = epoch * updates_epoch + iteration
        ner_eval, ner_loc_eval, ner_cls_eval = evaluator.compute_scores()
        self._log_eval(*ner_eval, *ner_loc_eval, *ner_cls_eval, epoch, iteration, global_iteration, dataset.label)

        # self.scheduler.step(ner_eval[2])

        if args.store_predictions:
            evaluator.store_predictions()

        if args.store_examples:
            evaluator.store_examples()
        
        return ner_eval

    def _get_optimizer_params(self, model):
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        # regressier
        optimizer_params = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

        return optimizer_params

    def _log_train(self, optimizer: Optimizer, loss: float, epoch: int,
                   iteration: int, global_iteration: int, label: str):
        # average loss
        avg_loss = loss / self.args.train_batch_size
        # get current learning rate
        lr = self._get_lr(optimizer)[0]

        # log to tensorboard
        self._log_tensorboard(label, 'loss', loss, global_iteration)
        self._log_tensorboard(label, 'loss_avg', avg_loss, global_iteration)
        self._log_tensorboard(label, 'lr', lr, global_iteration)

        # log to csv
        self._log_csv(label, 'loss', loss, epoch, iteration, global_iteration)
        self._log_csv(label, 'loss_avg', avg_loss, epoch, iteration, global_iteration)
        self._log_csv(label, 'lr', lr, epoch, iteration, global_iteration)

    def _log_eval(self, ner_prec_micro: float, ner_rec_micro: float, ner_f1_micro: float,
                  ner_prec_macro: float, ner_rec_macro: float, ner_f1_macro: float,
                  loc_prec_micro: float, loc_rec_micro: float, loc_f1_micro: float,
                  loc_prec_macro: float, loc_rec_macro: float, loc_f1_macro: float,
                  cls_prec_micro: float, cls_rec_micro: float, cls_f1_micro: float,
                  cls_prec_macro: float, cls_rec_macro: float, cls_f1_macro: float,
                  epoch: int, iteration: int, global_iteration: int, label: str):

        # log to tensorboard
        self._log_tensorboard(label, 'eval/ner_prec_micro', ner_prec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_recall_micro', ner_rec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_f1_micro', ner_f1_micro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_prec_macro', ner_prec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_recall_macro', ner_rec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_f1_macro', ner_f1_macro, global_iteration)


        self._log_tensorboard(label, 'eval/loc_prec_micro', loc_prec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/loc_recall_micro', loc_rec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/loc_f1_micro', loc_f1_micro, global_iteration)
        self._log_tensorboard(label, 'eval/loc_prec_macro', loc_prec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/loc_recall_macro', loc_rec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/loc_f1_macro', loc_f1_macro, global_iteration)

        self._log_tensorboard(label, 'eval/cls_prec_micro', cls_prec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/cls_recall_micro', cls_rec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/cls_f1_micro', cls_f1_micro, global_iteration)
        self._log_tensorboard(label, 'eval/cls_prec_macro', cls_prec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/cls_recall_macro', cls_rec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/cls_f1_macro', cls_f1_macro, global_iteration)


        # log to csv
        self._log_csv(label, 'eval', ner_prec_micro, ner_rec_micro, ner_f1_micro,
                      ner_prec_macro, ner_rec_macro, ner_f1_macro,
                      loc_prec_micro, loc_rec_micro, loc_f1_micro,
                      loc_prec_macro, loc_rec_macro, loc_f1_macro,
                      cls_prec_micro, cls_rec_micro, cls_f1_micro,
                      cls_prec_macro, cls_rec_macro, cls_f1_macro,
                      epoch, iteration, global_iteration)

    def _log_datasets(self, input_reader):
        # self._logger.info("Relation type count: %s" % input_reader.relation_type_count)
        self._logger.info("Entity type count: %s" % input_reader.entity_type_count)

        self._logger.info("Entities:")
        for e in input_reader.entity_types.values():
            self._logger.info(e.verbose_name + '=' + str(e.index))

        # self._logger.info("Relations:")
        # for r in input_reader.relation_types.values():
        #     self._logger.info(r.verbose_name + '=' + str(r.index))

        for k, d in input_reader.datasets.items():
            self._logger.info('Dataset: %s' % k)
            self._logger.info("Document count: %s" % d.document_count)
            # self._logger.info("Relation count: %s" % d.relation_count)
            self._logger.info("Entity count: %s" % d.entity_count)

        self._logger.info("Context size: %s" % input_reader.context_size)

    def _init_train_logging(self, label):
        self._add_dataset_logging(label,
                                  data={'lr': ['lr', 'epoch', 'iteration', 'global_iteration'],
                                        'loss': ['loss', 'epoch', 'iteration', 'global_iteration'],
                                        'loss_avg': ['loss_avg', 'epoch', 'iteration', 'global_iteration']})

    def _init_eval_logging(self, label):
        self._add_dataset_logging(label,
                                  data={'eval': ['ner_prec_micro', 'ner_rec_micro', 'ner_f1_micro',
                                                 'ner_prec_macro', 'ner_rec_macro', 'ner_f1_macro',
                                                 'loc_prec_micro', 'loc_rec_micro', 'loc_f1_micro',
                                                 'loc_prec_macro', 'loc_rec_macro', 'loc_f1_macro',
                                                 'cls_prec_micro', 'cls_rec_micro', 'cls_f1_micro',
                                                 'cls_prec_macro', 'cls_rec_macro', 'cls_f1_macro',
                                                 'epoch', 'iteration', 'global_iteration']})

 
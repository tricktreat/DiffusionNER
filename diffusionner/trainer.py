import argparse
import datetime
import logging
import os
import shutil
import sys
from typing import List, Dict, Tuple

import torch
from torch.nn import DataParallel
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.optim import Optimizer
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizer

from diffusionner import util
import torch.utils.tensorboard as tensorboard

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))


class BaseTrainer:
    """ Trainer base class with common methods """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self._debug = self.args.debug

        self.local_rank = args.local_rank
        self.record = False
        self.save_code = args.save_code
        if self.local_rank < 1:
            self.record = True

        # logging
        # name = str(datetime.datetime.now()).replace(' ', '_') + f'_B{args.train_batch_size}_E{args.epochs}_LR{args.lr}' 
        name = str(datetime.datetime.now()).replace(' ', '_')
        self._log_path = os.path.join(self.args.log_path, self.args.label, name)
        if self.record:
            util.create_directories_dir(self._log_path)

        if self.save_code:
            code_dir = os.path.join(self._log_path, "code")
            util.create_directories_dir(code_dir)
            for filename in ["args.py", "config_reader.py", "diffusionner.py"]:
                shutil.copyfile(os.path.join(os.path.dirname(SCRIPT_PATH), filename), os.path.join(code_dir, filename))
            shutil.copytree(SCRIPT_PATH,  os.path.join(code_dir, "diffusionner"))

        if hasattr(args, 'save_path') and self.record:
            self._save_path = os.path.join(self.args.save_path, self.args.label, name)
            util.create_directories_dir(self._save_path)

        self._log_paths = dict()

        # file + console logging
        log_formatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
        self._logger = logging.getLogger()
        util.reset_logger(self._logger)

        if self.record:
            file_handler = logging.FileHandler(os.path.join(self._log_path, 'all.log'))
            file_handler.setFormatter(log_formatter)
            self._logger.addHandler(file_handler)


        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_formatter)
        if "pretrain" not in self.args.label:
            self._logger.addHandler(console_handler)

        if self._debug:
            self._logger.setLevel(logging.DEBUG)
        else:
            self._logger.setLevel(logging.INFO)

        # tensorboard summary
        if self.record:
            self._summary_writer = tensorboard.SummaryWriter(self._log_path)

        self._best_results = dict()
        if self.record:
            self._log_arguments()

        # CUDA devices
        # self._device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
        if args.cpu:
            device="cpu"
        else:
            device="cuda:" + str(args.device_id)


        # set seed
        if args.seed is not None:
            util.set_seed(args.seed)

        if args.local_rank != -1 and "eval"  not in args.label:
            torch.cuda.set_device(args.local_rank)
            torch.distributed.init_process_group(backend='nccl', init_method='env://', rank = args.local_rank, world_size = args.world_size)
            self._device = torch.device('cuda', args.local_rank)
        else:
            self._device = torch.device(device)
            self._gpu_count = torch.cuda.device_count()

    def _add_dataset_logging(self, *labels, data: Dict[str, List[str]]):
        for label in labels:
            dic = dict()

            for key, columns in data.items():
                path = os.path.join(self._log_path, '%s_%s.csv' % (key, label))
                util.create_csv(path, *columns)
                dic[key] = path

            self._log_paths[label] = dic
            self._best_results[label] = 0

    def _log_arguments(self):
        util.save_dict(self._log_path, self.args, 'args')
        if self._summary_writer is not None:
            util.summarize_dict(self._summary_writer, self.args, 'args')

    def _log_tensorboard(self, dataset_label: str, data_label: str, data: object, iteration: int):
        if self._summary_writer is not None:
            self._summary_writer.add_scalar('data/%s/%s' % (dataset_label, data_label), data, iteration)

    def _log_csv(self, dataset_label: str, data_label: str, *data: Tuple[object]):
        logs = self._log_paths[dataset_label]
        util.append_csv(logs[data_label], *data)
    
    def _save_model(self, save_path: str, model: PreTrainedModel, tokenizer: PreTrainedTokenizer,
                    iteration: int, optimizer: Optimizer = None, save_as_best: bool = False,
                    extra: dict = None, include_iteration: int = True, name: str = 'model'):
        extra_state = dict(iteration=iteration)

        if optimizer:
            extra_state['optimizer'] = optimizer.state_dict()

        if extra:
            extra_state.update(extra)

        if save_as_best:
            dir_path = os.path.join(save_path, 'best_%s' % name)
        else:
            dir_name = '%s_%s' % (name, iteration) if include_iteration else name
            dir_path = os.path.join(save_path, dir_name)

        util.create_directories_dir(dir_path)

        # save model
        if isinstance(model, (DataParallel, DistributedDataParallel)):
            model.module.save_pretrained(dir_path)
        else:
            model.save_pretrained(dir_path)

        # save vocabulary
        tokenizer.save_pretrained(dir_path)

        # save extra
        state_path = os.path.join(dir_path, 'extra.state')
        torch.save(extra_state, state_path)

    def _get_lr(self, optimizer):
        lrs = []
        for group in optimizer.param_groups:
            lr_scheduled = group['lr']
            lrs.append(lr_scheduled)
        return lrs

    def _close_summary_writer(self):
        if self._summary_writer is not None:
            self._summary_writer.close()

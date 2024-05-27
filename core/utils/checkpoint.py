import logging
import os

import torch
from torch.nn.parallel import DistributedDataParallel


class CheckPointer:
    _last_checkpoint_name = 'last_checkpoint.txt'

    def __init__(self,
                 model,
                 optimizer=None,
                 scheduler=None,
                 save_dir="",
                 save_to_disk=None,
                 logger=None,
                 ckpt_path=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.save_to_disk = save_to_disk
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger
        self.ckpt_path = ckpt_path

    def save(self, name, **kwargs):
        if not self.save_dir:
            return

        if not self.save_to_disk:
            return

        data = {}
        if isinstance(self.model, DistributedDataParallel):
            self.logger.info("Saving model state dict")
            data['model'] = self.model.module.state_dict()
        else:
            self.logger.info("Saving model state dict")
            data['model'] = self.model.state_dict()
        if self.optimizer is not None:
            self.logger.info("Saving optimizer state dict")
            data["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            self.logger.info("Saving scheduler state dict")
            data["scheduler"] = self.scheduler.state_dict()
        data.update(kwargs)

        save_file = os.path.join(self.save_dir, "{}.pth".format(name))
        self.logger.info("Saving checkpoint to {}".format(save_file))
        torch.save(data, save_file)

        self.tag_last_checkpoint(save_file)

    def load(self, f=None, use_latest=True):
        if self.has_checkpoint() and use_latest:
            # override argument with existing checkpoint
            f = self.get_checkpoint_file()
        if not f:
            # no checkpoint could be found
            self.logger.info("No checkpoint found.")
            return {}

        self.logger.info("Loading checkpoint from {}".format(f))
        checkpoint = self._load_file(f)

        model = self.model
        if isinstance(model, DistributedDataParallel):
            model = self.model.module

        self.logger.info("Loading model state dict from {}".format(f))
        if "model" in checkpoint and self.model:
            try:
                model.load_state_dict(checkpoint.pop("model"), strict=False)
            except ValueError:
                self.logger.info("Model state dict load failed")
        elif "state_dict" in checkpoint and self.model:
            try:
                model.load_state_dict(checkpoint.pop("state_dict"), strict=False)
            except ValueError:
                self.logger.info("Model state dict load failed")
        else:
            try:
                model.dmc.load_state_dict(checkpoint, strict=False)
            except ValueError:
                self.logger.info("Model state dict load failed")

        if "optimizer" in checkpoint and self.optimizer:
            self.logger.info("Loading optimizer state dict from {}".format(f))
            try:
                opt_params = checkpoint.pop("optimizer")

                # NOTE: This section is needed to load optimizer state dict
                # from old torch vesrion format < 1.12 to higher versions
                occurrences = 0
                for val in opt_params['state'].values():
                    if 'step' in val:
                        if not isinstance(val['step'], torch.Tensor):
                            val['step'] = torch.tensor(val['step'], dtype=torch.float32)
                            occurrences += 1
                if occurrences > 0:
                    self.logger.warning(
                        "Optimizer state steps were converted (int to tensor, {} occurrences)".format(occurrences))

                self.optimizer.load_state_dict(opt_params)
            except ValueError:
                self.logger.info("Optimizer state dict load failed")

        if "scheduler" in checkpoint and self.scheduler:
            self.logger.info("Loading scheduler state dict from {}".format(f))
            try:
                self.scheduler.load_state_dict(checkpoint.pop("scheduler"))
            except ValueError:
                self.logger.info("Scheduler state dict load failed")

        return checkpoint

    def get_checkpoint_file(self):
        if self.ckpt_path is None:
            save_file = os.path.join(self.save_dir, self._last_checkpoint_name)
            try:
                with open(save_file, "r") as f:
                    last_saved = f.read()
                    last_saved = last_saved.strip()
            except IOError:
                # if file doesn't exist, maybe because it has just been
                # deleted by a separate process
                last_saved = ""
        else:
            last_saved = self.ckpt_path
        return last_saved

    def has_checkpoint(self):
        save_file = os.path.join(self.save_dir, self._last_checkpoint_name)
        return os.path.exists(save_file)

    def tag_last_checkpoint(self, last_filename):
        save_file = os.path.join(self.save_dir, self._last_checkpoint_name)
        with open(save_file, "w") as f:
            f.write(last_filename)

    def _load_file(self, f):
        return torch.load(f, map_location=torch.device("cpu"))

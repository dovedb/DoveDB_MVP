import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import dataloader, distributed, Sampler, DataLoader
from mmengine.registry import RUNNERS
from mmengine.config import Config

from .mmutils import RANK, PIN_MEMORY, set_data_info
from .modules.fasterrcnn import CFG_DICT

class _RepeatSampler(Sampler):
    """
    Sampler that repeats forever.

    Args:
        sampler (Dataset.sampler): The sampler to repeat.
    """

    def __init__(self, sampler, *args, **kwargs):
        """Initializes an object that repeats a given sampler indefinitely."""
        self.sampler = sampler

    @property
    def batch_size(self):
        """Returns the batch size of the sampler."""
        return self.sampler.batch_size

    @property
    def drop_last(self):
        """Returns whether or not the sampler drops the last batch."""
        return self.sampler.drop_last
    
    def __len__(self):
        """Returns the length of the sampler."""
        return len(self.sampler)
    
    def __iter__(self):
        """Iterates over the 'sampler' and yields its contents."""
        while True:
            yield from iter(self.sampler)


class InfiniteDataLoader(dataloader.DataLoader):
    """Dataloader that reuses workers. Uses same syntax as vanilla DataLoader."""

    def __init__(self, *args, **kwargs):
        """Dataloader that infinitely recycles workers, inherits from DataLoader."""
        super().__init__(*args, **kwargs)
        self.iterator = super().__iter__()

    def __len__(self):
        """Returns the length of the batch sampler's sampler."""
        return len(self.batch_sampler)

    # def __iter__(self):
    #     """Creates a sampler that repeats indefinitely."""
    #     for _ in range(len(self)):
    #         yield next(self.iterator)

    def reset(self):
        """Reset iterator.
        This is useful when we want to modify settings of dataset while training.
        """
        self.iterator = self._get_iterator()
        
        
def seed_worker(worker_id):  # noqa
    # Set dataloader worker seed https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataloader(dataset, batch, workers, shuffle=True, rank=-1):
    """Return an InfiniteDataLoader or DataLoader for training or validation set."""
    batch = min(batch, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min(min([os.cpu_count() // max(nd, 1), batch if batch > 1 else 0, workers]), 4)  # number of workers
    sampler = None # if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + RANK)
    return InfiniteDataLoader(dataset=dataset,
                              batch_size=batch,
                              shuffle=shuffle and sampler is None,
                              num_workers=nw,
                              sampler=sampler,
                              pin_memory=PIN_MEMORY,
                              collate_fn=getattr(dataset, 'collate_fn', None),
                              worker_init_fn=seed_worker,
                              generator=generator)


def build_mm_dataloader(model_size):
    current_file = os.path.realpath(__file__)
    current_directory = os.path.dirname(current_file)
    cfg_path = os.path.join(current_directory, "configs", "faster_rcnn", CFG_DICT[model_size])
    
    cfg = Config.fromfile(cfg_path)
    cfg.work_dir = os.path.join('./work_dirs', os.path.splitext(os.path.basename(cfg_path))[0])
    annotation_path = os.path.join(cfg.train_dataloader.dataset.data_root, "annotations/train.json")
    set_data_info(cfg, annotation_path)
        
    runner = RUNNERS.build(cfg)
    dataset = runner.train_dataloader.dataset
    
    def collate_fn(batch):
        # batch_inputs, batch_samples = [], []
        # mean = torch.tensor(cfg.model.data_preprocessor.mean, dtype=torch.float32).view(-1, 1, 1)
        # std = torch.tensor(cfg.model.data_preprocessor.std, dtype=torch.float32).view(-1, 1, 1)
        # for sample in batch:
        #     batch_input = (sample['inputs'] - mean) / std
        #     batch_inputs.append(batch_input)
        #     batch_samples.append(sample['data_samples'])
        # batch_inputs = torch.stack(batch_inputs, dim=0)
        batch_inputs, batch_samples = [], []
        for sample in batch:
            batch_inputs.append(sample['inputs'])
            batch_samples.append(sample['data_samples'])
        batch_inputs = torch.stack(batch_inputs, dim=0)
        batchs = {'inputs': batch_inputs, 'data_samples': batch_samples}
        return batchs
    
    dataloader = DataLoader(dataset, batch_size=2,
                            shuffle=True, num_workers=4, pin_memory=True, 
                            collate_fn=collate_fn, worker_init_fn=seed_worker)
    
    return dataloader
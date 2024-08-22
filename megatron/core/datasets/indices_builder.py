import logging
import os
import numpy
import torch
from torch import multiprocessing as mp

from megatron.core.datasets.blended_megatron_dataset_config import BlendedMegatronDatasetConfig
from megatron.core.utils import log_single_rank

logger = logging.getLogger(__name__)


def build_blending_indices(
    weights: list[float] | numpy.ndarray[float],
    num_datasets: int,
    size: int,
) -> tuple[numpy.ndarray[numpy.int16], numpy.ndarray[numpy.int64]]:
    """
    Given multiple datasets and a weighting array, build samples
    such that it follows those wieghts.
    """
    fsamples = numpy.array(weights) * size
    isamples = fsamples.astype(int)
    errors = fsamples - isamples
    error_ranks = numpy.argsort(errors)
    add = size - sum(isamples)
    isamples[error_ranks[-add:]] += 1
    #
    dataset_index = numpy.repeat(numpy.arange(num_datasets, dtype=numpy.int16)[error_ranks], isamples[error_ranks])
    c = 0
    dataset_sample_index = numpy.zeros(size, dtype=numpy.int64)
    maxrange = numpy.arange(max(isamples))
    for i in error_ranks:
        isample = isamples[i]
        dataset_sample_index[c:c+isample] = maxrange[0:isample]
        c += isample
    return dataset_index, dataset_sample_index


class AsyncShuffleBuilder:
    """ Build dataset shuffle indices asynchronously with multiprocessing
    """

    def __init__(
        self,
        num_samples: int,
        numpy_random_state: numpy.random.RandomState,
        config: BlendedMegatronDatasetConfig,
        ):
        self.num_samples = num_samples
        self.numpy_random_state = numpy_random_state
        self.config = config
        #
        self.shuffle_index = None
        self.manager = None
        self.queue = None
        self.process = None
        self.is_running = False
        #
        self.try_load_from_cache()

    def get_cache_path(self):
        path_to_cache = self.config.path_to_cache
        if path_to_cache:
            return os.path.join(path_to_cache, f"dataset_shuffle_index_{self.num_samples:d}.npy")
        return None

    def try_load_from_cache(self):
        path_to_dataset_shuffle_index = self.get_cache_path()
        if path_to_dataset_shuffle_index and os.path.exists(path_to_dataset_shuffle_index):
            log_single_rank(
                logger,
                logging.INFO,
                f"Load shuffle index from file {path_to_dataset_shuffle_index}",
            )
            self.shuffle_index = numpy.load(path_to_dataset_shuffle_index, allow_pickle=True, mmap_mode='r')

    def save(self):
        path_to_dataset_shuffle_index = self.get_cache_path()
        if path_to_dataset_shuffle_index:
            if os.path.exists(path_to_dataset_shuffle_index):
                raise RuntimeError(f"File {path_to_dataset_shuffle_index} already exists.")
            log_single_rank(
                logger,
                logging.INFO,
                f"Save shuffle index to file {path_to_dataset_shuffle_index}",
            )
            numpy.save(path_to_dataset_shuffle_index, self.get_result(), allow_pickle=True)

    @staticmethod
    def build_shuffle_index(
        num_samples: int,
        numpy_random_state: numpy.random.RandomState,
        queue: mp.Queue,
        save_cache: str = None,
    ) -> numpy.ndarray:
        dataset_shuffle_index = numpy.arange(num_samples, dtype=numpy.int64)
        numpy_random_state.shuffle(dataset_shuffle_index)
        queue.put(dataset_shuffle_index)
        if save_cache:
            numpy.save(save_cache, dataset_shuffle_index, allow_pickle=True)

    def start(self):
        if self.shuffle_index is not None:
            return
        ctx = mp.get_context("fork")
        self.manager = mp.Manager()
        self.queue = self.manager.Queue()
        self.process = ctx.Process(
            target=self.build_shuffle_index,
            args=(self.num_samples, self.numpy_random_state, self.queue, self.get_cache_path()),
            )
        self.process.start()
        self.is_running = True

    def join(self):
        if self.is_running:
            self.process.join()
            self.is_running = False

    def get_result(self):
        if self.shuffle_index is None:
            self.join()
            self.shuffle_index = self.queue.get()
        return self.shuffle_index

    def run(self):
        self.start()
        self.join()
        self.save()

    def shutdown(self):
        self.manager.shutdown()

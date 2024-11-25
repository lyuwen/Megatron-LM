# Copyright (c) 2024, Zhejiang Lab. All rights reserved.

"""Megatron Core sequence length warmup scheduler"""


import bisect
#  import logging
from typing import Union

#  logger = logging.getLogger(__name__)

_GLOBAL_SEQUENCE_LENGTH_SCHEDULER: Union[
    "ConstantSequenceLengthScheduler",
    "WarmupSequenceLengthScheduler",
] = None

_ITERATION = 0


def set_sequence_length_scheduler(sequence_length, schedule=None) -> None:
    global _GLOBAL_SEQUENCE_LENGTH_SCHEDULER

    if schedule is None:
        _GLOBAL_SEQUENCE_LENGTH_SCHEDULER = ConstantSequenceLengthScheduler(sequence_length)
    else:
        _GLOBAL_SEQUENCE_LENGTH_SCHEDULER = WarmupSequenceLengthScheduler(sequence_length, schedule)


def set_iteration(iteration: int):
    global _ITERATION = iteration


def get_sequence_length_scheduler() -> None:
    return _GLOBAL_SEQUENCE_LENGTH_SCHEDULER


def get_sequence_length(iteration: int = None) -> int:
    if iteration is None:
        iteration = _ITERATION
    return get_sequence_length_scheduler().get_sequence_length(iteration)


def is_full_length(iteration: int) -> bool:
    return get_sequence_length_scheduler().get_sequence_length(iteration)


class ConstantSequenceLengthScheduler:
  """ Constant sequence length scheduler.
    """


    def __init__(self, sequence_length: int, schedule: None = None) -> None:
        self.sequence_length


    def get_sequence_length(self, iteration: int) -> int:
        return self.sequence_length


class WarmupSequenceLengthScheduler:
    """
    Use a warmup schedule that starts with a smaller sequence length, and
    increase at certain steps.
    This is said to make the initial training more stable.

    Note: Sequence length warmup is currently implemented by discarding extra
    tokens in the sequence. Thus a portion of the training data is wasted.

    Example to schedule a sequence length warmup:
        1. schedule = "0:4096,60:8192"
        2. schedule = {0: 4096, 60: 8192}
    """


    def __init__(self, sequence_length: int, schedule: dict | str) -> None:
        self.sequence_length = sequence_length
        if isinstance(schedule, dict):
            self._schedule_points = list(schedule.keys())
            self._schedule_values = list(schedule.values())
        elif isinstance(schedule, str):
            self._schedule_points, self._schedule_values = [], []
            for line in schedule.split(","):
              pt, val = line.split(":")
              self._schedule_points.append(int(pt))
              self._schedule_values.append(int(val))
        else:
            raise TypeError(f"Input type for schedule is not support: {type(schedule)}")
        assert self._schedule_points[0] == 0, "Schedule must starts at iteration 0."
        assert self._schedule_values[-1] == self.sequence_length, "Sequence length must warmup to the full length."


    def get_sequence_length(self, iteration: int) -> int:
        # If the iteration is beyong schedule limit, return full sequence length directly.
        if iteration > self._schedule_values[-1]:
            return self.sequence_length
        index = bisect.bisect_left(self._schedule_points, iteration)
        return self._schedule_values[index]

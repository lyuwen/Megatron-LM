#!/usr/bin/env python
# Copyright (c) 2025, Author. All rights reserved.

import argparse
import numpy as np
from typing import List, Tuple, Dict, Optional

def generate_s6_mb12_test_schedule_masks(schedule: List[List[Tuple[str, int]]]) -> Dict[str, List[List[int]]]:
    num_stages = len(schedule)
    max_steps = max(len(stage) for stage in schedule)
    
    sendF_mask = [[0 for _ in range(max_steps)] for _ in range(num_stages)]
    recvF_mask = [[0 for _ in range(max_steps)] for _ in range(num_stages)]
    sendB_mask = [[0 for _ in range(max_steps)] for _ in range(num_stages)]
    recvB_mask = [[0 for _ in range(max_steps)] for _ in range(num_stages)]
    
    for stage in range(num_stages):
        for step, (op, _) in enumerate(schedule[stage]):
            if op == 'F':
                sendF_mask[stage][step] = 1
                recvF_mask[stage][step] = 1
                sendB_mask[stage][step] = 0
                recvB_mask[stage][step] = 0
            else:
                sendF_mask[stage][step] = 0
                recvF_mask[stage][step] = 0
                sendB_mask[stage][step] = 1
                recvB_mask[stage][step] = 1
    
    return {
        'sendF': sendF_mask,
        'recvF': recvF_mask,
        'sendB': sendB_mask,
        'recvB': recvB_mask
    }


def generate_s16_mb32_test_schedule() -> List[List[Tuple[str, int]]]:
    if True:
        raw = """
        F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 B0 F16 B1 F17 B2 F18 B3 F19 B4 F20 B5 F21 B6 F22 B7 F23 B8 F24 B9 F25 B10 F26 B11 F27 B12 F28 B13 F29 B14 F30 B15 F31 B16 B17 B18 B19 B20 B21 B22 B23 B24 B25 B26 B27 B28 B29 B30 B31
        F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 B0 F15 B1 F16 B2 F17 B3 F18 B4 F19 B5 F20 B6 F21 B7 F22 B8 F23 B9 F24 B10 F25 B11 F26 B12 F27 B13 F28 B14 F29 B15 F30 B16 F31 B17 B18 B19 B20 B21 B22 B23 B24 B25 B26 B27 B28 B29 B30 B31
        F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 B0 F14 B1 F15 B2 F16 B3 F17 B4 F18 B5 F19 B6 F20 B7 F21 B8 F22 B9 F23 B10 F24 B11 F25 B12 F26 B13 F27 B14 F28 B15 F29 B16 F30 B17 F31 B18 B19 B20 B21 B22 B23 B24 B25 B26 B27 B28 B29 B30 B31
        F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 B0 F13 B1 F14 B2 F15 B3 F16 B4 F17 B5 F18 B6 F19 B7 F20 B8 F21 B9 F22 B10 F23 B11 F24 B12 F25 B13 F26 B14 F27 B15 F28 B16 F29 B17 F30 B18 F31 B19 B20 B21 B22 B23 B24 B25 B26 B27 B28 B29 B30 B31
        F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 B0 F12 B1 F13 B2 F14 B3 F15 B4 F16 B5 F17 B6 F18 B7 F19 B8 F20 B9 F21 B10 F22 B11 F23 B12 F24 B13 F25 B14 F26 B15 F27 B16 F28 B17 F29 B18 F30 B19 F31 B20 B21 B22 B23 B24 B25 B26 B27 B28 B29 B30 B31
        F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 B0 F11 B1 F12 B2 F13 B3 F14 B4 F15 B5 F16 B6 F17 B7 F18 B8 F19 B9 F20 B10 F21 B11 F22 B12 F23 B13 F24 B14 F25 B15 F26 B16 F27 B17 F28 B18 F29 B19 F30 B20 F31 B21 B22 B23 B24 B25 B26 B27 B28 B29 B30 B31
        F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 B0 F10 B1 F11 B2 F12 B3 F13 B4 F14 B5 F15 B6 F16 B7 F17 B8 F18 B9 F19 B10 F20 B11 F21 B12 F22 B13 F23 B14 F24 B15 F25 B16 F26 B17 F27 B18 F28 B19 F29 B20 F30 B21 F31 B22 B23 B24 B25 B26 B27 B28 B29 B30 B31
        F0 F1 F2 F3 F4 F5 F6 F7 F8 B0 F9 B1 F10 B2 F11 B3 F12 B4 F13 B5 F14 B6 F15 B7 F16 B8 F17 B9 F18 B10 F19 B11 F20 B12 F21 B13 F22 B14 F23 B15 F24 B16 F25 B17 F26 B18 F27 B19 F28 B20 F29 B21 F30 B22 F31 B23 B24 B25 B26 B27 B28 B29 B30 B31
        F0 F1 F2 F3 F4 F5 F6 F7 B0 F8 B1 F9 B2 F10 B3 F11 B4 F12 B5 F13 B6 F14 B7 F15 B8 F16 B9 F17 B10 F18 B11 F19 B12 F20 B13 F21 B14 F22 B15 F23 B16 F24 B17 F25 B18 F26 B19 F27 B20 F28 B21 F29 B22 F30 B23 F31 B24 B25 B26 B27 B28 B29 B30 B31
        F0 F1 F2 F3 F4 F5 F6 B0 F7 B1 F8 B2 F9 B3 F10 B4 F11 B5 F12 B6 F13 B7 F14 B8 F15 B9 F16 B10 F17 B11 F18 B12 F19 B13 F20 B14 F21 B15 F22 B16 F23 B17 F24 B18 F25 B19 F26 B20 F27 B21 F28 B22 F29 B23 F30 B24 F31 B25 B26 B27 B28 B29 B30 B31
        F0 F1 F2 F3 F4 F5 B0 F6 B1 F7 B2 F8 B3 F9 B4 F10 B5 F11 B6 F12 B7 F13 B8 F14 B9 F15 B10 F16 B11 F17 B12 F18 B13 F19 B14 F20 B15 F21 B16 F22 B17 F23 B18 F24 B19 F25 B20 F26 B21 F27 B22 F28 B23 F29 B24 F30 B25 F31 B26 B27 B28 B29 B30 B31
        F0 F1 F2 F3 F4 B0 F5 B1 F6 B2 F7 B3 F8 B4 F9 B5 F10 B6 F11 B7 F12 B8 F13 B9 F14 B10 F15 B11 F16 B12 F17 B13 F18 B14 F19 B15 F20 B16 F21 B17 F22 B18 F23 B19 F24 B20 F25 B21 F26 B22 F27 B23 F28 B24 F29 B25 F30 B26 F31 B27 B28 B29 B30 B31
        F0 F1 F2 F3 B0 F4 B1 F5 B2 F6 B3 F7 B4 F8 B5 F9 B6 F10 B7 F11 B8 F12 B9 F13 B10 F14 B11 F15 B12 F16 B13 F17 B14 F18 B15 F19 B16 F20 B17 F21 B18 F22 B19 F23 B20 F24 B21 F25 B22 F26 B23 F27 B24 F28 B25 F29 B26 F30 B27 F31 B28 B29 B30 B31
        F0 F1 F2 B0 F3 B1 F4 B2 F5 B3 F6 B4 F7 B5 F8 B6 F9 B7 F10 B8 F11 B9 F12 B10 F13 B11 F14 B12 F15 B13 F16 B14 F17 B15 F18 B16 F19 B17 F20 B18 F21 B19 F22 B20 F23 B21 F24 B22 F25 B23 F26 B24 F27 B25 F28 B26 F29 B27 F30 B28 F31 B29 B30 B31
        F0 F1 F2 B0 B1 F3 B2 F4 B3 F5 B4 F6 B5 F7 B6 F8 B7 F9 B8 F10 B9 F11 B10 F12 B11 F13 B12 F14 B13 F15 B14 F16 B15 F17 B16 F18 B17 F19 B18 F20 B19 F21 B20 F22 B21 F23 B22 F24 B23 F25 B24 F26 B25 F27 B26 F28 B27 F29 B28 F30 B29 F31 B30 B31
        F0 B0 F1 B1 F2 B2 F3 B3 F4 B4 F5 B5 F6 B6 F7 B7 F8 B8 F9 B9 F10 B10 F11 B11 F12 B12 F13 B13 F14 B14 F15 B15 F16 B16 F17 B17 F18 B18 F19 B19 F20 B20 F21 B21 F22 B22 F23 B23 F24 B24 F25 B25 F26 B26 F27 B27 F28 B28 F29 B29 F30 B30 F31 B31
        """
    pp_schedule = [line.strip().split() for line in raw.strip().split('\n')]
    data_schedule = []
    for row in pp_schedule:
        parsed_row = []
        for item in row:
            flag = item[0]
            number = int(item[1:])
            parsed_row.append((flag, number))
        data_schedule.append(parsed_row)

    return data_schedule

def generate_s16_mb32_test_schedule_aa() -> List[List[Tuple[str, int]]]:
    if True:
        raw = """
        F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 B0 F16 B1 F17 B2 F18 B3 F19 B4 F20 B5 F21 B6 F22 B7 F23 B8 F24 B9 F25 B10 F26 B11 F27 B12 F28 B13 F29 B14 F30 B15 F31 B16 B17 B18 B19 B20 B21 B22 B23 B24 B25 B26 B27 B28 B29 B30 B31
        F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 B0 F15 B1 F16 B2 F17 B3 F18 B4 F19 B5 F20 B6 F21 B7 F22 B8 F23 B9 F24 B10 F25 B11 F26 B12 F27 B13 F28 B14 F29 B15 F30 B16 F31 B17 B18 B19 B20 B21 B22 B23 B24 B25 B26 B27 B28 B29 B30 B31
        F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 B0 F14 B1 F15 B2 F16 B3 F17 B4 F18 B5 F19 B6 F20 B7 F21 B8 F22 B9 F23 B10 F24 B11 F25 B12 F26 B13 F27 B14 F28 B15 F29 B16 F30 B17 F31 B18 B19 B20 B21 B22 B23 B24 B25 B26 B27 B28 B29 B30 B31
        F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 B0 F13 B1 F14 B2 F15 B3 F16 B4 F17 B5 F18 B6 F19 B7 F20 B8 F21 B9 F22 B10 F23 B11 F24 B12 F25 B13 F26 B14 F27 B15 F28 B16 F29 B17 F30 B18 F31 B19 B20 B21 B22 B23 B24 B25 B26 B27 B28 B29 B30 B31
        F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 B0 F12 B1 F13 B2 F14 B3 F15 B4 F16 B5 F17 B6 F18 B7 F19 B8 F20 B9 F21 B10 F22 B11 F23 B12 F24 B13 F25 B14 F26 B15 F27 B16 F28 B17 F29 B18 F30 B19 F31 B20 B21 B22 B23 B24 B25 B26 B27 B28 B29 B30 B31
        F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 B0 F11 B1 F12 B2 F13 B3 F14 B4 F15 B5 F16 B6 F17 B7 F18 B8 F19 B9 F20 B10 F21 B11 F22 B12 F23 B13 F24 B14 F25 B15 F26 B16 F27 B17 F28 B18 F29 B19 F30 B20 F31 B21 B22 B23 B24 B25 B26 B27 B28 B29 B30 B31
        F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 B0 F10 B1 F11 B2 F12 B3 F13 B4 F14 B5 F15 B6 F16 B7 F17 B8 F18 B9 F19 B10 F20 B11 F21 B12 F22 B13 F23 B14 F24 B15 F25 B16 F26 B17 F27 B18 F28 B19 F29 B20 F30 B21 F31 B22 B23 B24 B25 B26 B27 B28 B29 B30 B31
        F0 F1 F2 F3 F4 F5 F6 F7 F8 B0 F9 B1 F10 B2 F11 B3 F12 B4 F13 B5 F14 B6 F15 B7 F16 B8 F17 B9 F18 B10 F19 B11 F20 B12 F21 B13 F22 B14 F23 B15 F24 B16 F25 B17 F26 B18 F27 B19 F28 B20 F29 B21 F30 B22 F31 B23 B24 B25 B26 B27 B28 B29 B30 B31
        F0 F1 F2 F3 F4 F5 F6 F7 B0 F8 B1 F9 B2 F10 B3 F11 B4 F12 B5 F13 B6 F14 B7 F15 B8 F16 B9 F17 B10 F18 B11 F19 B12 F20 B13 F21 B14 F22 B15 F23 B16 F24 B17 F25 B18 F26 B19 F27 B20 F28 B21 F29 B22 F30 B23 F31 B24 B25 B26 B27 B28 B29 B30 B31
        F0 F1 F2 F3 F4 F5 F6 B0 F7 B1 F8 B2 F9 B3 F10 B4 F11 B5 F12 B6 F13 B7 F14 B8 F15 B9 F16 B10 F17 B11 F18 B12 F19 B13 F20 B14 F21 B15 F22 B16 F23 B17 F24 B18 F25 B19 F26 B20 F27 B21 F28 B22 F29 B23 F30 B24 F31 B25 B26 B27 B28 B29 B30 B31
        F0 F1 F2 F3 F4 F5 B0 F6 B1 F7 B2 F8 B3 F9 B4 F10 B5 F11 B6 F12 B7 F13 B8 F14 B9 F15 B10 F16 B11 F17 B12 F18 B13 F19 B14 F20 B15 F21 B16 F22 B17 F23 B18 F24 B19 F25 B20 F26 B21 F27 B22 F28 B23 F29 B24 F30 B25 F31 B26 B27 B28 B29 B30 B31
        F0 F1 F2 F3 F4 B0 F5 B1 F6 B2 F7 B3 F8 B4 F9 B5 F10 B6 F11 B7 F12 B8 F13 B9 F14 B10 F15 B11 F16 B12 F17 B13 F18 B14 F19 B15 F20 B16 F21 B17 F22 B18 F23 B19 F24 B20 F25 B21 F26 B22 F27 B23 F28 B24 F29 B25 F30 B26 F31 B27 B28 B29 B30 B31
        F0 F1 F2 F3 B0 F4 B1 F5 B2 F6 B3 F7 B4 F8 B5 F9 B6 F10 B7 F11 B8 F12 B9 F13 B10 F14 B11 F15 B12 F16 B13 F17 B14 F18 B15 F19 B16 F20 B17 F21 B18 F22 B19 F23 B20 F24 B21 F25 B22 F26 B23 F27 B24 F28 B25 F29 B26 F30 B27 F31 B28 B29 B30 B31
        F0 F1 F2 F3 B0 B1 F4 B2 F5 B3 F6 B4 F7 B5 F8 B6 F9 B7 F10 B8 F11 B9 F12 B10 F13 B11 F14 B12 F15 B13 F16 B14 F17 B15 F18 B16 F19 B17 F20 B18 F21 B19 F22 B20 F23 B21 F24 B22 F25 B23 F26 B24 F27 B25 F28 B26 F29 B27 F30 B28 F31 B29 B30 B31
        F0 F1 F2 F3 B0 B1 B2 F4 B3 F5 B4 F6 B5 F7 B6 F8 B7 F9 B8 F10 B9 F11 B10 F12 B11 F13 B12 F14 B13 F15 B14 F16 B15 F17 B16 F18 B17 F19 B18 F20 B19 F21 B20 F22 B21 F23 B22 F24 B23 F25 B24 F26 B25 F27 B26 F28 B27 F29 B28 F30 B29 F31 B30 B31
        F0 B0 F1 B1 F2 B2 F3 B3 F4 B4 F5 B5 F6 B6 F7 B7 F8 B8 F9 B9 F10 B10 F11 B11 F12 B12 F13 B13 F14 B14 F15 B15 F16 B16 F17 B17 F18 B18 F19 B19 F20 B20 F21 B21 F22 B22 F23 B23 F24 B24 F25 B25 F26 B26 F27 B27 F28 B28 F29 B29 F30 B30 F31 B31
        """
    pp_schedule = [line.strip().split() for line in raw.strip().split('\n')]
    data_schedule = []
    for row in pp_schedule:
        parsed_row = []
        for item in row:
            flag = item[0]
            number = int(item[1:])
            parsed_row.append((flag, number))
        data_schedule.append(parsed_row)

    return data_schedule

def generate_s16_mb32_test_schedule_ab() -> List[List[Tuple[str, int]]]:
    if True:
        raw = """
        F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15 B0 F16 B1 F17 B2 F18 B3 F19 B4 F20 B5 F21 B6 F22 B7 F23 B8 F24 B9 F25 B10 F26 B11 F27 B12 F28 B13 F29 B14 F30 B15 F31 B16 B17 B18 B19 B20 B21 B22 B23 B24 B25 B26 B27 B28 B29 B30 B31
        F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 B0 F15 B1 F16 B2 F17 B3 F18 B4 F19 B5 F20 B6 F21 B7 F22 B8 F23 B9 F24 B10 F25 B11 F26 B12 F27 B13 F28 B14 F29 B15 F30 B16 F31 B17 B18 B19 B20 B21 B22 B23 B24 B25 B26 B27 B28 B29 B30 B31
        F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 B0 F14 B1 F15 B2 F16 B3 F17 B4 F18 B5 F19 B6 F20 B7 F21 B8 F22 B9 F23 B10 F24 B11 F25 B12 F26 B13 F27 B14 F28 B15 F29 B16 F30 B17 F31 B18 B19 B20 B21 B22 B23 B24 B25 B26 B27 B28 B29 B30 B31
        F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 B0 F13 B1 F14 B2 F15 B3 F16 B4 F17 B5 F18 B6 F19 B7 F20 B8 F21 B9 F22 B10 F23 B11 F24 B12 F25 B13 F26 B14 F27 B15 F28 B16 F29 B17 F30 B18 F31 B19 B20 B21 B22 B23 B24 B25 B26 B27 B28 B29 B30 B31
        F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 B0 F12 B1 F13 B2 F14 B3 F15 B4 F16 B5 F17 B6 F18 B7 F19 B8 F20 B9 F21 B10 F22 B11 F23 B12 F24 B13 F25 B14 F26 B15 F27 B16 F28 B17 F29 B18 F30 B19 F31 B20 B21 B22 B23 B24 B25 B26 B27 B28 B29 B30 B31
        F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 B0 F11 B1 F12 B2 F13 B3 F14 B4 F15 B5 F16 B6 F17 B7 F18 B8 F19 B9 F20 B10 F21 B11 F22 B12 F23 B13 F24 B14 F25 B15 F26 B16 F27 B17 F28 B18 F29 B19 F30 B20 F31 B21 B22 B23 B24 B25 B26 B27 B28 B29 B30 B31
        F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 B0 F10 B1 F11 B2 F12 B3 F13 B4 F14 B5 F15 B6 F16 B7 F17 B8 F18 B9 F19 B10 F20 B11 F21 B12 F22 B13 F23 B14 F24 B15 F25 B16 F26 B17 F27 B18 F28 B19 F29 B20 F30 B21 F31 B22 B23 B24 B25 B26 B27 B28 B29 B30 B31
        F0 F1 F2 F3 F4 F5 F6 F7 F8 B0 F9 B1 F10 B2 F11 B3 F12 B4 F13 B5 F14 B6 F15 B7 F16 B8 F17 B9 F18 B10 F19 B11 F20 B12 F21 B13 F22 B14 F23 B15 F24 B16 F25 B17 F26 B18 F27 B19 F28 B20 F29 B21 F30 B22 F31 B23 B24 B25 B26 B27 B28 B29 B30 B31
        F0 F1 F2 F3 F4 F5 F6 F7 B0 F8 B1 F9 B2 F10 B3 F11 B4 F12 B5 F13 B6 F14 B7 F15 B8 F16 B9 F17 B10 F18 B11 F19 B12 F20 B13 F21 B14 F22 B15 F23 B16 F24 B17 F25 B18 F26 B19 F27 B20 F28 B21 F29 B22 F30 B23 F31 B24 B25 B26 B27 B28 B29 B30 B31
        F0 F1 F2 F3 F4 F5 F6 B0 F7 B1 F8 B2 F9 B3 F10 B4 F11 B5 F12 B6 F13 B7 F14 B8 F15 B9 F16 B10 F17 B11 F18 B12 F19 B13 F20 B14 F21 B15 F22 B16 F23 B17 F24 B18 F25 B19 F26 B20 F27 B21 F28 B22 F29 B23 F30 B24 F31 B25 B26 B27 B28 B29 B30 B31
        F0 F1 F2 F3 F4 F5 B0 F6 B1 F7 B2 F8 B3 F9 B4 F10 B5 F11 B6 F12 B7 F13 B8 F14 B9 F15 B10 F16 B11 F17 B12 F18 B13 F19 B14 F20 B15 F21 B16 F22 B17 F23 B18 F24 B19 F25 B20 F26 B21 F27 B22 F28 B23 F29 B24 F30 B25 F31 B26 B27 B28 B29 B30 B31
        F0 F1 F2 F3 F4 B0 F5 B1 F6 B2 F7 B3 F8 B4 F9 B5 F10 B6 F11 B7 F12 B8 F13 B9 F14 B10 F15 B11 F16 B12 F17 B13 F18 B14 F19 B15 F20 B16 F21 B17 F22 B18 F23 B19 F24 B20 F25 B21 F26 B22 F27 B23 F28 B24 F29 B25 F30 B26 F31 B27 B28 B29 B30 B31
        F0 F1 F2 F3 F4 B0 B1 F5 B2 F6 B3 F7 B4 F8 B5 F9 B6 F10 B7 F11 B8 F12 B9 F13 B10 F14 B11 F15 B12 F16 B13 F17 B14 F18 B15 F19 B16 F20 B17 F21 B18 F22 B19 F23 B20 F24 B21 F25 B22 F26 B23 F27 B24 F28 B25 F29 B26 F30 B27 F31 B28 B29 B30 B31
        F0 F1 F2 F3 F4 B0 B1 B2 F5 B3 F6 B4 F7 B5 F8 B6 F9 B7 F10 B8 F11 B9 F12 B10 F13 B11 F14 B12 F15 B13 F16 B14 F17 B15 F18 B16 F19 B17 F20 B18 F21 B19 F22 B20 F23 B21 F24 B22 F25 B23 F26 B24 F27 B25 F28 B26 F29 B27 F30 B28 F31 B29 B30 B31
        F0 F1 F2 F3 F4 B0 B1 B2 B3 F5 B4 F6 B5 F7 B6 F8 B7 F9 B8 F10 B9 F11 B10 F12 B11 F13 B12 F14 B13 F15 B14 F16 B15 F17 B16 F18 B17 F19 B18 F20 B19 F21 B20 F22 B21 F23 B22 F24 B23 F25 B24 F26 B25 F27 B26 F28 B27 F29 B28 F30 B29 F31 B30 B31
        F0 B0 F1 B1 F2 B2 F3 B3 F4 B4 F5 B5 F6 B6 F7 B7 F8 B8 F9 B9 F10 B10 F11 B11 F12 B12 F13 B13 F14 B14 F15 B15 F16 B16 F17 B17 F18 B18 F19 B19 F20 B20 F21 B21 F22 B22 F23 B23 F24 B24 F25 B25 F26 B26 F27 B27 F28 B28 F29 B29 F30 B30 F31 B31
        """
    pp_schedule = [line.strip().split() for line in raw.strip().split('\n')]
    data_schedule = []
    for row in pp_schedule:
        parsed_row = []
        for item in row:
            flag = item[0]
            number = int(item[1:])
            parsed_row.append((flag, number))
        data_schedule.append(parsed_row)

    return data_schedule

def generate_s6_mb12_test_schedule() -> List[List[Tuple[str, int]]]:
    if False:  #远古换一列 or 阻塞远古
        raw = """
        F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 B0 F11 B1 B2 B3 B4 B5 B6 B7 B8 B9 B10 B11
        F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 B0 F11 B1 B2 B3 B4 B5 B6 B7 B8 B9 B10 B11
        F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 B0 F11 B1 B2 B3 B4 B5 B6 B7 B8 B9 B10 B11
        F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 B0 F11 B1 B2 B3 B4 B5 B6 B7 B8 B9 B10 B11
        F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 B0 F11 B1 B2 B3 B4 B5 B6 B7 B8 B9 B10 B11
        F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 B0 F10 F11 B1 B2 B3 B4 B5 B6 B7 B8 B9 B10 B11
        """
    elif False:  #远古
        raw = """
        F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 B0 B1 B2 B3 B4 B5 B6 B7 B8 B9 B10 B11
        F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 B0 B1 B2 B3 B4 B5 B6 B7 B8 B9 B10 B11
        F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 B0 B1 B2 B3 B4 B5 B6 B7 B8 B9 B10 B11
        F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 B0 B1 B2 B3 B4 B5 B6 B7 B8 B9 B10 B11
        F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 B0 B1 B2 B3 B4 B5 B6 B7 B8 B9 B10 B11
        F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 B0 B1 B2 B3 B4 B5 B6 B7 B8 B9 B10 B11
        """
        # send 之后执行，recv之前执行
        raw_mask_sendF = """
        1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 
        1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 
        1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 
        1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 
        1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 
        1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 
        """
        raw_mask_recvF = """
        1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 
        1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 
        1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 
        1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 
        1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 
        1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 
        """
        raw_mask_sendB = """
        0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1  
        0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1  
        0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1  
        0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1  
        0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1  
        0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1  
        """
        raw_mask_recvB = """
        0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1  
        0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1  
        0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1  
        0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1  
        0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1  
        0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1  
        """
    elif False: #1f1b normal
        raw = """
        F0 F1 F2 F3 F4 F5 B0 F6 B1 F7 B2 F8 B3 F9 B4 F10 B5 F11 B6 B7 B8 B9 B10 B11
        F0 F1 F2 F3 F4 B0 F5 B1 F6 B2 F7 B3 F8 B4 F9 B5 F10 B6 F11 B7 B8 B9 B10 B11
        F0 F1 F2 F3 B0 F4 B1 F5 B2 F6 B3 F7 B4 F8 B5 F9 B6 F10 B7 F11 B8 B9 B10 B11
        F0 F1 F2 B0 F3 B1 F4 B2 F5 B3 F6 B4 F7 B5 F8 B6 F9 B7 F10 B8 F11 B9 B10 B11
        F0 F1 B0 F2 B1 F3 B2 F4 B3 F5 B4 F6 B5 F7 B6 F8 B7 F9 B8 F10 B9 F11 B10 B11
        F0 B0 F1 B1 F2 B2 F3 B3 F4 B4 F5 B5 F6 B6 F7 B7 F8 B8 F9 B9 F10 B10 F11 B11
        """
        raw_mask_sendF = """
        1 1 1 1 1 1 0 1 0 1 0 1 0 1 0 1 0 1 0 0 0 0 0 0 
        1 1 1 1 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 0 0 0 0 
        1 1 1 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 0 0 0 
        1 1 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 0 0 
        1 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 0 
        1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 
        """
        raw_mask_recvF = """
        1 1 1 1 1 1 0 1 0 1 0 1 0 1 0 1 0 1 0 0 0 0 0 0 
        1 1 1 1 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 0 0 0 0 
        1 1 1 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 0 0 0 
        1 1 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 0 0 
        1 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 0 
        1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 
        """
        raw_mask_sendB = """
        0 0 0 0 0 0 1 0 1 0 1 0 1 0 1 0 1 0 1 1 1 1 1 1
        0 0 0 0 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 1 1 1 1
        0 0 0 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 1 1 1
        0 0 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 1 1
        0 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 1
        0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1
        """
        raw_mask_recvB = """
        0 0 0 0 0 0 1 0 1 0 1 0 1 0 1 0 1 0 1 1 1 1 1 1
        0 0 0 0 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 1 1 1 1
        0 0 0 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 1 1 1
        0 0 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 1 1
        0 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 1
        0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1
        """
    elif False: #cacu 6 12  time equal, F16 B44 comm0
        raw = """
        F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 B0 B1 B2 B3 B4 B5 B6 B7 B8 B9 B10 B11
        F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 B0 B1 B2 B3 B4 B5 B6 B7 B8 B9 B10 B11
        F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 B0 B1 B2 B3 B4 B5 B6 B7 B8 B9 B10 B11
        F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 B0 F11 B1 B2 B3 B4 B5 B6 B7 B8 B9 B10 B11
        F0 F1 F2 F3 F4 F5 B0 F6 B1 F7 B2 F8 B3 B4 F9 F10 B5 F11 B6 B7 B8 B9 B10 B11
        F0 F1 B0 F2 B1 F3 B2 F4 B3 F5 B4 F6 B5 F7 F8 B6 F9 B7 F10 F11 B8 B9 B10 B11
        """
    elif True: # cacu 6 12 ; time equal, F16 B44 comm0
        raw = """
        F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 B0 B1 B2 B3 B4 B5 B6 B7 B8 B9 B10 B11
        F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 B0 B1 B2 B3 B4 B5 B6 B7 B8 B9 B10 B11
        F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 B0 B1 B2 B3 B4 B5 B6 B7 B8 B9 B10 B11
        F0 F1 F2 F3 F4 F5 F6 F7 F8 B0 F9 B1 B2 F10 B3 B4 B5 B6 F11 B7 B8 B9 B10 B11
        F0 F1 F2 F3 F4 B0 F5 F6 F7 B1 F8 B2 B3 F9 F10 B4 B5 B6 B7 B8 F11 B9 B10 B11
        F0 B0 F1 F2 F3 B1 B2 F4 F5 B3 F6 F7 B4 F8 F9 B5 B6 B7 B8 F10 B9 F11 B10 B11
        """
        raw_mask_sendF = """
        1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 
        1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 
        1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 
        1 1 1 1 1 1 1 1 1 0 1 0 0 1 0 0 0 0 1 0 0 0 0 0
        1 1 1 1 1 0 1 1 1 0 1 0 0 1 1 0 0 0 0 0 1 0 0 0
        1 0 1 1 1 0 0 1 1 0 1 1 0 1 1 0 0 0 0 1 0 1 0 0
        """
        raw_mask_recvF = """
        1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 
        1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 
        1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 
        1 1 1 1 1 1 1 1 1 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
        1 1 1 1 1 3 0 0 1 0 1 0 0 1 0 0 0 0 1 0 0 0 0 0
        1 3 0 0 4 0 0 0 0 0 1 0 0 1 1 0 0 0 0 0 1 0 0 0
        """
        raw_mask_sendB = """
        0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1  
        0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1  
        0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1  
        0 0 0 0 0 0 0 0 0 1 0 1 1 0 1 1 1 1 0 1 1 1 1 1  
        0 0 0 0 0 1 0 0 0 1 0 1 1 0 0 1 1 1 1 1 0 1 1 1  
        0 1 0 0 0 1 1 0 0 1 0 0 1 0 0 1 1 1 1 0 1 0 1 1  
        """
        raw_mask_recvB = """
        0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1  
        0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1  
        0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1  
        0 0 0 0 0 0 0 0 1 0 0 1 1 0 1 1 1 1 0 1 1 1 1 1  
        0 0 0 0 1 0 0 0 0 1 0 1 1 0 0 1 1 1 1 1 0 1 1 1  
        0 1 0 0 0 1 1 0 0 1 0 0 1 0 0 1 1 1 1 0 1 0 1 1  
        """
    else:
        assert False

    pp_schedule = [line.strip().split() for line in raw.strip().split('\n')]
    data_schedule = []
    for row in pp_schedule:
        parsed_row = []
        for item in row:
            flag = item[0]
            number = int(item[1:])
            parsed_row.append((flag, number))
        data_schedule.append(parsed_row)

    return data_schedule

def generate_1f1b_schedule(num_stages: int, num_microbatches: int) -> List[List[Tuple[str, int]]]:
    schedule = [[] for _ in range(num_stages)]
    
    for stage in range(num_stages):
        num_warmup = num_stages - stage - 1
        
        for mb in range(num_warmup):
            schedule[stage].append(('F', mb))
        
        for mb in range(num_warmup, num_microbatches):
            schedule[stage].append(('F', mb))
            backward_mb = mb - num_warmup
            schedule[stage].append(('B', backward_mb))
        
        for mb in range(num_microbatches - num_warmup, num_microbatches):
            schedule[stage].append(('B', mb))
    
    return schedule

def parse_schedule(schedule: List[List[Tuple[str, int]]], Ftime: List[float], Btime: List[float]) -> List[List[Tuple[float, float]]]:
    num_stages = len(schedule)
    assert num_stages == len(Ftime)
    assert num_stages == len(Btime)

    time_table = [[(None, None) for _ in stage] for stage in schedule]
    time_table[0][0] = (0.0, Ftime[0])
     
    flags = [0 for _ in range(num_stages)]
    flags[0] = 1
    
    looptimes = 0
    while any(flags[i] < len(schedule[i]) for i in range(num_stages)):
        #print('ff')
        successtime = 0
        for stage_idx in range(num_stages):
            stage = schedule[stage_idx]
            if flags[stage_idx] >= len(stage):
                continue
                
            handled_idx = flags[stage_idx]
            op, mb = stage[handled_idx]
            
            dependencies = []
            
            # Add dependency from previous operation in same stage
            if handled_idx > 0: 
                prev_end = time_table[stage_idx][handled_idx-1][1]
                if prev_end is not None:
                    dependencies.append(prev_end)
            
            # Add cross-stage dependencies
            # TODO 这里可以优化，不需要遍历所有
            shouldlen = 0
            if op == 'F':
                if stage_idx > 0:
                    shouldlen = 2
                    if handled_idx == 0:
                        shouldlen = 1
                    for prev_op_idx, (prev_op, prev_mb) in enumerate(schedule[stage_idx-1]):
                        if prev_op == 'F' and prev_mb == mb:
                            prev_end = time_table[stage_idx-1][prev_op_idx][1]
                            if prev_end is not None:
                                dependencies.append(prev_end)
                            break
                else:
                    shouldlen = 1
                    
            else:
                if stage_idx < num_stages - 1:
                    shouldlen = 2
                    if handled_idx == 0:
                        shouldlen = 1
                    for next_op_idx, (next_op, next_mb) in enumerate(schedule[stage_idx+1]):
                        if next_op == 'B' and next_mb == mb:
                            next_end = time_table[stage_idx+1][next_op_idx][1]
                            if next_end is not None:
                                dependencies.append(next_end)
                            break
                else: 
                    shouldlen = 1 
            #TODO 这里不对，应该是需要的dependencies 全了才能确定
            #TODO 直接根据位置确定，如果是F且第一行 则只要一个元素。否则要两个，然后才能赋值
            #print(f'lendepend = {len(dependencies)} should = {shouldlen}')
            if len(dependencies) == shouldlen:
                successtime += 1
                start_time = max(dependencies)
                if op == 'F':
                    end_time = start_time + Ftime[stage_idx]
                else:
                    end_time = start_time + Btime[stage_idx]
                time_table[stage_idx][handled_idx] = (start_time, end_time)
                if flags[stage_idx] != handled_idx:
                    print(f'flags = {flags[stage_idx]} next =  {handled_idx}')
                    assert False

                flags[stage_idx] = handled_idx + 1

        if successtime == 0:
            assert False

        #for stage_idx, stage in enumerate(schedule):
        #    print(f' stageidx = {stage_idx} flags = {flags[stage_idx]}')

        #print("\n## Timing Information:")
        #for stage_idx, stage in enumerate(schedule):
        #    for op_idx, (op, mb) in enumerate(stage):
        #        start_time, end_time = time_table[stage_idx][op_idx]
        #        print(f"S {stage_idx} {op} {mb} begin {start_time}")
        #        print(f"S {stage_idx} {op} {mb} end {end_time}")
    
    return time_table

def generate_communication_masks(schedule: List[List[Tuple[str, int]]]) -> Dict[str, List[List[int]]]:
    num_stages = len(schedule)
    max_steps = max(len(stage) for stage in schedule)
    
    sendF_mask = [[0 for _ in range(max_steps)] for _ in range(num_stages)]
    recvF_mask = [[0 for _ in range(max_steps)] for _ in range(num_stages)]
    sendB_mask = [[0 for _ in range(max_steps)] for _ in range(num_stages)]
    recvB_mask = [[0 for _ in range(max_steps)] for _ in range(num_stages)]
    
    for stage in range(num_stages):
        for step, (op, _) in enumerate(schedule[stage]):
            if op == 'F':
                sendF_mask[stage][step] = 1
                recvF_mask[stage][step] = 1
                sendB_mask[stage][step] = 0
                recvB_mask[stage][step] = 0
            else:
                sendF_mask[stage][step] = 0
                recvF_mask[stage][step] = 0
                sendB_mask[stage][step] = 1
                recvB_mask[stage][step] = 1
    
    return {
        'sendF': sendF_mask,
        'recvF': recvF_mask,
        'sendB': sendB_mask,
        'recvB': recvB_mask
    }

def generate_communication_masks_zerocross(schedule: List[List[Tuple[str, int]]], timetable: List[List[Tuple[float, float]]]) -> Dict[str, List[List[int]]]:
    num_stages = len(schedule)
    max_steps = max(len(stage) for stage in schedule)
    
    sendF_mask = [[0 for _ in range(max_steps)] for _ in range(num_stages)]
    recvF_mask = [[0 for _ in range(max_steps)] for _ in range(num_stages)]
    sendB_mask = [[0 for _ in range(max_steps)] for _ in range(num_stages)]
    recvB_mask = [[0 for _ in range(max_steps)] for _ in range(num_stages)]
    
    for stage in range(num_stages):
        for step, (op, _) in enumerate(schedule[stage]):
            if op == 'F':
                sendF_mask[stage][step] = 1
                sendB_mask[stage][step] = 0
                if stage == 0:
                    recvF_mask[stage][step] = 1
                if stage == num_stages-1:
                    recvB_mask[stage][step] = 0
            else:
                sendF_mask[stage][step] = 0
                sendB_mask[stage][step] = 1
                if stage == 0:
                    recvF_mask[stage][step] = 0
                if stage == num_stages -1:
                    recvB_mask[stage][step] = 1
    
    #理论上直接recv被send自然形成相同顺序
    for stage in range(1, num_stages):
        last_flag = 0
        this_flag = 0
        for step, (op, idmb) in enumerate(schedule[stage-1]):    
            if op == 'F':
                et = timetable[stage-1][step][1]
                while timetable[stage][this_flag][0] < et - 1e-3 and this_flag < max_steps:
                    this_flag += 1
                recvF_mask[stage][this_flag] += 1
        for step, (op, idmb) in enumerate(schedule[stage]):
            if op == 'B':
                et = timetable[stage][step][1]
                while timetable[stage-1][last_flag][0] < et - 1e-3 and last_flag < max_steps:
                    last_flag += 1
                recvB_mask[stage-1][last_flag] += 1 


    #for stage in range(1, num_stages):
    #    last_posF = 0
    #    this_posB = 0
    #    while True:
    #        while schedule[stage-1][last_posF] != 'F' and last_posF < max_steps-1:
    #            last_posF += 1
    #        while schedule[stage][this_posB] != 'B' and this_posB < max_steps -1:
    #            this_posB += 1
    #        if last_posF < max_steps and this_posB < max_steps
    #            last_et = timetable[stage-1][last_posF]
    #            this_et = timetable[stage][this_posB]
    #            if last_et < this_et
                    
                
    
    
    masks =  {
        'sendF': sendF_mask,
        'recvF': recvF_mask,
        'sendB': sendB_mask,
        'recvB': recvB_mask
    }

    return masks

def generate_communication_masks_1f1b_zerocross(schedule: List[List[Tuple[str, int]]]) -> Dict[str, List[List[int]]]:
    num_stages = len(schedule)
    max_steps = max(len(stage) for stage in schedule)
    
    sendF_mask = [[0 for _ in range(max_steps)] for _ in range(num_stages)]
    recvF_mask = [[0 for _ in range(max_steps)] for _ in range(num_stages)]
    sendB_mask = [[0 for _ in range(max_steps)] for _ in range(num_stages)]
    recvB_mask = [[0 for _ in range(max_steps)] for _ in range(num_stages)]
    
    for stage in range(num_stages):
        for step, (op, _) in enumerate(schedule[stage]):
            if op == 'F':
                sendF_mask[stage][step] = 1
                recvB_mask[stage][step] = 0
                sendB_mask[stage][step] = 0
                
                if stage == 0:
                    recvF_mask[stage][step] = 1

                if stage < (num_stages -1 ):
                    recvF_mask[stage+1][step] = 1
            else:
                sendF_mask[stage][step] = 0
                recvB_mask[stage][step] = 1
                sendB_mask[stage][step] = 1
                if stage == 0:
                    recvF_mask[stage][step] = 0
                if stage < (num_stages - 1):
                    recvF_mask[stage+1][step] = 0
    #for stage in range(1,num_stages):
    #    beginpos = num_stages + 1 - stage    
    #    for ii in range(beginpos, max_steps-1)
    #        recvF_mask[stage][ii] =recvF_mask[stage][ii+1]
            

    masks =  {
        'sendF': sendF_mask,
        'recvF': recvF_mask,
        'sendB': sendB_mask,
        'recvB': recvB_mask
    }

    return masks

def format_schedule_for_display(schedule: List[List[Tuple[str, int]]]) -> str:
    result = []
    for stage_idx, stage in enumerate(schedule):
        row = []
        for op, mb in stage:
            row.append(f"{op}{mb}")
        result.append(f"Stage {stage_idx}: " + " ".join(row))
    return "\n".join(result)

def format_mask_for_display(mask: List[List[int]], name: str) -> str:
    result = [f"{name} mask:"]
    for stage_idx, stage in enumerate(mask):
        result.append(f"Stage {stage_idx}: " + " ".join(map(str, stage)))
    return "\n".join(result)

def main():
    parser = argparse.ArgumentParser(description="Generate 1F1B pipeline parallel schedule")
    parser.add_argument("--stages", type=int, default=6, help="Number of pipeline stages")
    parser.add_argument("--microbatches", type=int, default=12, help="Number of microbatches")
    parser.add_argument("--output", type=str, default=None, help="Output file path (if not specified, print to stdout)")
    parser.add_argument("--print", action="store_true", help="Print the schedule details")
    args = parser.parse_args()
    
    #schedule = generate_1f1b_schedule(args.stages, args.microbatches)
    schedule = generate_s16_mb32_test_schedule()

    # Generate timing information
    Ftime = [1.0 for _ in range(args.stages)]  # Example forward times
    #Ftime = [1.0, 2.0, 3.0, 2.0, 2.0, 4.0]
    Btime = [2.0 for _ in range(args.stages)]  # Example backward times
    time_table = parse_schedule(schedule, Ftime, Btime)


    #masks = generate_communication_masks_1f1b_zerocross(schedule)
    masks = generate_communication_masks_zerocross(schedule, time_table)
    
    
    if args.print:
        print(f"# 1F1B Schedule: {args.stages} stages, {args.microbatches} microbatches")
        print("\n## Schedule:")
        if True:
            print(format_schedule_for_display(schedule))
        
        print("\n## Communication Masks:")
        if True:
            print(format_mask_for_display(masks['sendF'], "Send Forward"))
            print(format_mask_for_display(masks['recvF'], "Receive Forward"))
            print(format_mask_for_display(masks['sendB'], "Send Backward"))
            print(format_mask_for_display(masks['recvB'], "Receive Backward"))
        
        print("\n## Timing Information:")
        if False:
            for stage_idx, stage in enumerate(schedule):
                for op_idx, (op, mb) in enumerate(stage):
                    start_time, end_time = time_table[stage_idx][op_idx]
                    print(f"S {stage_idx} {op} {mb} begin {start_time}")
                    print(f"S {stage_idx} {op} {mb} end {end_time}")
    
    if args.output:
        output = []
        output.append("raw = \"\"\"")
        for stage in schedule:
            output.append("    " + " ".join([f"{op}{mb}" for op, mb in stage]))
        output.append("\"\"\"")
        
        for mask_name in ['sendF', 'recvF', 'sendB', 'recvB']:
            output.append(f"raw_mask_{mask_name} = \"\"\"")
            for stage in masks[mask_name]:
                output.append("    " + " ".join(map(str, stage)))
            output.append("\"\"\"")
        
        # Add timing information to output
        output.append("\n# Timing Information:")
        for stage_idx, stage in enumerate(schedule):
            for op_idx, (op, mb) in enumerate(stage):
                start_time, end_time = time_table[stage_idx][op_idx]
                output.append(f"S {stage_idx} {op} {mb} begin {start_time}")
                output.append(f"S {stage_idx} {op} {mb} end {end_time}")
        
        output_str = "\n".join(output)
        with open(args.output, "w") as f:
            f.write(output_str)
        print(f"Schedule written to {args.output}")
        
    return schedule, masks, time_table

if __name__ == "__main__":
    main()


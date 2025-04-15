#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
该脚本读取 SFT 的 JSON 数据文件，使用 Megatron-LM 的 tokenizer 进行 token 化，
按照预设的对话模板处理后，将每个样本固定长度化（max_padding_length），
将所有样本依次拼接写入一个二进制文件，并生成对应的索引文件。
"""

import os
import io
import json
import numpy as np
import torch
import argparse
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer

def preprocess_sample(messages, tokenizer, max_padding_length):
    """
    对单个样本进行预处理：利用 tokenizer.apply_chat_template 对整个对话进行 token 化，
    并根据对 assistant 回复的标记生成 label。对长度超过 max_padding_length 的样本进行截断，
    对不足的样本进行 padding（使用 tokenizer.pad 对应的标记）。

    Args:
        messages: 对话样本，列表形式，每个元素为包含 "role" 等信息的字典。
        tokenizer: Megatron-LM 提供的 tokenizer，必须具有 apply_chat_template 方法，
                   以及 pad 和 eod 属性。
        max_padding_length: 固定序列的最大长度。

    Returns:
        tokens: numpy.array，预处理后的 token id 序列（长度固定为 max_padding_length）。
        label:  numpy.array，与 tokens 对应的 label 序列，非 assistant 部分置为 IGNORE_INDEX。
        如果样本没有有效 assistant 回复，则返回 None。
    """
    IGNORE_INDEX = tokenizer.pad_token_id
    eos_token_id = tokenizer.eos_token_id
    is_pad_token_eos_token = (tokenizer.pad == eos_token_id)
    
    # 对整个对话应用模板
    full = tokenizer.apply_chat_template(messages)
    # 去掉末尾4个token（例如模板尾部特殊标记）
    full = full[:-4]

    # 超长截断
    if len(full) > max_padding_length:
        full = full[:max_padding_length]
    elif is_pad_token_eos_token:
        # 保证未截断的样本最后一个 token 是 eos，否则报错
        assert full[-1] == eos_token_id, f"未截断样本应以 <eos> 结尾！当前结果: {tokenizer.detokenize(full)}"
        full[-1] = -1 - full[-1]

    # 初始化 label（全部置为 IGNORE_INDEX）
    label = [IGNORE_INDEX] * len(full)
    
    # 如果对话消息少于2条，则跳过该样本
    if len(messages) < 2:
        return None

    # 对每个 assistant 的回复（从第二条起）进行 label 标注
    for idx, msg in enumerate(messages):
        if msg['role'] != 'assistant' or idx < 1:
            continue
        # 生成当前消息为止的 token 序列
        partial_ids = tokenizer.apply_chat_template(messages[:idx+1])
        # 生成 assistant 回复之前的 token 序列
        prompt_ids = tokenizer.apply_chat_template(messages[:idx])
        current_end = len(partial_ids[:-4])
        start_idx = len(prompt_ids)
        end_idx = min(current_end, len(full))
        if start_idx < end_idx:
            label[start_idx:end_idx] = full[start_idx:end_idx]

    # 检查至少有一个 token 被标记为有效 label
    if all(x == IGNORE_INDEX for x in label):
        return None

    # 填充至固定长度
    if len(full) < max_padding_length:
        pad_len = max_padding_length - len(full)
        full = full + [IGNORE_INDEX] * pad_len
        label = label + [IGNORE_INDEX] * pad_len

    return np.array(full, dtype=np.int32), np.array(label, dtype=np.int32)

def preprocess_wrapper(sample, max_padding_length):
    """
    用于并行 map 的包装函数，每个进程中都会调用该函数：
      - 重新加载 tokenizer
      - 读取 sample 中的 messages 并调用 preprocess_sample 处理
      - 若处理成功，则将 numpy 数组转为列表（便于序列化）
    
    返回的字典中，若 input_ids 为 None 表示该样本应跳过。
    """
    # 每个进程中重新加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained("/mnt/data/lufanfeng/Megatron-Core/zjllm-llama3-tokenizer/")

    # 如果没有 eod 属性，则将 eos_token_id 赋值给 eod
    if not hasattr(tokenizer, "eod"):
        tokenizer.eod = tokenizer.eos_token_id

    # 如果没有 pad 属性，则设置 pad 为 pad_token_id，如果没有则设为默认值（例如 0）
    if not hasattr(tokenizer, "pad"):
        tokenizer.pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    messages = sample.get("messages", None)
    if messages is None:
        return {"input_ids": None, "labels": None}
    result = preprocess_sample(messages, tokenizer, max_padding_length)
    if result is None:
        return {"input_ids": None, "labels": None}
    tokens, labels = result
    # 转换为列表便于存储
    return {"input_ids": tokens.tolist(), "labels": labels.tolist()}

def pretokenize_data_parallel(input_file, max_padding_length, split='train', num_proc=8):
    """
    利用 Hugging Face datasets 的 map 接口并行处理 JSON 数据文件。

    Args:
        input_file: JSON 数据文件路径。
        max_padding_length: 固定序列长度。
        split: 数据集划分（如 "train", "valid", "test"）。
        num_proc: 并行处理的进程数。

    Returns:
        input_ids_list: List，每个元素为一个样本的 token id numpy 数组。
        labels_list: List，每个元素为一个样本的 label numpy 数组。
        offsets: List，每个样本在二进制文件中的起始偏移位置（单位：token 数）。
    """

    input_file = input_file.split(" ")
    dataset = load_dataset('json', data_files=input_file, split=split)

    # 并行 map 处理，注意 remove_columns 删除原有字段，避免重复
    processed_dataset = dataset.map(
         lambda x: preprocess_wrapper(x, max_padding_length),
         batched=False,
         num_proc=num_proc,
         remove_columns=dataset.column_names,
         load_from_cache_file=False,
         desc="Parallel Tokenization"
    )
    # 过滤掉无效样本（input_ids 为 None）
    processed_dataset = processed_dataset.filter(lambda x: x["input_ids"] is not None)

    # 提取处理后的样本
    input_ids_list = [np.array(x["input_ids"], dtype=np.int32) for x in processed_dataset]
    labels_list = [np.array(x["labels"], dtype=np.int32) for x in processed_dataset]

    # 生成索引，注意每个样本固定长度为 max_padding_length
    offsets = [0]
    for _ in range(len(input_ids_list)):
         offsets.append(offsets[-1] + max_padding_length)
    return input_ids_list, labels_list, offsets

def write_bin_and_idx(bin_path, idx_path, samples, offsets):
    """
    将样本列表写入二进制文件，并将样本在二进制文件中的起始偏移位置写入索引文件。

    Args:
        bin_path: 输出的二进制文件路径。
        idx_path: 输出的索引文件路径。
        samples: List，每个元素为一个 numpy 数组（样本）。
        offsets: List，每个样本在二进制文件中的起始偏移位置。
    """
    if len(samples) == 0:
        print("没有样本可写入。")
        return
    # 将所有样本拼接成一个长数组
    all_data = np.concatenate(samples, axis=0)
    all_data.astype(np.int32).tofile(bin_path)
    np.array(offsets, dtype=np.int64).tofile(idx_path)
    print(f"写入二进制文件: {bin_path}")
    print(f"写入索引文件: {idx_path}")


def pretokenize_data(input_file, max_padding_length, split='train'):
    """
    加载 JSON 数据文件，并对每个样本进行 token 化预处理。

    Args:
        input_file: JSON 数据文件路径。
        max_padding_length: 固定的序列长度。
        split: 数据集划分（如 "train", "valid", "test"）。

    Returns:
        input_ids_list: List，每个元素为一个样本的 token id numpy 数组。
        labels_list: List，每个元素为一个样本的 label numpy 数组。
        offsets: List，每个样本在二进制文件中的起始偏移位置（单位：token 数）。
    """
    # 每个进程中重新加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained("/mnt/data/lufanfeng/Megatron-Core/zjllm-llama3-tokenizer/")

    # 如果没有 eod 属性，则将 eos_token_id 赋值给 eod
    if not hasattr(tokenizer, "eod"):
        tokenizer.eod = tokenizer.eos_token_id

    # 如果没有 pad 属性，则设置 pad 为 pad_token_id，如果没有则设为默认值（例如 0）
    if not hasattr(tokenizer, "pad"):
        tokenizer.pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    input_file = input_file.split(" ")
    dataset = load_dataset('json', data_files=input_file, split=split)
    
    input_ids_list = []
    labels_list = []
    offsets = [0]  # 第一个样本从位置 0 开始

    print(f"开始处理 {input_file} 中的 {len(dataset)} 个样本 ...")
    for sample in tqdm(dataset):
        messages = sample.get("messages", None)
        if messages is None:
            continue
        processed = preprocess_sample(messages, tokenizer, max_padding_length)
        if processed is None:
            continue
        tokens, label = processed
        input_ids_list.append(tokens)
        labels_list.append(label)
        # 每个样本固定长度为 max_padding_length
        offsets.append(offsets[-1] + max_padding_length)

    print(f"有效样本数: {len(input_ids_list)}")
    return input_ids_list, labels_list, offsets

def main():
    parser = argparse.ArgumentParser(description="离线预处理 SFT JSON 数据，生成预tokenize后的 .bin 和 .idx 文件")
    parser.add_argument('--input_file', default="/mnt/data/lusongshuo/datasets/tulu3_mixed_part_1.json /mnt/data/lusongshuo/datasets/tulu3_mixed_part_2.json /mnt/data/lusongshuo/datasets/tulu3_mixed_part_3.json /mnt/data/lusongshuo/datasets/tulu3_mixed_part_4.json /mnt/data/lusongshuo/datasets/tulu3_mixed_part_5.json", type=str, help="SFT JSON 数据文件路径")
    parser.add_argument('--split', type=str, default='train', help="数据集划分（如 train, valid, test）")
    parser.add_argument('--max_padding_length', default=4096, type=int, help="固定序列最大长度")
    parser.add_argument('--output_dir', default="/mnt/workspace/lusongshuo/datasets", type=str, help="输出目录，用于存放生成的 .bin 和 .idx 文件")
    parser.add_argument('--num_proc', type=int, default=64, help="并行处理使用的进程数")
    args = parser.parse_args()

    input_ids_list, labels_list, offsets = pretokenize_data(args.input_file, args.max_padding_length, split=args.split)

    
    os.makedirs(args.output_dir, exist_ok=True)
        
    # 分别写入 input_ids 与 labels 文件
    input_bin_path = os.path.join(args.output_dir, f"tuluv3_{args.split}_input_ids.bin")
    input_idx_path = os.path.join(args.output_dir, f"tuluv3_{args.split}_input_ids.idx")
    label_bin_path = os.path.join(args.output_dir, f"tuluv3_{args.split}_labels.bin")
    label_idx_path = os.path.join(args.output_dir, f"tuluv3_{args.split}_labels.idx")
    
    write_bin_and_idx(input_bin_path, input_idx_path, input_ids_list, offsets)
    write_bin_and_idx(label_bin_path, label_idx_path, labels_list, offsets)
    
    print("预处理完成。")

if __name__ == '__main__':
    main()

# Copyright (c) 2025 Alibaba PAI Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import io
import copy
import json
import torch
from datasets import load_dataset
from tqdm import tqdm

from megatron.training import get_args
from megatron.training import get_tokenizer

class JSONSFTDataset(torch.utils.data.Dataset):
    """
    Experimental: This dataset is aimed for SFT of arbitrary models with a default chat_template, 
    but not tested on all cases.

    A class for processing a conversation dataset
    """

    def __init__(self, path, max_padding_length, split='train'):
        super().__init__()
        self.tokenizer = get_tokenizer()
        assert hasattr(self.tokenizer, 'apply_chat_template'), \
            "The SFT-Raw Dataset is valid for tokenizers with chat template, please provide a template."
        self.IGNORE_INDEX = self.tokenizer.pad
        self.eos_token_id = self.tokenizer.eod
        self.is_pad_token_eos_token = self.tokenizer.pad == self.eos_token_id
        self.max_padding_length = max_padding_length

        list_data_dict = load_dataset(
            'json',
            data_files=path,
            split=split,
        )

        train_dataset = list_data_dict.map(
            self.preprocess,
            batched=True,
            batch_size=3000,
            num_proc=128,
            remove_columns=list_data_dict.column_names,
            load_from_cache_file=False,
            desc="Running Encoding"
        )

        self.input_ids = np.array(train_dataset['input_ids'])
        self.labels = np.array(train_dataset['labels'])
        self.samples = []

        for inputs, labels in tqdm(zip(self.input_ids, self.labels)):
            self.samples.append([inputs, labels])

        print('  >> total number of samples: {}'.format(len(self.samples)))

    def _make_r_io_base(self, f, mode: str):
        if not isinstance(f, io.IOBase):
            f = open(f, mode=mode, encoding='utf-8')
        return f

    def jload(self, f, mode='r'):
        """
        Load a .json file into a dictionary.
        Args:
            f: The file object or string representing the file path.
            mode: The mode in which to open the file (e.g., 'r', 'w', 'a').
        Returns:
            A dictionary containing the contents of the JSON file.
        """
        f = self._make_r_io_base(f, mode)
        jdict = json.load(f)
        f.close()
        return jdict

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        raw_sample = self.samples[idx]
        return self.gpt_convert_example_to_feature(raw_sample)

    def preprocess(self, examples):
        """
        Preprocess the data by tokenizing.
        Args:
            sources (List[str]): a list of source strings
            targets (List[str]): a list of target strings
            tokenizer (Tokenizer): a tokenizer object used for tokenization
        Returns:
            dict: a dictionary containing the input_ids and labels for the examples
        """


        # datas = []

        # if 'instruction' in examples:
        #     datas = [ examples['instruction']]
        # elif 'query' in examples:
        #     datas = [ examples['query']]
        # else:
        #     raise ValueError('Cannot find key instruction or query!')

        # if 'input' in examples:
        #     datas.append(examples['input'])

        # if 'output' in examples:
        #     datas.append(examples['output'])
        # elif 'content' in examples:
        #     datas.append(examples['content'])
        # elif 'response' in examples:
        #     datas.append(examples['response'])
        # else:
        #     raise ValueError('Cannot find output key `output`, `content` or `response`!')

        datas = examples["messages"]
        input_ids = []
        labels = []
        for data in datas:
            # text = [
            #     {'role': 'user', 'content': ''.join(data[:-1])},
            #     {'role': 'assistant', 'content': data[-1]}
            # ]
            # text = data
            # source = self.tokenizer.apply_chat_template(text[:-1])[:-4]
            # full = self.tokenizer.apply_chat_template(text)[:-4]

            # for t1, t2 in zip(source, full):
            #     assert t1 == t2, "The user input_ids are not a prefix of the full input_ids! Please check the template."

            # if len(source) >= self.max_padding_length:
            #     continue

            # if len(full) > self.max_padding_length:
            #     full = full[:self.max_padding_length]
            # elif self.is_pad_token_eos_token:
            #     assert full[-1] == self.eos_token_id, f"Assume any untruncated sample ends with <eos>! But got: {self.tokenizer.detokenize(full)}"
            #     full[-1] = - 1 - full[-1]

            # if self.max_padding_length > len(full):
            #     full = full + [self.IGNORE_INDEX] * (self.max_padding_length - len(full))

            # NOTE: in get_batch_on_this_tp_rank_original, tokens use [:-1] and labels use [1:]
            # we add an extra token to use the old api
            # TODO: update get_batch_on_this_tp_rank_original and replace the following line with
            # label = [self.IGNORE_INDEX] * (len(source) - 1) + full[len(input_ids):] + [self.IGNORE_INDEX]

            full = self.tokenizer.apply_chat_template(data)
            full = full[:-4]

            if len(full) > self.max_padding_length:
                full = full[:self.max_padding_length]
            elif self.is_pad_token_eos_token:
                assert full[-1] == self.eos_token_id, f"Assume any untruncated sample ends with <eos>! But got: {self.tokenizer.detokenize(full)}"
                full[-1] = - 1 - full[-1]


            # ---- Chaged for tulu-v3 Start ----
            # 初始化标签（全部设为 -100）
            label = [self.IGNORE_INDEX] * len(full)

            # 遍历每个消息，标记 assistant 回复位置
            if len(data) < 2:
                continue

            for idx, msg in enumerate(data):
                if msg['role'] != 'assistant' or idx < 1:
                    continue
                # 生成到当前消息为止的 token_ids
                partial_ids = self.tokenizer.apply_chat_template(
                    data[:idx+1], 
                )

                prompt_ids = self.tokenizer.apply_chat_template(
                    data[:idx], 
                )
                current_end = len(partial_ids[:-4])

                # 计算当前消息的 token 范围（考虑整体截断）
                start = len(prompt_ids)
                end = min(current_end, len(full))
                
                # 标记有效标签区域
                if start < end:
                    label[start:end] = full[start:end]

            # 最终检查：至少有一个有效标签
            if all(y == self.IGNORE_INDEX for y in label):
                # print('No assistant content, skip sample')
                continue

            # ---- Chaged for tulu-v3 End ----
            
            if self.max_padding_length > len(full):
                full = full + [self.IGNORE_INDEX] * (self.max_padding_length - len(full))
                label = label + [self.IGNORE_INDEX] * (self.max_padding_length - len(label))
                
            # full = full + [self.IGNORE_INDEX]
            # label = [self.IGNORE_INDEX] + label
            

            input_ids.append(full)
            labels.append(label)
        
        return dict(input_ids=input_ids, labels=labels)

    def gpt_convert_example_to_feature(self, sample):
        """
        Convert a single sample containing input_id, label and loss_mask into a format suitable for GPT training.
        """
        input_ids, labels = sample
        train_sample = {
            'input_ids': input_ids,
            'labels': labels
        }

        return train_sample

class SFTPreTokenizedDataset(torch.utils.data.Dataset):
    """
    使用内存映射方式加载预处理好的 SFT 数据。
    数据存储在同一个文件夹中，文件命名格式为：
        {split}_input_ids.bin
        {split}_input_ids.idx
        {split}_labels.bin
        {split}_labels.idx

    使用时只需要提供数据目录和数据集划分（如 'train'、'valid' 或 'test'）。
    """
    def __init__(self, data_dir, split='train'):
        self.split = split
        # # 拼接生成 input_ids 的二进制和索引文件路径
        # input_bin_path = os.path.join(data_dir, f"{split}_input_ids.bin")
        # input_idx_path = os.path.join(data_dir, f"{split}_input_ids.idx")
        # # 拼接生成 labels 的二进制和索引文件路径
        # label_bin_path = os.path.join(data_dir, f"{split}_labels.bin")
        # label_idx_path = os.path.join(data_dir, f"{split}_labels.idx")


         # 拼接生成 input_ids 的二进制和索引文件路径
        input_bin_path = f"{data_dir}_input_ids.bin"
        input_idx_path = f"{data_dir}_input_ids.idx"
        # 拼接生成 labels 的二进制和索引文件路径
        label_bin_path = f"{data_dir}_labels.bin"
        label_idx_path = f"{data_dir}_labels.idx"

        # 加载 input_ids 文件与对应的索引
        self.input_ids = np.memmap(input_bin_path, dtype=np.int32, mode='r')
        self.input_offsets = np.fromfile(input_idx_path, dtype=np.int64)

        # 加载 labels 文件与对应的索引
        self.labels = np.memmap(label_bin_path, dtype=np.int32, mode='r')
        self.label_offsets = np.fromfile(label_idx_path, dtype=np.int64)

        # 简单校验：两个索引文件的长度应该一致
        assert len(self.input_offsets) == len(self.label_offsets), \
            "input 与 label 的索引文件长度不匹配！"

    def __len__(self):
        # 样本数 = 索引文件中的元素数 - 1
        return len(self.input_offsets) - 1

    def __getitem__(self, idx):
        # 根据索引文件获取当前样本在二进制文件中的起始和结束位置
        input_start = self.input_offsets[idx]
        input_end = self.input_offsets[idx + 1]
        sample_input_ids = self.input_ids[input_start:input_end].copy()

        label_start = self.label_offsets[idx]
        label_end = self.label_offsets[idx + 1]
        sample_labels = self.labels[label_start:label_end].copy()

        return {'input_ids': sample_input_ids, 'labels': sample_labels}

# 使用示例：
if __name__ == '__main__':
    # 假设预处理后的数据都放在目录 "pretokenized_data" 下，
    # 且已经生成了 "train_input_ids.bin", "train_input_ids.idx",
    # "train_labels.bin", "train_labels.idx" 等文件。
    data_dir = "/mnt/workspace/lusongshuo/ZJ-Megatron-LM/megatron/core/datasets/tuluv3_train_10k"
    split = "train"  # 或者 'valid' 或 'test'

    dataset = SFTPreTokenizedDataset(data_dir, split)
    print("数据集样本数:", len(dataset))

    # 取第 0 个样本看一下
    sample = dataset[0]
    print("第 0 个样本 input_ids 长度:", len(sample['input_ids']))
    print("第 0 个样本 labels 长度:", len(sample['labels']))


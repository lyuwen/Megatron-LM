import os
import sys
import numpy
import numpy as np
from typing import Union, Optional, Any
import importlib

from megatron.core.datasets.megatron_tokenizer import MegatronTokenizer


def import_tokenizer_class(path):
    if os.path.isdir(path):
        path = os.path.join(path, "tokenizer.py")
    if not os.path.exists(path):
        raise OSError(f"File {path} not exists.")
    module_name = "tokenizer"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.Tokenizer


def get_dir(path):
    if os.path.isdir(path):
        return path
    else:
        return os.path.dirname(path)


class TikTokenizer(MegatronTokenizer):
    """Abstract class for tokenizer

    Absent a config or class-specific tracking of which objects are uniquely identifying, we must
    include all key word arguments as unique identifiers

    Args:
        tokenizer_paths (Tuple[str]): All tokenizer source paths or prefixes

        kwargs (Dict[str, Any]): All tokenizer options
    """

    def __init__(self, *tokenizer_paths: str, **tokenizer_options: Any):

        super().__init__()

        tokenizer_path = tokenizer_paths[0]
        self.tokenizer_cls = import_tokenizer_class(tokenizer_options.get("tokenizer_class", get_dir(tokenizer_path)))
        self.tokenizer = self.tokenizer_cls(tokenizer_path)
        #  self._inv_vocab = dict((v, k) for k, v in self.vocab)

    def tokenize(self, text: str) -> numpy.ndarray:
        """Convert text to embedding ids

        Args:
            text (str): The text to convert

        Returns:
            numpy.ndarray: The converted embedding ids
        """
        return self.tokenizer.encode(text)

    def detokenize(self, ids: numpy.ndarray) -> str:
        """Convert embedding ids to text

        Args:
            ids (numpy.ndarray): The ids to convert

        Returns:
            str: The converted text

        Raises:
            NotImplementedError: Non-abstract, optional method
        """
        return self.tokenizer.decode(ids)

    @property
    def vocab(self):
        """Dictionary from vocab text token to id token
        """
        raise NotImplementedError
        return self.tokenizer.model._mergeable_ranks

    @property
    def inv_vocab(self):
        """Dictionary from vocab id token to text token
        """
        raise NotImplementedError
        return self._inv_vocab

    @property
    def vocab_size(self):
        """The vocabulary size
        """
        return self.tokenizer.n_words

    @property
    def pad(self):
        """The PAD token id

        Raises:
            NotImplementedError: Non-abstract, optional attribute
        """
        self.tokenizer.pad_id

    @property
    def bos(self):
        return self.tokenizer.bos_id

    @property
    def cls(self):
        return -1

    @property
    def sep(self):
        return -1

    @property
    def mask(self):
        return -1

    @property
    def eod(self):
        return self.tokenizer.eos_id

    @property
    def additional_special_tokens_ids(self):
        return list(self.tokenizer.special_tokens.values())

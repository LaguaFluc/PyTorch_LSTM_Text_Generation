
import torch
from torch.utils.data import TensorDataset, DataLoader

from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from typing import Tuple

import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def data_process(raw_text_iter) -> torch.Tensor:
    """Converts raw text into a flat Tensor."""
    train_iter = WikiText2(split='train')
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(
        map(tokenizer, train_iter), 
        specials=['<unk>', '<sos>', '<eos>', '<pad>']
        )
    vocab.set_default_index(vocab['<unk>'])
    data = [torch.tensor(
        vocab(tokenizer(item)
        ), dtype=torch.long) for item in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))


def get_dataset_vocab():
    train_iter = WikiText2(split='train')
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(
        map(tokenizer, train_iter), 
        specials=['<unk>', '<sos>', '<eos>', '<pad>']
        )
    vocab.set_default_index(vocab['<unk>'])
    return vocab, len(vocab)

def batchify(data: torch.Tensor, seq_len: int, bsz: int) -> torch.Tensor:
    """Divides the data into ``bsz`` separate sequences, removing extra elements
    that wouldn't cleanly fit.

    Arguments:
        data: Tensor, shape ``[N]``
        bsz: int, batch size

    Returns:
        Tensor of shape ``[N // bsz, bsz]``
    """
    n_batches = data.size(0) // bsz
    data = data[:seq_len * bsz]
    x = []
    y = []

    for i in range(0, len(data) - seq_len):
        i_end = i + seq_len
        batch_x = data[i:i + seq_len]
        x.append(batch_x)
        batch_y = data[i_end]
        y.append(batch_y)

    x = torch.from_numpy(np.asarray(x))
    y = torch.from_numpy(np.asarray(y))
    data = TensorDataset(x, y)
    data_loader = DataLoader(data, shuffle=True, batch_size=bsz)
    return data_loader

def get_dataloader(seq_len: int, batch_size: int, eval_batch_size: int):
    # ``train_iter`` was "consumed" by the process of building the vocab,
    # so we have to create it again
    train_iter, val_iter, test_iter = WikiText2()
    train_data = data_process(train_iter)
    val_data = data_process(val_iter)
    test_data = data_process(test_iter)

    train_data_loader = batchify(train_data, seq_len, batch_size)  # shape ``[seq_len, batch_size]``
    val_data_loader = batchify(val_data, seq_len, eval_batch_size)
    test_data_loader = batchify(test_data, seq_len, eval_batch_size)

    return train_data_loader, val_data_loader, test_data_loader

if __name__ == "__main__":
    # vocab, vocab_size = get_dataset_vocab()
    seq_len = 10
    batch_size = 32
    eval_batch_size = 16
    train_data_loader, eval_data_loader, test_data_loader = get_dataloader(seq_len, batch_size, eval_batch_size)
    for batch_idx, data in enumerate(train_data_loader):
        input_word_vector, output_word_vector = data
        print("data.shape: ", input_word_vector.shape)
        print("target.shape: ", output_word_vector.shape)
        break

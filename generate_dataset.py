
import torch

from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from typing import Tuple
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def data_process(raw_text_iter) -> torch.Tensor:
    """Converts raw text into a flat Tensor."""
    train_iter = WikiText2(split='train')
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(
        map(tokenizer, train_iter), 
        specials=['<unk>', '<sos>', '<eos>']
        )
    vocab.set_default_index(vocab['<unk>'])
    data = [torch.tensor(
        vocab(tokenizer(item)
        ), dtype=torch.long) for item in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

def batchify(data: torch.Tensor, bsz: int) -> torch.Tensor:
    """Divides the data into ``bsz`` separate sequences, removing extra elements
    that wouldn't cleanly fit.

    Arguments:
        data: Tensor, shape ``[N]``
        bsz: int, batch size

    Returns:
        Tensor of shape ``[N // bsz, bsz]``
    """
    seq_len = data.size(0) // bsz
    data = data[:seq_len * bsz]
    data = data.view(bsz, seq_len).t().contiguous()
    return data.to(device)

def get_init_dataset():
    train_iter = WikiText2(split='train')
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(
        map(tokenizer, train_iter), 
        specials=['<unk>', '<sos>', '<eos>']
        )
    vocab.set_default_index(vocab['<unk>'])
    return vocab, len(vocab)

def get_dataset(batch_size: int, eval_batch_size: int):
    # ``train_iter`` was "consumed" by the process of building the vocab,
    # so we have to create it again
    train_iter, val_iter, test_iter = WikiText2()
    train_data = data_process(train_iter)
    val_data = data_process(val_iter)
    test_data = data_process(test_iter)

    # batch_size = 20
    # eval_batch_size = 10
    train_data = batchify(train_data, batch_size)  # shape ``[seq_len, batch_size]``
    val_data = batchify(val_data, eval_batch_size)
    test_data = batchify(test_data, eval_batch_size)

    return train_data, val_data, test_data

bptt = 35
def get_batch(source: torch.Tensor, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        source: Tensor, shape ``[full_seq_len, batch_size]``
        i: int

    Returns:
        tuple (data, target), where data has shape ``[seq_len, batch_size]`` and
        target has shape ``[seq_len * batch_size]``
    """
    seq_len = min(bptt, len(source) - 1 - i)
    source = source
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len]
    return data, target


if __name__ == "__main__":
    vocab, vocab_size = get_init_dataset()
    batch_size = 20
    eval_batch_size = 10
    train_data, val_data, test_data = get_dataset(batch_size, eval_batch_size)
    
    data, target = get_batch(train_data, 0)
    print(data.shape, data)
    print(target.shape, target)

    print(vocab_size)
    print(train_data.shape)
    print(val_data.shape)
    print(test_data.shape)

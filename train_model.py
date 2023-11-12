
import torch
import torch.nn as nn
import torchtext

from tempfile import TemporaryDirectory

import time
import math

from build_model import CustomLSTM

import logging

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 创建一个logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)  # Log等级总开关

# 创建一个handler，用于写入日志文件
logfile = './pytorch_lstm_text_generation_log.log'
fh = logging.FileHandler(logfile, mode='a')
fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关

# 再创建一个handler，用于输出到控制台
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)  # 输出到console的log等级的开关

# 定义handler的输出格式
formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
fh.setFormatter(formatter)
ch.setFormatter(formatter)

# 将logger添加到handler里面
logger.addHandler(fh)
logger.addHandler(ch)

def train_epoch(
    epoch_idx: int,
    data_loader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    ) -> nn.Module:
    model.train()
    total_loss = 0.
    log_interval = 200
    start_time = time.time()

    for batch_idx, data in enumerate(data_loader):
        input_word_vector, output_word_vector = data
        output = model(input_word_vector)
        # print(output.shape, output_word_vector.shape)
        loss = criterion(output, output_word_vector)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        if (batch_idx + 1) % log_interval == 0 and batch_idx > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss)
            logger.debug(f'| epoch_idx {epoch_idx:3d} | {batch_idx:5d}/{num_batches:5d} batches | '
                  f'lr {lr:02.5f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            total_loss = 0
            start_time = time.time()
    return model

def train(
    train_data_loader,
    eval_data_loader,
    model: nn.Module,
    n_epochs: int,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    ) -> nn.Module:
    best_val_loss = float('inf')

    # with TemporaryDirectory() as tempdir:
    import os
    import pathlib

    cwd = pathlib.Path(os.getcwd())
    best_model_params_path = cwd / "best_model_params.pt"
    # best_model_params_path = os.path.join(cwd, "best_model_params.pt")

    for epoch in range(1, n_epochs + 1):
        epoch_start_time = time.time()
        train_epoch(
            epoch,
            train_data_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            )
        val_loss = evaluate(model, eval_data_loader, criterion)
        val_ppl = math.exp(val_loss)
        elapsed = time.time() - epoch_start_time
        logger.info('-' * 89)
        logger.info(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
            f'valid loss {val_loss:5.5f} | valid ppl {val_ppl:8.2f}')
        logger.info('-' * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_params_path)

        scheduler.step()
    model.load_state_dict(torch.load(best_model_params_path)) # load best model states
    return model


def evaluate(
    model: nn.Module, 
    eval_data_loader, 
    criterion: nn.Module,
    ) -> float:
    model.eval()  # turn on evaluation mode
    total_loss = 0
    with torch.no_grad():
        # print("eval_data.shape: ", eval_data.shape, eval_data.size())
        for batch_idx, data in enumerate(eval_data_loader):
            input_word_vector, output_word_vector = data
            output = model(input_word_vector)
            total_loss += criterion(output, output_word_vector).item()
    return total_loss / len(eval_data_loader)

def predict(
    model: nn.Module,
    text: str,
    next_words: int,
    seq_len: int,
    vocab: torchtext.vocab.Vocab
):
    model.eval()

    from torchtext.data.utils import get_tokenizer
    tokenizer = get_tokenizer('basic_english')

    words = text.split(' ')
    input_ids = [vocab(tokenizer(item)) for item in words]

    # padding if input is shorter than seq_len
    sample_len = len(input_ids)
    pad_len = (seq_len - sample_len) if (sample_len < seq_len) else 0
    if (sample_len <= seq_len):
        padded_input_ids = input_ids + [[vocab['<pad>']]] * pad_len
    else:
        padded_input_ids = input_ids[sample_len - seq_len:]

    output_words = []
    for i in range(0, next_words):
        # shape: (seq_len, 1)
        input_word_vector = torch.tensor(padded_input_ids).to(device) # input_ids[i:]
        # shape: (seq_len, )
        input_word_vector = input_word_vector.squeeze(1)
        # shape: (1, seq_len)
        input_word_vector = input_word_vector.unsqueeze(0)
        # y_pred.shape: (1, vocab_size)
        y_pred = model(input_word_vector)
        # topi.shape: (1, 1)
        _, topi = torch.topk(y_pred, 1)
        output_ids = topi.squeeze()
        output_word_idx = output_ids.item()
        output_words.append(vocab.lookup_token(output_word_idx))

        # add output_word_idx to the end of input_ids(before padding)
        if (pad_len == 0):
            padded_input_ids = padded_input_ids[1:] + [[output_word_idx]]
        else:
            pad_len -= 1
            padded_input_ids = padded_input_ids[1: seq_len - pad_len] + [[output_word_idx]] \
                + [[vocab['<pad>']]] * pad_len
    return ' '.join(output_words)



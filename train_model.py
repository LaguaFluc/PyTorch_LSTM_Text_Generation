
import torch
import torch.nn as nn
import torchtext

from tempfile import TemporaryDirectory

import time
import math

from generate_dataset import get_batch
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
    train_data: torch.Tensor,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    vocab_size: int
    ) -> None:
    model.train()  # turn on train mode
    total_loss = 0.
    log_interval = 200
    start_time = time.time()
    
    bptt = 35
    num_batches = len(train_data) // bptt
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i)
        output = model(data)
        # print(output.shape)
        # output_flat = output.view(-1, vocab_size)
        # print(output_flat.shape, targets.shape)
        # loss = criterion(output_flat, targets)
        loss = criterion(output.transpose(-1, -2), targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss)
            logger.debug(f'| epoch_idx {epoch_idx:3d} | {batch:5d}/{num_batches:5d} batches | '
                  f'lr {lr:02.5f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            total_loss = 0
            start_time = time.time()
    return model

def train(
    train_data: torch.Tensor,
    eval_data: torch.Tensor,
    model: nn.Module,
    n_epochs: int,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    vocab_size: int
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
            train_data,
            model,
            criterion,
            optimizer,
            scheduler,
            vocab_size
            )
        val_loss = evaluate(model, eval_data, criterion, vocab_size)
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
    eval_data: torch.Tensor,
    criterion: nn.Module,
    vocab_size: int
    ) -> float:
    model.eval()  # turn on evaluation mode
    total_loss = 0
    bptt = 35
    with torch.no_grad():
        # print("eval_data.shape: ", eval_data.shape, eval_data.size())
        end = int(eval_data.size(0))
        for i in range(0, end - 1, bptt):
            data, targets = get_batch(eval_data, i)
            seq_len = data.size(0)
            output = model(data)
            # output_flat = output.view(-1, vocab_size)
            # total_loss += seq_len * criterion(output_flat, targets).item()
            total_loss += seq_len * criterion(output.transpose(-1, -2), targets).item()
    return total_loss / (len(eval_data) - 1)

def predict(
    model: nn.Module,
    text: str,
    next_words: int,
    vocab_size: int,
    vocab: torchtext.vocab.Vocab
):
    model.eval()
    words = text.split(' ')
    # state_h, state_c = model.init_state(1)
    for i in range(0, next_words):
        x = torch.tensor([[vocab[w] for w in words[i:]]]).to(device)
        # y_pred, (state_h, state_c) = model(x, (state_h, state_c))
        y_pred = model(x)
        last_word_logits = y_pred[0][-1]

        _, topi = torch.topk(last_word_logits, 1)
        words.append(vocab.lookup_token(topi.item()))
    return ' '.join(words)



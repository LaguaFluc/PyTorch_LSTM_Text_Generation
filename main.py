

# 1. generate data
# 2. build model
# 3. train model
# 4. evalate model

import torch
import torch.nn as nn

from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

import math

from generate_dataset import data_process, batchify
# from generate_dataset import get_dataset
from generate_dataset import get_dataloader
from generate_dataset import get_dataset_vocab
from build_model import CustomLSTM
from train_model import train_epoch, train, evaluate, predict

from train_model import logger

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ======================================
# 1. generate data
# ======================================
import yaml
with open("./config.yml", "r", encoding='utf-8') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    batch_size = config["batch_size"]
    input_size = config["input_size"]
    hidden_size = config["hidden_size"]
    seq_len = config["seq_len"]

    learning_rate = config["learning_rate"]
    n_epochs = config["n_epochs"]
    eval_batch_size = config["eval_batch_size"]

vocab, vocab_size = get_dataset_vocab()
train_data_loader, eval_data_loader, test_data_loader = get_dataloader(seq_len, batch_size, eval_batch_size)

# ======================================
# 2. build model
# ======================================
model = CustomLSTM(vocab_size, input_size, hidden_size).to(device)

criterion = nn.CrossEntropyLoss()
# lr = 5.0  # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

# ======================================
# 3. train model
# ======================================
# train(train_data, eval_data, model, n_epochs, criterion, optimizer, scheduler, vocab_size)
train(train_data_loader, eval_data_loader, model, n_epochs, criterion, optimizer, scheduler)
model.load_state_dict(torch.load("./best_model_params.pt"))

# ======================================
# 4. evaluate model
# ======================================
test_loss = evaluate(model, test_data_loader, criterion)
test_ppl = math.exp(test_loss)
logger.info('=' * 89)
logger.info(f'| End of training | test loss {test_loss:5.2f} | '
      f'test ppl {test_ppl:8.2f}')
logger.info('=' * 89)

# ======================================
# 5. get results
# ======================================
input_words = "Homarus gammarus is a large"
predicted_words = predict(model, input_words, 2, seq_len, vocab)
logger.info(f"predicted words: %s" % predicted_words)

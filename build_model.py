
import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class CustomLSTM(nn.Module):
    def __init__(
        self, 
        vocab_size, input_size, hidden_size, 
        num_layers=1, bidirectional=False
        ) -> None:
        super(CustomLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.input_size = input_size
        self.hidden_size = hidden_size      # word embedding dim

        self.num_layers = num_layers
        self.D = 2 if bidirectional else 1
        self.initrange = torch.sqrt(
            torch.tensor([1 / self.hidden_size], dtype=torch.float32)
            ).item()
        self.embedding = nn.Embedding(vocab_size, input_size)
        self.lstm = nn.LSTM(
            self.input_size, self.hidden_size, 
            num_layers=self.num_layers,
            batch_first=True
            )

        self.linear = nn.Linear(self.hidden_size, self.vocab_size)
        self.init_weights()

    def forward(self, inputs):
        self.embedded = self.embedding(inputs)
        h0 = self.init_hidden(inputs.size(0))
        c0 = self.init_cell(inputs.size(0))
        lstm_out, (hidden, cell) = self.lstm(self.embedded, (h0, c0))
        output = self.linear(lstm_out)
        return output

    def init_hidden(self, batch_size):
        initrange = self.initrange
        hidden = torch.empty([self.D * self.num_layers, batch_size, self.hidden_size])
        torch.nn.init.uniform_(hidden, -initrange, initrange)
        return hidden.to(device)

    def init_cell(self, batch_size):
        initrange = self.initrange
        cell = torch.empty([self.D * self.num_layers, batch_size, self.hidden_size])
        torch.nn.init.uniform_(cell, -initrange, initrange)
        return cell.to(device)
    
    def init_weights(self) -> None:
        initrange = self.initrange
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.uniform_(-initrange, initrange)
        self.linear.weight.data.uniform_(-initrange, initrange)

if __name__ == "__main__":
    import yaml
    with open("./config.yml", "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)
        batch_size = config["batch_size"]
        input_size = config["input_size"]
        hidden_size = config["hidden_size"]

        seq_len = config["seq_len"]
    vocab_size = 1000
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    model = CustomLSTM(vocab_size, input_size, hidden_size)
    output = model(x)
    # embedding = nn.Embedding(vocab_size, input_size)
    # output = embedding(x)
    print(output.size())
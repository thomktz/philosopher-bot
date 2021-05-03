# %%
from sequence_maker import X,Y, chars, chars_to_int, max_len, window, n_vocab
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tqdm

X = torch.Tensor(X)
Y = torch.Tensor(Y)

device = torch.device("cuda")

class TextGenerator(nn.ModuleList):
    def __init__(self, batch_size = 64, hidden_dim = 128, vocab_size = n_vocab):
        super(TextGenerator, self).__init__()
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.input_size = vocab_size
        self.num_classes = vocab_size
        self.sequence_len = window

        self.dropout = nn.Dropout(0.25)

        self.embedding = nn.Embedding(self.input_size, self.hidden_dim, padding_idx=0)

        #Bi-LSTM
        self.lstm_cell_forward = nn.LSTMCell(self.hidden_dim, self.hidden_dim)
        self.lstm_cell_backward = nn.LSTMCell(self.hidden_dim, self.hidden_dim)

        self.lstm_cell = nn.LSTMCell(self.hidden_dim*2, self.hidden_dim*2)

        self.linear = nn.Linear(self.hidden_dim*2, self.num_classes)

    def forward(self, x):
        
        hs_forward = torch.zeros(x.size(0), self.hidden_dim).to(device)
        cs_forward = torch.zeros(x.size(0), self.hidden_dim).to(device)
        hs_backward = torch.zeros(x.size(0), self.hidden_dim).to(device)
        cs_backward = torch.zeros(x.size(0), self.hidden_dim).to(device)
        
        hs_lstm = torch.zeros(x.size(0), self.hidden_dim * 2).to(device)
        cs_lstm = torch.zeros(x.size(0), self.hidden_dim * 2).to(device)
        
        torch.nn.init.kaiming_normal_(hs_forward)
        torch.nn.init.kaiming_normal_(cs_forward)
        torch.nn.init.kaiming_normal_(hs_backward)
        torch.nn.init.kaiming_normal_(cs_backward)
        torch.nn.init.kaiming_normal_(hs_lstm)
        torch.nn.init.kaiming_normal_(cs_lstm)
        
        out = self.embedding(x)
        
        out = out.view(self.sequence_len, x.size(0), -1)
        
        forward = []
        backward = []
        # Forward
        for i in range(self.sequence_len):
            hs_forward, cs_forward = self.lstm_cell_forward(out[i], (hs_forward, cs_forward))
            hs_forward = self.dropout(hs_forward)
            cs_forward = self.dropout(cs_forward)
            forward.append(hs_forward)
            
        # Backward
        for i in reversed(range(self.sequence_len)):
            hs_backward, cs_backward = self.lstm_cell_backward(out[i], (hs_backward, cs_backward))
            hs_backward = self.dropout(hs_backward)
            cs_backward = self.dropout(cs_backward)
            backward.append(hs_backward)
            
        # LSTM
        for fwd, bwd in zip(forward, backward):
            input_tensor = torch.cat((fwd, bwd), 1)
            hs_lstm, cs_lstm = self.lstm_cell(input_tensor, (hs_lstm, cs_lstm))
            
        out = self.linear(hs_lstm)
        return out


    def train_model(self, learning_rate, num_epochs):
        optimizer = optim.RMSprop(self.parameters(), lr = learning_rate)
        num_batches = int(len(X) / self.batch_size)
        
        self.train()
        self = self.to(device)
        
        for epoch in range(num_epochs):
            running_loss=0
            for i in tqdm.tqdm(range(num_batches)):
                try:
                    x_batch = X[i * self.batch_size : (i + 1) * self.batch_size]
                    y_batch = Y[i * self.batch_size : (i + 1) * self.batch_size]
                except:
                    x_batch = X[i * self.batch_size :]
                    y_batch = Y[i * self.batch_size :]

                x = x_batch.type(torch.LongTensor).to(device)
                y = y_batch.type(torch.LongTensor).to(device)
                y_pred = self(x)
                loss = F.cross_entropy(y_pred, y.squeeze())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print("Epoch: %d ,  loss: %.5f " % (epoch, running_loss/num_batches))
# %%

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, n_classes):
        super().__init__()
        # Initialize layers
        self.linear = nn.Linear(input_dim, n_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.linear(x)
        result = self.relu(output)
        return result

class EmotionTransformerPrototypeMLP(nn.Module):
    def __init__(self, input_dim, num_class, num_layers=2, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.encoder = MLP(input_dim, hidden_dim)
        self.mid_encoder = MLP(hidden_dim, hidden_dim)
        self.decoder = MLP(hidden_dim, num_class)
        self.input_dim = input_dim
        self.num_class = num_class
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout

    def forward(self, inputs):
        # Might include dropout to regularize
        log_probs = None
        embed = self.encoder(inputs)
        for i in range(self.num_layers):
            embed = self.mid_encoder(embed)
        m = nn.Dropout(p=self.dropout)
        drop = m(embed)
        output = self.decoder(drop)
        log_probs = F.log_softmax(output, dim=-1)
        return log_probs

    def get_loss(self, probs, targets):
        loss  = F.nll_loss(probs, targets)
        return loss    

class EmotionTransformerPrototypeImproved(nn.Module):
    def __init__(self, input_dim, num_class, num_layers=2, hidden_dim=128):
        super().__init__()
        self.encoder = MLP(input_dim, hidden_dim)
        self.mid_encoder = MLP(hidden_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, num_class)
        self.input_dim = input_dim
        self.num_class = num_class
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

    def forward(self, inputs):
        log_probs = None
        embed = self.encoder(inputs)
        for i in range(self.num_layers):
            embed = self.mid_encoder(embed)
        output = self.decoder(embed)
        log_probs = F.log_softmax(output, dim=-1)
        return log_probs

    def get_loss(self, probs, targets):
        loss  = F.nll_loss(probs, targets)
        return loss
    

class EmotionTransformerPrototype(nn.Module):
    def __init__(self, input_dim, num_class, num_layers=2, hidden_dim=128):
        super().__init__()
        self.encoder = MLP(input_dim, hidden_dim)
#         self.mid_encoder = MLP(hidden_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, num_class)
        self.input_dim = input_dim
        self.num_class = num_class
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

    def forward(self, inputs):
        log_probs = None
        embed = self.encoder(inputs)
        output = self.decoder(embed)
        log_probs = F.log_softmax(output, dim=-1)
        return log_probs

    def get_loss(self, probs, targets):
        loss  = F.nll_loss(probs, targets)
        return loss

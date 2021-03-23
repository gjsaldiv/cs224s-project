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

class EmotionLSTMTransformer(nn.Module):
    """
  Encoder-Decoder model to classify emotions for utterances.

  Args:
    input_dim: integer
                number of input features
    num_class: integer
                number of class labels
    num_layers: integer (default: 2)
                number of layers in encoder LSTM
    hidden_dim: integer (default: 128)
                number of hidden dimensions for encoder LSTM
    bidirectional: boolean (default: True)
                    is the encoder LSTM bidirectional?
  """
    def __init__(
          self, input_dim, num_class, num_layers=2, hidden_dim=128,
          bidirectional=True, dropout=0.2):
        super().__init__()
        # Note: `batch_first=True` argument implies the inputs to the LSTM should
        # be of shape (batch_size x T x D) instead of (T x batch_size x D).
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, 
                                bidirectional=bidirectional, batch_first=True)
        self.decoder = nn.Linear(hidden_dim * 2, num_class)
        self.input_dim = input_dim
        self.num_class = num_class
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.embedding_dim = hidden_dim * num_layers * 2 * \
                              (2 if bidirectional else 1)

    def combine_h_and_c(self, h, c):
        """Combine the signals from RNN hidden and cell states."""
        batch_size = h.size(1)
        h = h.permute(1, 0, 2).contiguous()
        c = c.permute(1, 0, 2).contiguous()
        h = h.view(batch_size, -1)
        c = c.view(batch_size, -1)
        return torch.cat([h, c], dim=1)  # just concatenate

    def forward(self, inputs):
        batch_size, max_length = inputs.size()
        # `torch.nn.utils.rnn.pack_padded_sequence` collapses padded sequences
        # to a contiguous chunk
        #input_lengths = [max_length for x in range(batch_size)]
        #input_lengths = torch.LongTensor(input_lengths).cpu()
        # Needs to be 3D, to include seq_len (number of time steps), so add that as dim=1
        inputs = torch.unsqueeze(inputs, 0)
        #inputs = torch.nn.utils.rnn.pack_padded_sequence(
        #    inputs, input_lengths, batch_first=True, enforce_sorted=False)
        log_probs = None
        h, c = None, None
        # - Refer to https://pytorch.org/docs/stable/nn.html
        # - Use `self.encoder` to get the encodings output which is of shape
        #   (batch_size, max_length, num_directions*hidden_dim) and the
        #   hidden states and cell states which are both of shape
        #   (batch_size, num_layers*num_directions, hidden_dim)
        # - Pad outputs with `0.` using `torch.nn.utils.rnn.pad_packed_sequence`
        #   (turn on batch_first and set total_length as max_length).
        # - Apply 50% dropout.
        # - Use `self.decoder` to take the embeddings sequence and return
        #   probabilities for each character.
        # - Make sure to then convert to log probabilities.

        # Get encodings
        encodings, (h, c) = self.encoder(inputs)

        # Pad outputs with '0'
        #padded, lengths = torch.nn.utils.rnn.pad_packed_sequence(encodings, batch_first=True, total_length=self.input_dim)

        # Apply dropout
        dropout = nn.Dropout(p=self.dropout)
        dropped_padded = dropout(encodings)

        # Decode to get probabilies dimensions (batch_size, time steps, num chars <vocab>)
        probabilities = self.decoder(dropped_padded)

        # Log probabilities
        # dim=2 because softmax over the vocabulary dimension. dim=1 is over time steps
        log_probs = F.log_softmax(probabilities, dim=-1)
        log_probs = torch.squeeze(log_probs)

        # The extracted embedding is not used for the ASR task but will be
        # needed for other auxiliary tasks.
        #embedding = self.combine_h_and_c(h, c)
        return log_probs

    def get_loss(self, log_probs, targets):
            # cross-entropy for softmax/negative log likelihood loss for log softmax
        return F.nll_loss(probs, targets)

    def decode(self, log_probs):
        # Use greedy decoding.
        decoded, pred = torch.argmax(log_probs, dim=-1)
        batch_size = decoded.size(0)
        
        return pred

    
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

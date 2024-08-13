import torch
import torch.nn as nn
import torch.nn.init as init


class GRU(nn.Module):
    """
    Gated Recurrent Unit.

    Arguments:
    - input_size: int
        The input size.
    - hidden_size: int
        The hidden state size.
    - num_layers: int
        The number of layers.
    - output_size: int
        The output size.
    """
    num_states = 1

    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.rnn = nn.GRU(input_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, h=None):
        """
        Computes the forward pass on the sequence using the hidden state if
        provided.

        Arguments:
        - x: tensor
            Input sequence
        - h: tuple of tensor
            Initial hidden state

        Returns:
        - x: tensor
            Output sequence
        - h: tuple of tensor
            Final hidden state
        """
        if h is not None:
            h, = h
        x, h = self.rnn(x, h)
        x = self.linear(x)
        return x, (h,)


class LSTM(nn.Module):
    """
    Long Short Term Memory.

    Arguments:
    - input_size: int
        The input size.
    - hidden_size: int
        The hidden state size.
    - num_layers: int
        The number of layers.
    - output_size: int
        The output size.
    """
    num_states = 2

    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.rnn = nn.LSTM(input_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, h=None):
        """
        Computes the forward pass on the sequence using the hidden state if
        provided.

        Arguments:
        - x: tensor
            Input sequence
        - h: tuple of tensors
            Initial hidden state

        Returns:
        - x: tensor
            Output sequence
        - h: tuple of tensors
            Final hidden state
        """
        x, h = self.rnn(x, h)
        x = self.linear(x)
        return x, h


class BRCLayer(nn.Module):
    """
    Bistable Recurrent Cell (see arXiv:2006.05252).

    Arguments:
    - input_size: int
        The input size.
    - hidden_size: int
        The hidden state size.
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.input_size = input_size
        self.hidden_size = hidden_size

        # Forget gate
        U_c = init.xavier_uniform_(torch.empty(hidden_size, input_size))
        self.U_c = nn.Parameter(U_c)
        self.w_c = nn.Parameter(init.constant_(torch.empty(hidden_size), 1.0))
        self.b_c = nn.Parameter(init.zeros_(torch.empty(hidden_size)))

        # Reset gate
        U_a = init.xavier_uniform_(torch.empty(hidden_size, input_size))
        self.U_a = nn.Parameter(U_a)
        self.w_a = nn.Parameter(init.constant_(torch.empty(hidden_size), 1.0))
        self.b_a = nn.Parameter(init.zeros_(torch.empty(hidden_size)))

        # Hidden state
        U_h = init.xavier_uniform_(torch.empty(hidden_size, input_size))
        self.U_h = nn.Parameter(U_h)
        self.b_h = nn.Parameter(init.zeros_(torch.empty(hidden_size)))

    def forward(self, x_seq, h0):
        """
        Computes the forward pass on the sequence using the hidden state if
        provided.

        Arguments:
        - x: tensor
            Input sequence
        - h: tuple of tensor
            Initial hidden state

        Returns:
        - x: tensor
            Output sequence
        - h: tuple of tensor
            Final hidden state
        """

        if h0 is None:
            h0 = torch.zeros(x_seq.size(1), self.hidden_size,
                             device=x_seq.device)
        else:
            h0, = h0

        assert h0.size(0) == x_seq.size(1)
        assert h0.size(1) == self.hidden_size
        assert x_seq.size(2) == self.input_size

        seq_len = x_seq.size(0)
        batch_size = x_seq.size(1)

        y_seq = torch.empty(seq_len, batch_size, self.hidden_size,
                            device=x_seq.device)

        h = h0
        for t in range(seq_len):
            x = x_seq[t, :, :]
            c = torch.sigmoid(
                torch.mm(x, self.U_c.T) + self.w_c * h + self.b_c)
            a = 1. + torch.tanh(
                torch.mm(x, self.U_a.T) + self.w_a * h + self.b_a)
            h = c * h + (1. - c) * torch.tanh(
                torch.mm(x, self.U_h.T) + a * h + self.b_h)
            y_seq[t, ...] = h

        return y_seq, (h,)


class nBRCLayer(nn.Module):
    """
    Recurrently Neuromodulated Bistable Recurrent Cell (see arXiv:2006.05252).

    Arguments:
    - input_size: int
        The input size.
    - hidden_size: int
        The hidden state size.
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.input_size = input_size
        self.hidden_size = hidden_size

        # Forget gate
        U_c = init.xavier_uniform_(torch.empty(hidden_size, input_size))
        self.U_c = nn.Parameter(U_c)
        W_c = init.xavier_uniform_(torch.empty(hidden_size, hidden_size))
        self.W_c = nn.Parameter(W_c)
        self.b_c = nn.Parameter(init.zeros_(torch.empty(hidden_size)))

        # Reset gate
        U_a = init.xavier_uniform_(torch.empty(hidden_size, input_size))
        self.U_a = nn.Parameter(U_a)
        W_a = init.xavier_uniform_(torch.empty(hidden_size, hidden_size))
        self.W_a = nn.Parameter(W_a)
        self.b_a = nn.Parameter(init.zeros_(torch.empty(hidden_size)))

        # Hidden state
        U_h = init.xavier_uniform_(torch.empty(hidden_size, input_size))
        self.U_h = nn.Parameter(U_h)
        self.b_h = nn.Parameter(init.zeros_(torch.empty(hidden_size)))

    def forward(self, x_seq, h0):
        """
        Computes the forward pass on the sequence using the hidden state if
        provided.

        Arguments:
        - x: tensor
            Input sequence
        - h: tuple of tensor
            Initial hidden state

        Returns:
        - x: tensor
            Output sequence
        - h: tuple of tensor
            Final hidden state
        """

        if h0 is None:
            h0 = torch.zeros(x_seq.size(1), self.hidden_size,
                             device=x_seq.device)
        else:
            h0, = h0

        assert h0.size(0) == x_seq.size(1)
        assert h0.size(1) == self.hidden_size
        assert x_seq.size(2) == self.input_size

        seq_len = x_seq.size(0)
        batch_size = x_seq.size(1)

        y_seq = torch.empty(seq_len, batch_size, self.hidden_size,
                            device=x_seq.device)

        h = h0
        for t in range(seq_len):
            x = x_seq[t, :, :]
            c = torch.sigmoid(
                torch.mm(x, self.U_c.T) + torch.mm(h, self.W_c.T) + self.b_c)
            a = 1. + torch.tanh(
                torch.mm(x, self.U_a.T) + torch.mm(h, self.W_a.T) + self.b_a)
            h = c * h + (1. - c) * torch.tanh(
                torch.mm(x, self.U_h.T) + a * h + self.b_h)
            y_seq[t, ...] = h

        return y_seq, (h,)


class MGULayer(nn.Module):
    """
    Minimal Gated Unit (see arXiv:1603.09420).

    Arguments:
    - input_size: int
        The input size.
    - hidden_size: int
        The hidden state size.
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.input_size = input_size
        self.hidden_size = hidden_size

        # Forget gate
        Wx_f = init.xavier_uniform_(torch.empty(hidden_size, input_size))
        Wh_f = init.xavier_uniform_(torch.empty(hidden_size, hidden_size))
        self.Wx_f = nn.Parameter(Wx_f)
        self.Wh_f = nn.Parameter(Wh_f)
        self.b_f = nn.Parameter(init.zeros_(torch.empty(hidden_size)))

        # Hidden state
        Wx_h = init.xavier_uniform_(torch.empty(hidden_size, input_size))
        Wh_h = init.xavier_uniform_(torch.empty(hidden_size, hidden_size))
        self.Wx_h = nn.Parameter(Wx_h)
        self.Wh_h = nn.Parameter(Wh_h)
        self.b_h = nn.Parameter(init.zeros_(torch.empty(hidden_size)))

    def forward(self, x_seq, h0):
        """
        Computes the forward pass on the sequence using the hidden state if
        provided.

        Arguments:
        - x: tensor
            Input sequence
        - h: tuple of tensor
            Initial hidden state

        Returns:
        - x: tensor
            Output sequence
        - h: tuple of tensor
            Final hidden state
        """

        if h0 is None:
            h0 = torch.zeros(x_seq.size(1), self.hidden_size,
                             device=x_seq.device)
        else:
            h0, = h0

        assert h0.size(0) == x_seq.size(1)
        assert h0.size(1) == self.hidden_size
        assert x_seq.size(2) == self.input_size

        seq_len = x_seq.size(0)
        batch_size = x_seq.size(1)

        y_seq = torch.empty(seq_len, batch_size, self.hidden_size,
                            device=x_seq.device)

        h = h0
        for t in range(seq_len):
            x = x_seq[t, :, :]

            f = torch.sigmoid(
                torch.mm(x, self.Wx_f.T) + torch.mm(h, self.Wh_f.T) + self.b_f)
            h_t = torch.tanh(
                torch.mm(f * h, self.Wh_h.T) + torch.mm(x, self.Wx_h.T) +
                self.b_h)
            h = (1.0 - f) * h + f * h_t

            y_seq[t, ...] = h

        return y_seq, (h,)


class CustomRNN(nn.Module):
    """
    Custom RNN implementation from layers (MGU, BRC, nBRC).

    Arguments:
    - input_size: int
        The input size.
    - hidden_size: int
        The hidden state size.
    - num_layers: int
        The number of layers.
    - output_size: int
        The output size.
    """
    num_states = 1

    def __init__(self, Cell, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        cells = [Cell(input_size, hidden_size)]
        for _ in range(num_layers - 1):
            cells.append(Cell(hidden_size, hidden_size))
        self.cells = nn.ModuleList(cells)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, h0=None):
        """
        Computes the forward pass on the sequence using the hidden state if
        provided.

        Arguments:
        - x: tensor
            Input sequence
        - h: tuple of tensor
            Initial hidden state

        Returns:
        - x: tensor
            Output sequence
        - h: tuple of tensor
            Final hidden state
        """
        hn = torch.empty(self.num_states, self.num_layers, x.size(1),
                         self.hidden_size)

        for i, cell in enumerate(self.cells):
            x, hni = cell(x, h0[:, i, :, :] if h0 is not None else None)
            for state_id, state in enumerate(hni):
                hn[state_id, i, ...] = state

        x = self.linear(x)

        return x, hn


class BRC(CustomRNN):
    """
    Bistable Recurrent Cell (see arXiv:2006.05252).

    Arguments:
    - input_size: int
        The input size.
    - hidden_size: int
        The hidden state size.
    - num_layers: int
        The number of layers.
    - output_size: int
        The output size.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(BRCLayer, *args, **kwargs)


class nBRC(CustomRNN):
    """
    Recurrently Neuromodulated Bistable Recurrent Cell (see arXiv:2006.05252).

    Arguments:
    - input_size: int
        The input size.
    - hidden_size: int
        The hidden state size.
    - num_layers: int
        The number of layers.
    - output_size: int
        The output size.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(nBRCLayer, *args, **kwargs)


class MGU(CustomRNN):
    """
    Minimal Gated Unit (see arXiv:1603.09420).

    Arguments:
    - input_size: int
        The input size.
    - hidden_size: int
        The hidden state size.
    - num_layers: int
        The number of layers.
    - output_size: int
        The output size.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(MGULayer, *args, **kwargs)


RNNS = {'lstm': LSTM, 'gru': GRU, 'brc': BRC, 'nbrc': nBRC, 'mgu': MGU}

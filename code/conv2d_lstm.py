import torch
import torch.nn as nn
from torch.autograd import Variable

device = "cuda" if torch.cuda.is_available else "cpu"

class ConvLSTMCell(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_features = 4

        self.padding = int((kernel_size - 1) / 2)

        self.Weight_x_i = nn.Conv2d(
            self.input_channels, self.hidden_channels,
            self.kernel_size, 1, self.padding, bias=True)
        
        self.Weight_h_i = nn.Conv2d(
            self.hidden_channels, self.hidden_channels,
            self.kernel_size, 1, self.padding, bias=False)
        
        self.Weight_x_f = nn.Conv2d(
            self.input_channels, self.hidden_channels,
            self.kernel_size, 1, self.padding, bias=True)
        
        self.Weight_h_f = nn.Conv2d(
            self.hidden_channels, self.hidden_channels,
            self.kernel_size, 1, self.padding, bias=False)
        
        self.Weight_x_c = nn.Conv2d(
            self.input_channels, self.hidden_channels,
            self.kernel_size, 1, self.padding, bias=True)
        
        self.Weight_h_c = nn.Conv2d(
            self.hidden_channels, self.hidden_channels,
            self.kernel_size, 1, self.padding, bias=False)
        
        self.Weight_x_o = nn.Conv2d(
            self.input_channels, self.hidden_channels,
            self.kernel_size, 1, self.padding, bias=True)
        
        self.Weight_h_o = nn.Conv2d(
            self.hidden_channels, self.hidden_channels,
            self.kernel_size, 1, self.padding, bias=False)

        self.Weight_c_i = None
        self.Weight_c_f = None
        self.Weight_c_o = None

    def forward(self, x, h, c):
        c_i = torch.sigmoid(self.Weight_x_i(x) + self.Weight_h_i(h) + c * self.Weight_c_i)
        c_f = torch.sigmoid(self.Weight_x_f(x) + self.Weight_h_f(h) + c * self.Weight_c_f)
        c_c = c_f * c + c_i * torch.tanh(self. Weight_x_c(x) + self.Weight_h_c(h))
        c_o = torch.sigmoid(self.Weight_x_o(x) + self.Weight_h_o(h) + c_c * self. Weight_c_o)
        c_h = c_o * torch.tanh(c_c)
        return c_h, c_c

    def init_hidden(self, batch_size, hidden, shape):
        if self.Weight_c_i is None:
            self.Weight_c_i = Variable(torch.zeros(1, hidden, shape[0], shape[1])).to("cpu")
            self.Weight_c_f = Variable(torch.zeros(1, hidden, shape[0], shape[1])).to("cpu")
            self.Weight_c_o = Variable(torch.zeros(1, hidden, shape[0], shape[1])).to("cpu")
        else:
            assert shape[0] == self.Weight_c_i.size()[2], 'Input Height Mismatched!'
            assert shape[1] == self.Weight_c_i.size()[3], 'Input Width Mismatched!'
        return (Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).to("cpu"),
                Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).to("cpu"))


class ConvLSTM(nn.Module):
    """_summary_

    Args:
        nn (_input_): _input channel
        _

    Returns:
        _type_: _description_
    """   
    def __init__(self, input_channels, hidden_channels, kernel_size, step=1, effective_step=[1]):
        """_summary_

        Args:
            input_channels (_array_): _sequence matrices as tensor_
            hidden_channels (_array_): _hidden units_
            kernel_size (_matrix_): _kernal size_
            step (int, optional): _description_. Defaults to 1.
            effective_step (list, optional): _description_. Defaults to [1].
        """
        super(ConvLSTM, self).__init__()
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.step = step
        self.effective_step = effective_step
        self._all_layers = []
        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size)
            setattr(self, name, cell)
            self._all_layers.append(cell)

    def forward(self, input):
        internal_state = []
        outputs = []
        for step in range(self.step):
            x = input
            for i in range(self.num_layers):               
                name = 'cell{}'.format(i)
                if step == 0:
                    bsize, _, height, width = x.size()
                    (h, c) = getattr(self, name).init_hidden(batch_size=bsize, hidden=self.hidden_channels[i],
                                                             shape=(height, width))
                    internal_state.append((h, c))

                
                (h, c) = internal_state[i]
                x, new_c = getattr(self, name)(x, h, c)
                internal_state[i] = (x, new_c)
            
            if step in self.effective_step:
                outputs.append(x)

        return outputs, (x, new_c)


if __name__ == '__main__':
    # gradient check
    convlstm = ConvLSTM(input_channels=512,
                        hidden_channels=[128, 64, 64, 32, 32],
                        kernel_size=3, step=5,
                        effective_step=[4]).to("cpu")
    
    loss_fn = torch.nn.MSELoss()

    input = Variable(torch.randn(5, 512, 64, 32)).to("cpu")
    target = Variable(torch.randn(1, 32, 64, 32)).double().to("cpu")

    output = convlstm(input)
    output = output[0][0].double()
    print(output.shape)    # check for output chape
    
    

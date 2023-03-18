import torch


class MLP(torch.nn.Module):
    """
    Since this model is used for policy-based problems, the output activation function is soft-max
    to estimate the probabilities corresponding to each action.
    """

    def __init__(self, n_input, n_output, n_hidden=None, random_state=110):

        assert (isinstance(n_hidden, list)) or (n_hidden is None), 'n_hidden should be either None or a list.'

        super(MLP, self).__init__()
        self.n_hidden = n_hidden
        self.n_input = n_input
        self.n_output = n_output
        self.random_state = random_state
        self.device = torch.device("cpu")
        torch.manual_seed(0)

        if n_hidden is None:
            nodes = [n_input, n_output]
        else:
            nodes = [n_input] + n_hidden + [n_output]
        self.nodes = nodes
        self.n_weights = sum([nodes[i] * nodes[i + 1] for i in range(len(nodes) - 1)])
        self.n_bias = sum([nodes[i + 1] for i in range(len(nodes) - 1)])
        self.tot_params = self.n_weights + self.n_bias

        if n_hidden is None:
            self.output_layer = torch.nn.Linear(n_input, n_output)
            self.weights_size = [self.output_layer.weight.shape]
            self.bias_size = [self.output_layer.bias.shape]

        else:
            self.hidden_layer = torch.nn.ModuleList()
            for i, n in enumerate(n_hidden):
                if i == 0:
                    layer = torch.nn.Linear(n_input, n)
                else:
                    layer = torch.nn.Linear(n_hidden[i - 1], n)
                self.hidden_layer.append(layer)
            self.output_layer = torch.nn.Linear(n_hidden[-1], n_output)

            self.weights_size = [x.weight.shape for x in self.hidden_layer] + [self.output_layer.weight.shape]
            self.bias_size = [x.bias.shape for x in self.hidden_layer] + [self.output_layer.bias.shape]

    def forward(self, x):
        if self.n_hidden is not None:
            for hidden_layer in self.hidden_layer:
                x = torch.nn.functional.relu(hidden_layer(x))
        z = torch.nn.functional.softmax(self.output_layer(x), dim=-1)
        return z

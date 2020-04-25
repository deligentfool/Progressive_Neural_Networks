import torch
import torch.nn as nn
import torch.nn.functional as F


class Pnn_Linear(nn.Module):
    def __init__(self, col, depth, input_dim, output_dim):
        super(Pnn_Linear, self).__init__()
        self.col = col
        self.depth = depth
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.w = nn.Linear(self.input_dim, self.output_dim)
        # * lateral connections matrix
        self.u = nn.ModuleList([])
        # * adapter: projection matrix
        self.v = nn.ModuleList([])
        # * a learned scala, initialized by a random small value
        self.alpha = []
        if self.depth > 0:
            self.u.extend([nn.Linear(self.input_dim, self.output_dim) for _ in range(self.col)])
            self.v.extend([nn.Linear(self.input_dim, self.input_dim) for _ in range(self.col)])
            self.alpha.extend([torch.FloatTensor([0.1]) for _ in range(self.col)])

    def forward(self, inputs):
        if type(inputs) is not list:
            inputs = [inputs]
        cur_h = self.w(inputs[-1])
        # * lateral connections output
        pre_h_list = [u(F.relu(v(alpha * input))) for u, v, alpha, input in zip(self.u, self.v, self.alpha, inputs)]
        return F.relu(cur_h + sum(pre_h_list))


class Progressive_neural_net(nn.Module):
    def __init__(self, layer_num):
        super(Progressive_neural_net, self).__init__()
        self.layer_num = layer_num
        self.columns = nn.ModuleList([])

    def add_new_column(self, sizes):
        # * sizes is a list which contains layer output size
        # * len(sizes) = self.layer_num + 1
        task_id = len(self.columns)
        modules = [Pnn_Linear(task_id, layer, sizes[layer], sizes[layer + 1]) for layer in range(self.layer_num)]
        new_column = nn.ModuleList(modules)
        self.columns.append(new_column)

    def forward(self, x, task_id=-1):
        # * first layer's outputs
        inputs = [column[0](x) for column in self.columns]

        for layer in range(1, self.layer_num):
            outputs = []
            for i, column in enumerate(self.columns):
                outputs.append(column[layer](inputs[: i + 1]))
            inputs = outputs
        return inputs[task_id]

    def freeze_columns(self):
        for column in self.columns:
            for params in column.parameters():
                params.requires_grad = False

    def parameters(self, col=None):
        if col is None:
            return super(Progressive_neural_net, self).parameters()
        else:
            return self.columns[col].parameters()
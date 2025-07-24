import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect
from loguru import logger
import ipdb


class PnnLinearBlock(nn.Module):
    def __init__(self, col, depth, n_in, n_out, add_activation=True):
        super(PnnLinearBlock, self).__init__()
        self.col = col
        self.depth = depth
        self.n_in = n_in
        self.n_out = n_out
        self.add_activation = add_activation
        self.w = nn.Linear(n_in, n_out)

        self.u = nn.ModuleList()
        if self.depth > 0:
            self.u.extend([nn.Linear(n_in, n_out) for _ in range(col)])

    def forward(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        cur_column_out = self.w(inputs[-1])
        prev_columns_out = [mod(x) for mod, x in zip(self.u, inputs)]
        output = cur_column_out + sum(prev_columns_out)

        return output if not self.add_activation else F.elu(output)


class PnnBaseModule(nn.Module):
    def __init__(self, obs_dim_dict, module_config_dict):
        super(PnnBaseModule, self).__init__()
        self.obs_dim_dict = obs_dim_dict
        self.module_config_dict = module_config_dict
        self.columns = nn.ModuleList([])
                
        self._calculate_input_dim()
        self._calculate_output_dim()
        
        self.use_cuda = True
    
    def _calculate_input_dim(self):
        # calculate input dimension based on the input specifications
        input_dim = 0
        for each_input in self.module_config_dict['input_dim']:
            if each_input in self.obs_dim_dict:
                # atomic observation type
                input_dim += self.obs_dim_dict[each_input]
            elif isinstance(each_input, (int, float)):
                # direct numeric input
                input_dim += each_input
            else:
                current_function_name = inspect.currentframe().f_code.co_name
                raise ValueError(f"{current_function_name} - Unknown input type: {each_input}")
        
        self.input_dim = input_dim

    def _calculate_output_dim(self):
        output_dim = 0
        for each_output in self.module_config_dict['output_dim']:
            if isinstance(each_output, (int, float)):
                output_dim += each_output
            else:
                current_function_name = inspect.currentframe().f_code.co_name
                raise ValueError(f"{current_function_name} - Unknown output type: {each_output}")
        self.output_dim = output_dim

    def new_task(self):
        task_id = len(self.columns)
        
        layers = []
        hidden_dims = self.module_config_dict.layer_config['hidden_dims']
        output_dim = self.output_dim
        activation = getattr(nn, self.module_config_dict.layer_config['activation'])()

        layers.append(PnnLinearBlock(task_id, 0, self.input_dim, hidden_dims[0]))
        
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                layers.append(PnnLinearBlock(task_id, l+1, hidden_dims[l], output_dim, add_activation=False))
            else:
                layers.append(PnnLinearBlock(task_id, l+1, hidden_dims[l], hidden_dims[l+1]))

        new_column = nn.ModuleList(layers)
        self.columns.append(new_column)
        
        self.n_layers = len(new_column)

        if self.use_cuda:
            self.cuda()

    def forward(self, x, task_id=-1):
        inputs = [c[0](x) for c in self.columns]

        for l in range(1, self.n_layers):
            outputs = []
            for i, column in enumerate(self.columns):
                outputs.append(column[l](inputs[:i+1]))

            inputs = outputs

        return inputs[task_id]

    def freeze_columns(self, skip=None):
        if skip == None:
            skip = []

        for i, c in enumerate(self.columns):
            if i not in skip:
                for params in c.parameters():
                    params.requires_grad = False

    def parameters(self, col=None):
        if col is None:
            return super(PnnBaseModule, self).parameters()
        return self.columns[col].parameters()

    def cuda(self, *args, **kwargs):
        self.use_cuda = True
        super(PnnBaseModule, self).cuda(*args, **kwargs)

    def cpu(self):
        self.use_cuda = False
        super(PnnBaseModule, self).cpu()
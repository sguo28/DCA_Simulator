import torch
import torch.nn as nn

class F_Network(nn.Module):
    def __init__(self):
        super(F_Network,self).__init__()
        ## global state
        self.global_fc = (nn.Linear(1*54*46, 256))
        self.output_f = nn.Linear(256, 1)

    def forward(self, global_state):
        flattened = torch.flatten(global_state, start_dim=1) # 1*54*46
        global_fc_out = self.global_fc(flattened)
        f_output = self.output_f(global_fc_out)
        return f_output

class Target_F_Network(nn.Module):
    def __init__(self):
        super(Target_F_Network,self).__init__()
        ## global state
        self.global_fc = (nn.Linear(1 * 54 * 46, 256))
        self.output_f = nn.Linear(256, 1)

    def forward(self, global_state):
        flattened = torch.flatten(global_state, start_dim=1)  # 1*54*46
        global_fc_out = self.global_fc(flattened)
        f_output = self.output_f(global_fc_out)
        return f_output

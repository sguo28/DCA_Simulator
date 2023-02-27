import torch
import torch.nn as nn
import hexagdly
import torch.nn.functional as F

class F_Network(nn.Module):
    def __init__(self):
        super(F_Network,self).__init__()
        ## global state
        self.global_fc = nn.Sequential(nn.Linear(1*54*46,256), nn.ReLU())
        self.fc1 = nn.Sequential(nn.Linear(256,64) , nn.ReLU())
        self.fc2 = nn.Linear(64, 2)


    def forward(self, global_state):
        flattened = torch.flatten(global_state, start_dim=1) # 1*54*46
        global_fc_out = self.global_fc(flattened)
        f_output1 = self.fc1(global_fc_out)
        f_output2 = self.fc2(f_output1)
        return f_output2


#this is to do convolution.
class F_Network_all(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(F_Network_all,self).__init__()
        ## global state
        self.global_fc = nn.Sequential((nn.Linear(64*6*6, 256)))  # ,nn.Dropout(0.5)
        ## local state
        self.output_f = nn.Linear(128,1+20) #two dims
        ## concat_fc
        ## concat_fc
        self.cat_fc = nn.Linear(64 * 6 * 6 + 1347 + 1, 256)
        self.dense1 = nn.Linear(54 * 46 * 3, 512)
        self.new_cat_fc = nn.Linear(512 + 1347 + 1, 128)

        self.local1=nn.Linear(1347+1,512)
        self.local2=nn.Linear(512,128)
        self.local_out=nn.Linear(128,1+8) #out dimension


    def forward(self, global_state,local_state):
        ## global state
        # conv1_out = F.relu(self.hexconv_1(global_state))  # 1, 16, 15, 18
        # conv2_out = F.relu(self.hexconv_2(conv1_out))  # 1, 64, 5, 6
        # flattened = torch.flatten(conv2_out, start_dim=1) # 1, 64*5*6
        # concat_fc = torch.cat((flattened,local_state[:,1:]),dim=1)
        # fc_out = F.relu(self.cat_fc(concat_fc))
        # global_state = global_state[:, [0, 1, 3], :, :]
        # g_feature = torch.flatten(global_state, start_dim=1)
        # d1 = F.relu(self.dense1(g_feature))
        # new_fc = torch.cat((d1, local_state[:, 1:]), dim=1)
        # fc_out = F.relu(self.new_cat_fc(new_fc))
        #
        # f_value = self.output_f(fc_out)

        state=F.relu(self.local1(local_state[:,1:]))
        state=F.relu(self.local2(state))
        f_value=self.output_f(state)

        return f_value





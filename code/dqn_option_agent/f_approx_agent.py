from collections import defaultdict
import numpy as np
import torch
from config.hex_setting import SAVING_CYCLE, EPSILON_DECAY_STEPS, F_AGENT_SAVE_PATH, NUM_REACHABLE_HEX
from dqn_option_agent.f_approx_network import F_Network
import random


class F_Agent:
    """
    F agent is to train the approximator of second eigenvector by hour.
    """
    def __init__(self,hex_diffusion,num_options,  isoption=False,islocal=True,ischarging=True):
        self.num_options = num_options
        self.learning_rate = 5e-4 # 5e-4
        self.epsilon_f_steps = EPSILON_DECAY_STEPS
        # self.traj_memory = TrajReplayMemory(TRAJECTORY_BUFFER_SIZE) # [TrajReplayMemory(REPLAY_BUFFER_SIZE) for _ in range(24)] # 24 hours
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.path = F_AGENT_SAVE_PATH
        # init option network
        self.f_network = [F_Network() for _ in range(24)] #24 agents # output a policy
        self.f_optimizer = [torch.optim.Adam(self.f_network[h].parameters(), lr=self.learning_rate) for h in range(24)]
        # self.f_optimizer = torch.optim.SGD(self.f_network.parameters(), lr=self.learning_rate,momentum=0.9)
        self.f_train_step = 0
        [self.f_network[h].to(self.device) for h in range(24)]

        self.record_list = []
        self.global_state_dict = defaultdict()
        self.with_option = isoption
        self.with_charging = ischarging
        self.local_matching = islocal
        self.hex_diffusion = hex_diffusion
        self.init_func = None
        self.term_func = None
        self.init_dist = 0.05
        self.term_dist = 0.05
        self.upper_threshold = 1e5
        self.lower_threshold = 1e5
        self.record_list = []
        self.training_data = [[] for _ in range (24)]

    def get_f_values(self, hex_ids,hr):
        hex_diffusions = [np.tile(self.hex_diffusion[int(hex_id)], (1, 1, 1)) for hex_id in hex_ids]  # state[1] is hex_id
        return self.f_network[hr].forward(torch.from_numpy(np.array(hex_diffusions)).to(dtype=torch.float32, device=self.device))

    # def add_hex_pair(self,current_hex,next_hex):
    #     self.traj_memory.push(current_hex,next_hex)
    def add_data(self,data):
        hr=data[0][0]//(60*60)%24 #get the hour
        self.training_data[hr].append(data)


    def add_od_pair(self,od_pairs):
        self.training_data = od_pairs

    def train(self,episode):
        for hr in range(24):
            print('Training in hours={}'.format(hr))
            for _ in range(episode):
                random.shuffle(self.training_data[hr])
                batch_size = 128
                for i in range(0,len(self.training_data[hr]),batch_size):  # 128 is batch size
                    self.f_train_step += 1
                    sample_batch = self.training_data[hr][i:i+batch_size]
                    f_values = self.get_f_values([item[0][1] for item in sample_batch],hr)
                    # add a mask
                    f_values_ = self.get_f_values([item[1][1] for item in sample_batch],hr)
                    eta = 2 # lagrangian multiplier, it was assumed as 1.0 in all scenarios, so we also try 1.0.
                    delta = 0.05 #according to Yifan Wang et al. 2019

                    f1=1 #(delta/NUM_REACHABLE_HEX)**0.5 #vaue in f(1)

                    # loss = (0.5 *(f_values_ - f_values).pow(2) + eta*((f_values_ - delta)*(f_values-delta) + f_values_.pow(2)*f_values.pow(2))).mean()  # + (f_values-f_values_).mean())
                    loss = (0.5 * (f_values_ - f_values).pow(2) + eta * ((f_values_**2 - delta)*(f_values**2-delta) + 2*(f1**2)*f_values_*f_values+(f1**2-delta)**2)).mean()  # + (f_values-f_values_).mean())
                    self.f_optimizer[hr].zero_grad()
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(self.f_network.parameters(), 1)
                    self.f_optimizer[hr].step()
                    self.record_list.append([self.f_train_step, round(float(loss), 4)])
                    if self.f_train_step%100==0:
                        print("Hour:{}, Step:{}, Loss:{}".format(hr, self.f_train_step,loss))
                        with open('saved_f/f_train_log_{}.csv'.format(self.num_options), 'a') as f:
                            f.writelines('Train step={}, hour={}, Loss = {}\n'.format(self.f_train_step, hr, loss))

    def save_f_vals(self,option_id):
        #threshold is the percentage to set terminate state
        all_f_by_hour = np.zeros((24, NUM_REACHABLE_HEX))
        for hr in range(24):
            zones=np.array([i for i in range(NUM_REACHABLE_HEX)],dtype=int)
            f_values = self.get_f_values(zones, hr)
            f_values=f_values.detach().cpu().numpy()
            all_f_by_hour[hr, :] = f_values.flatten()

        with open('saved_f/hex_p_value_1000_%d.csv'%(option_id), 'w') as f:
            f.writelines('hid,f_value\n')
            for hr in range(24):
                for zone in range(NUM_REACHABLE_HEX):
                    f.writelines('{},{},{}\n'.format(hr,zone,all_f_by_hour[hr,zone]))

        print('Value_saved')


    def save_parameter(self, hr, hist_file,episode):
        for hr in range(24):
            checkpoint = {
                "net": self.f_network[hr].state_dict()
            }
            print('f_approx is saving at {}'.format(
                self.path + 'f_network_o%d_h%d_e%d.pkl' % (self.num_options, hr, episode)))
            torch.save(checkpoint, self.path + 'f_network_o%d_h%d_e%d.pkl' % (self.num_options, hr, episode))
            # (bool(self.with_option),bool(self.with_charging),bool(self.local_matching)))
        for item in self.record_list:
            hist_file.writelines('{},{},{}\n'.format(item[0], hr, item[1]))
        self.record_list = []

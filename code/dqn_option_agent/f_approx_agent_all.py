from collections import defaultdict
import numpy as np
import torch
from config.hex_setting import SAVING_CYCLE, EPSILON_DECAY_STEPS, F_AGENT_SAVE_PATH, NUM_REACHABLE_HEX,INPUT_DIM
from dqn_option_agent.f_approx_network import F_Network_all
import random
from dqn_agent.dqn_feature_constructor import FeatureConstructor

class F_Agent:
    """
    F agent is to train the approximator of second eigenvector by hour.
    """
    def __init__(self,hex_diffusion,num_options,  isoption=False,islocal=True,ischarging=True):
        self.num_options = num_options
        self.learning_rate = 1e-3 # 5e-4
        self.epsilon_f_steps = EPSILON_DECAY_STEPS
        # self.traj_memory = TrajReplayMemory(TRAJECTORY_BUFFER_SIZE) # [TrajReplayMemory(REPLAY_BUFFER_SIZE) for _ in range(24)] # 24 hours
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.path = F_AGENT_SAVE_PATH
        # init option network
        self.f_network = F_Network_all(INPUT_DIM)
        self.f_optimizer = torch.optim.Adam(self.f_network.parameters(), lr=self.learning_rate)
        # self.f_optimizer = torch.optim.SGD(self.f_network.parameters(), lr=self.learning_rate,momentum=0.9)
        self.f_train_step = 0
        self.f_network.to(self.device)
        self.state_feature_constructor = FeatureConstructor()

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
        self.training_data = []

    # def add_hex_pair(self,current_hex,next_hex):
    #     self.traj_memory.push(current_hex,next_hex)
    def add_data(self,data):
        self.training_data.append(data)

    def train(self,episode,global_state):
        if len(self.training_data)>128:
            for _ in range(episode):
                random.shuffle(self.training_data)
                batch_size = 128
                for i in range(0,len(self.training_data),batch_size):  # 128 is batch size
                    self.f_train_step += 1
                    sample_batch = self.training_data[i:i+batch_size]
                    #first item is the first transition
                    global_state_reps = np.array([global_state[int(state[0][0] / 60)] for state in
                                                  sample_batch])  # should be list of np.array
                    global_next_state_reps = np.array([global_state[int(state[1][0] / 60)] for state in
                                                  sample_batch])  # should be list of np.array
                    state_reps = [self.state_feature_constructor.construct_state_features(state[0]) for state in
                                  sample_batch]
                    next_state_reps = [self.state_feature_constructor.construct_state_features(state[1]) for state in
                                       sample_batch]
                    hex_diffusion = [np.tile(self.hex_diffusion[state[0][1]], (1, 1, 1)) for state in sample_batch]
                    hex_diffusion_ = [np.tile(self.hex_diffusion[state[1][1]], (1, 1, 1)) for state in sample_batch]

                    state_batch = torch.from_numpy(np.array(state_reps)).to(dtype=torch.float32, device=self.device)
                    next_state_batch = torch.from_numpy(np.array(next_state_reps)).to(device=self.device,
                                                                                      dtype=torch.float32)
                    global_state_batch = torch.from_numpy(
                        np.concatenate([np.array(global_state_reps), np.array(hex_diffusion)], axis=1)).to(
                        dtype=torch.float32,
                        device=self.device)
                    global_next_state_batch = torch.from_numpy(
                        np.concatenate([np.array(global_next_state_reps), np.array(hex_diffusion_)], axis=1)).to(
                        dtype=torch.float32, device=self.device)
                    f_values=self.f_network.forward(global_state_batch,state_batch)
                    f_values_=self.f_network.forward(global_next_state_batch,next_state_batch)
                    # print(f_values.mean(),f_values.max(),f_values.min())
                    eta = 2 # lagrangian multiplier, it was assumed as 1.0 in all scenarios, so we also try 1.0.
                    delta = 0.05 #according to Yifan Wang et al. 2019
                    f1=1 #(delta/NUM_REACHABLE_HEX)**0.5 #vaue in f(1)
                    # loss = (0.5 *(f_values_ - f_values).pow(2) + eta*((f_values_ - delta)*(f_values-delta) + f_values_.pow(2)*f_values.pow(2))).mean()  # + (f_values-f_values_).mean())
                    loss = (0.5 * (f_values_ - f_values).pow(2) + eta * ((f_values_**2 - delta)*(f_values**2-delta) + 2*(f1**2)*f_values_*f_values+(f1**2-delta)**2)).mean()  # + (f_values-f_values_).mean())
                    self.f_optimizer.zero_grad()
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(self.f_network.parameters(), 1)
                    self.f_optimizer.step()
                    self.record_list.append([self.f_train_step, round(float(loss), 4)])
                    if self.f_train_step%100==0:
                        print("Step:{}, Loss:{}".format(self.f_train_step,loss))
                        with open('saved_f/f_train_log_{}.csv'.format(self.num_options), 'a') as f:
                            f.writelines('Train step={}, Loss = {}\n'.format(self.f_train_step, loss))


    def save_parameter(self):
        # torch.save(self.q_network.state_dict(), self.dqn_path)
        checkpoint = {
                "net": self.f_network.state_dict()
                # "step": self.train_step,
                # "lr_scheduler": self.lr_scheduler.state_dict()
            }
        torch.save(checkpoint, 'saved_f/f_network_option_1000_%d.pkl' % (self.num_options))
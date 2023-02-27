from config.hex_setting import LEARNING_RATE, GAMMA, REPLAY_BUFFER_SIZE, BATCH_SIZE, RELOCATION_DIM, \
    INPUT_DIM, FINAL_EPSILON, HIGH_SOC_THRESHOLD, LOW_SOC_THRESHOLD, CLIPPING_VALUE, START_EPSILON, \
    EPSILON_DECAY_STEPS, H_AGENT_SAVE_PATH, SAVING_CYCLE,  NUM_REACHABLE_HEX, \
    TERMINAL_STATE_SAVE_PATH
from .option_network import TargetOptionNetwork, OptionNetwork
from .replay_buffers import OptionReplayMemory
from dqn_agent.replay_buffer import Prime_ReplayMemory
from collections import deque, defaultdict, OrderedDict
import os
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dqn_agent.dqn_feature_constructor import FeatureConstructor
from torch.optim.lr_scheduler import StepLR
from dqn_option_agent.f_approx_network import F_Network_all


class H_Agent:

    def __init__(self, hex_diffusion, num_option, H_memory=None, H_global=None, isoption=True, islocal=True, ischarging=False):
        self.num_option =int(num_option)
        self.learning_rate = 1e-4
        self.gamma = GAMMA
        self.start_epsilon = START_EPSILON
        self.final_epsilon = FINAL_EPSILON
        self.epsilon_steps = EPSILON_DECAY_STEPS
        self.beta_threshold=1
        if H_memory is None:
            self.memory = Prime_ReplayMemory(int(2e6)) #Prime_ReplayMemory(int(1e5))
        else:
            self.memory=H_memory

        self.batch_size = int(512)
        self.clipping_value = CLIPPING_VALUE
        self.input_dim = INPUT_DIM
        self.relocation_dim = RELOCATION_DIM
        self.option_dim = num_option  # put 0 first, change it later. OPTION_DIM  # type: int
        self.premitive_action_dim = 1 + 6 + 5
        self.output_dim=self.premitive_action_dim+self.option_dim
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.path = H_AGENT_SAVE_PATH
        self.state_feature_constructor = FeatureConstructor()
        self.option_network = OptionNetwork(self.input_dim, self.premitive_action_dim).to(self.device)
        self.f_network = F_Network_all(INPUT_DIM)
        self.load_f_params()
        self.f_network.to(self.device)
        self.f_threshold=0.3


        self.target_option_network = TargetOptionNetwork(self.input_dim, self.premitive_action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.option_network.parameters(), lr=self.learning_rate)
        self.lr_scheduler = StepLR(optimizer=self.optimizer, step_size=1000,
                                   gamma=0.99)  # 1.79 e-6 at 0.5 million step.
        self.train_step = 0
        self.option_network.to(self.device)
        self.target_option_network.to(self.device)
        self.decayed_epsilon = self.start_epsilon

        self.hex_diffusion = hex_diffusion
        # self.init_f_values_and_terminate_state()

        self.record_list = []
        self.hex_xy=np.zeros((NUM_REACHABLE_HEX,2))
        if H_global is None:
            self.global_state_dict = OrderedDict()
        else:
            self.global_state_dict=H_global
        self.global_state_capacity = 7 * 1440  # we store 5 days' global states to fit replay buffer size.
        self.time_interval = int(0)
        self.with_option = isoption
        self.with_charging = ischarging
        self.local_matching = islocal
        self.option_queue = deque()
        self.init_func = None
        self.term_func = None
        self.init_dist = 0.05
        self.term_dist = 0.05
        self.upper_threshold = 1e5
        self.lower_threshold = 1e5

    def load_f_params(self):
        checkpoint = torch.load('saved_f/f_network_option_1000_%d.pkl' % (0))  # lets try the saved networks after the 14th day.
        self.f_network.load_state_dict(checkpoint['net'])  # , False
        print('Successfully load f network {}, total option network num is {}'.format(0,1))

    def init_f_values_and_terminate_state(self):

        all_f_by_hour = np.zeros((24, NUM_REACHABLE_HEX))
        with open('saved_f/hex_p_value_1000_%d.csv'%(self.num_option), 'r') as f:
            next(f)
            for lines in f:
                line = lines.strip().split(',')
                hr, hid, f_value = line  # hour, oridin, dest, trip_time/num of trip
                all_f_by_hour[int(hr)][int(hid)] = float(f_value)
        self.all_f_values = all_f_by_hour
        self.f_means_per_hour = np.median(all_f_by_hour, axis=1)

        f_threshold_dict = defaultdict()
        middle_mask = defaultdict()
        term_percentile = 35 #do 7.5 percentile

        for hr in range(24):
            f_middle_sorted = np.sort(self.all_f_values[hr])
            zero_percentile=sum(f_middle_sorted<0)/len(f_middle_sorted)*100 #find the percentile of the 0 value
            zero_percentile=50
            f_lower_threshold = np.percentile(f_middle_sorted ,max((zero_percentile - term_percentile),0))
            f_higher_threshold = np.percentile(f_middle_sorted ,min((zero_percentile + term_percentile),100))
            # f_lower_threshold = -0.2;
            # f_higher_threshold = 0.2;
            f_threshold_dict[hr] = [f_lower_threshold, f_higher_threshold]
            #middle_mask[hr] = (self.all_f_values[hr]>f_lower_threshold) & (self.all_f_values[hr]<f_higher_threshold)

            #this is based on relocation only
            middle_mask[hr] = (self.all_f_values[hr] < f_lower_threshold) | (
                        self.all_f_values[hr] > f_higher_threshold)
        self.middle_mask = middle_mask  # 24 by 1347
        with open('saved_f/term_states_%d.csv'%(self.num_option),'w') as ts:
            for hr in range(24):
                for hex_id,term_flag in enumerate(middle_mask[hr]):
                    if term_flag == 1:
                        ts.writelines('{},{}\n'.format(hr,hex_id))
            print('finished record terminal state!!!')

    def copy_parameter(self):
        self.target_option_network.load_state_dict(self.option_network.state_dict())

    def get_f_values(self, state_batch):
        """
        get f values from pre-stored dict.
        :param state_batch:
        :return:
        """
        return [self.all_f_values[state[0] // (60 * 60) % 24][state[1]] for state in state_batch]

    def get_f_mean_by_hour(self, state_batch):
        return [self.f_means_per_hour[state[0] // (60 * 60) % 24] for state in state_batch]

    def is_middle_terminal(self, states):
        return [True if self.middle_mask[state[0] // (60 * 60) % 24][state[1]] == 1 else False for state in states]

    # shuffle ==> minibatch ==> deep learning
    def add_transition(self, state, action, next_state, trip_flag, time_steps, valid_action):
        self.memory.push(state, action, next_state, trip_flag, time_steps, valid_action)

    def batch_sample(self):
        samples = self.memory.sample(self.batch_size)  # random.sample(self.memory, self.batch_size)
        return samples

    def get_main_Q(self, local_state, global_state):
        return self.option_network.forward(local_state, global_state)

    def get_target_Q(self, local_state, global_state):
        return self.target_option_network.forward(local_state, global_state)


    def add_global_state_dict(self, global_state_list):
        for tick in global_state_list.keys():
            if tick not in self.global_state_dict.keys():
                self.global_state_dict[tick] = global_state_list[tick]
        if len(self.global_state_dict.keys()) > self.global_state_capacity: #capacity limit for global states
            for _ in range(len(self.global_state_dict.keys())-self.global_state_capacity):
                self.global_state_dict.popitem(last=False)


    def soft_target_update(self, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(self.target_option_network.parameters(), self.option_network.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def train(self):
        self.train_step += 1
        if len(self.memory) < self.batch_size:
            print('batches in replay buffer is {}'.format(len(self.memory)))
            return

        transitions = self.batch_sample()
        batch = self.memory.Transition(*zip(*transitions))

        global_state_reps = np.array([self.global_state_dict[int(state[0] / 60)] for state in
                             batch.state])  # should be list of np.array

        global_next_state_reps = np.array([self.global_state_dict[int(state_[0] / 60)] for state_ in
                                  batch.next_state]) # should be list of np.array

        next_zones = np.array([state_[1] for state_ in batch.next_state])  # zone id for choosing actions

        state_reps = [self.state_feature_constructor.construct_state_features(state) for state in batch.state]
        next_state_reps = [self.state_feature_constructor.construct_state_features(state_) for state_ in
                           batch.next_state]

        hex_diffusion = [np.tile(self.hex_diffusion[state[1]], (1, 1, 1)) for state in batch.state]
        hex_diffusion_ = [np.tile(self.hex_diffusion[state_[1]], (1, 1, 1)) for state_ in batch.next_state]

        state_batch = torch.from_numpy(np.array(state_reps)).to(dtype=torch.float32, device=self.device)

        action_batch = torch.from_numpy(np.array(batch.action)).unsqueeze(1).to(dtype=torch.int64, device=self.device)

        time_step_batch = torch.from_numpy(np.array(batch.time_steps)).unsqueeze(1).to(dtype=torch.float32, device=self.device)

        trip_flag=torch.from_numpy(np.array(batch.trip_flag)).unsqueeze(1).to(dtype=torch.float32, device=self.device)

        next_state_batch = torch.from_numpy(np.array(next_state_reps)).to(device=self.device, dtype=torch.float32)
        global_state_batch = torch.from_numpy(
            np.concatenate([np.array(global_state_reps), np.array(hex_diffusion)], axis=1)).to(dtype=torch.float32,
                                                                                               device=self.device)
        global_next_state_batch = torch.from_numpy(
            np.concatenate([np.array(global_next_state_reps), np.array(hex_diffusion_)], axis=1)).to(
            dtype=torch.float32, device=self.device)

        # print('The shape for the global state reps after fusion are {} and {}'.format(len(global_state_batch),
        #                                                                       [i.sum() for i in global_state_batch]))
        #
        # print('The shape for the global next state reps after fusion are {} and {}'.format(len(global_next_state_batch),
        #                                                                       [i.sum() for i in global_next_state_batch]))
        #
        # f_s = torch.from_numpy(np.array(self.get_f_values(batch.state))).unsqueeze(1).to(dtype=torch.float32, device=self.device)
        # f_s_ = torch.from_numpy(np.array(self.get_f_values(batch.next_state))).unsqueeze(1).to(dtype=torch.float32,
        #                                                                           device=self.device)
        # f_median = torch.from_numpy(np.array(self.get_f_mean_by_hour(batch.state))).unsqueeze(1).to(dtype=torch.float32,
        #                                                                                device=self.device)

        f_s=self.f_network.forward(global_state_batch,state_batch)
        f_s=f_s.detach()
        f_s_ = self.f_network.forward(global_next_state_batch, next_state_batch)
        f_s_ = f_s_.detach()

        middle_terminal_flag = ((f_s<-self.f_threshold) | (f_s>self.f_threshold)).to(dtype=torch.float32,device=self.device)
        middle_next_terminal_flag = ((f_s_<-self.f_threshold) | (f_s_>self.f_threshold)).to(dtype=torch.float32,device=self.device)

        # middle_next_terminal_flag = torch.from_numpy(np.array((self.is_middle_terminal(batch.next_state)))).unsqueeze(1).to(
        #     dtype=torch.float32, device=self.device)

        q_state_action = self.get_main_Q(state_batch, global_state_batch).gather(1, action_batch.long())

        # add a mask
        all_q_ = self.get_target_Q(next_state_batch, global_next_state_batch) #lets change this to global state batch and see if error continues
        mask = self.get_action_mask(batch.next_state, batch.valid_action_num)  # action mask for next state
        all_q_[mask] = -9e10
        maxq = all_q_.max(1)[0].detach().unsqueeze(1)
        #
        # f_s=f_s-f_median #convert to 0 median
        # f_s_=f_s_-f_median #convert to 0 median
        #

        #pseudo_reward = torch.abs(f_s)-torch.abs(f_s_)
        #this following is for relocating from middle to peripheral
        pseudo_reward = torch.sign(f_s)*(f_s_-f_s)
        mean_diff =pseudo_reward.mean()

        pseudo_reward+=0.02* trip_flag #bonus for matching
        #use demand and supply gap to update hex_zones

        # hex_rows=self.hex_xy[next_zones,0] #find the rows.
        # hex_cols=self.hex_xy[next_zones,1] #find the columns
        # demand_supply_gap=global_state_reps[:,0,:]/5-global_state_reps[:,1,:]
        # gap_reward=demand_supply_gap[np.arange(len(hex_rows)),hex_rows,hex_cols]
        # #
        # pseudo_reward+=0.01*torch.from_numpy(gap_reward).unsqueeze(1).to(dtype=torch.float32, device=self.device) #demand supply gap

        mean_pseudo=pseudo_reward.mean()
        y =(1-middle_terminal_flag)*(pseudo_reward + (1 - middle_next_terminal_flag)*(1-trip_flag)*maxq*torch.pow(self.gamma,time_step_batch))
        loss = F.smooth_l1_loss(q_state_action,y)


        if self.train_step%100==0:
            print('Max Q={}, Max target Q={}, Loss = {}, Gamma={}, mean diff={}, mean reward={}'.format(torch.max(q_state_action), torch.max(maxq),loss, self.gamma,mean_diff,mean_pseudo))
            print('Mean of main f={}, mean of target f={}'.format(f_s.mean(),f_s_.mean()))
            with open('saved_h/ht_train_log_{}.csv'.format(self.num_option),'a') as f:
                f.writelines('Train step={}, Max Q={}, Max target Q={}, Loss = {}, Mean f_diff={}, Mean pseudo-reward={}\n'.format(self.train_step,torch.max(q_state_action), torch.max(maxq),loss,mean_diff,mean_pseudo))

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.option_network.parameters(), self.clipping_value)
        self.optimizer.step()
        self.record_list.append([self.train_step, round(float(loss), 4), torch.max(maxq)])
        self.lr_scheduler.step()


    def save_parameter(self):
        # torch.save(self.q_network.state_dict(), self.dqn_path)
        checkpoint = {
                "net": self.option_network.state_dict()
                # "step": self.train_step,
                # "lr_scheduler": self.lr_scheduler.state_dict()
            }
        torch.save(checkpoint, 'saved_h/ht_network_option_1000_%d.pkl' % (self.num_option))

    def get_action_mask(self, batch_state, batch_valid_action):
        mask = np.zeros([len(batch_state), self.premitive_action_dim])
        for i, state in enumerate(batch_state):
            mask[i][batch_valid_action[i]:self.relocation_dim] = 1
            # here the SOC in state is still continuous. the categorized one is in state reps.
            if state[-1] > HIGH_SOC_THRESHOLD:
                mask[i][self.relocation_dim:] = 1  # no charging, must relocate
            elif state[-1] < LOW_SOC_THRESHOLD:
                mask[i][:self.relocation_dim] = 1  # no relocation, must charge

        mask = torch.from_numpy(mask).to(dtype=torch.bool, device=self.device)
        return mask

    # def get_option_mask(self,states):
    #     """
    #     append masks to the options that think the states as terminate.
    #     if no option is generated yet, we keep the option masked as 1. (we initial the mask by np.ones)
    #     :param states:
    #     :return:
    #     """
    #     termiante_option_mask = np.ones((len(states),self.option_dim))
    #     for hr,op in enumerate(self.option_queue):
    #         for state in states:
    #             if self.is_initial(state):
    #                 termiante_option_mask[:,] = 0 # the j-th element of the i-th column is masked
    #
    #     termiante_option_mask = torch.from_numpy(termiante_option_mask).to(dtype=torch.bool, device=self.device)
    #     return termiante_option_mask

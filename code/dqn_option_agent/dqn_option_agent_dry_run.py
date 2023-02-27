import os
import random
from collections import OrderedDict,defaultdict
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config.setting import LEARNING_RATE, GAMMA, REPLAY_BUFFER_SIZE, BATCH_SIZE, RELOCATION_DIM, CHARGING_DIM, \
    INPUT_DIM, OPTION_DIM, FINAL_EPSILON, HIGH_SOC_THRESHOLD, LOW_SOC_THRESHOLD, CLIPPING_VALUE, START_EPSILON, \
    EPSILON_DECAY_STEPS, SAVING_CYCLE, STORE_TRANSITION_CYCLE, NUM_REACHABLE_HEX, OPTION_DQN_SAVE_PATH, \
    DQN_OUTPUT_DIM, H_AGENT_SAVE_PATH, TERMINAL_STATE_SAVE_PATH
from .dqn_option_network import DQN_network, DQN_target_network
from .dqn_option_feature_constructor import FeatureConstructor
from .replay_buffers import BatchReplayMemory
from torch.optim.lr_scheduler import StepLR
from .option_network import OptionNetwork


class DeepQNetworkOptionAgent:
    def __init__(self,hex_diffusion, option_num, isoption=False,islocal=True,ischarging=True):
        self.learning_rate = 1e-3  # 1e-4
        self.gamma = GAMMA
        self.start_epsilon = START_EPSILON
        self.final_epsilon = FINAL_EPSILON
        self.epsilon_steps = EPSILON_DECAY_STEPS
        self.memory = BatchReplayMemory(256)
        self.batch_size = BATCH_SIZE
        self.clipping_value = CLIPPING_VALUE
        self.input_dim = INPUT_DIM  # 3 input state
        self.relocation_dim = RELOCATION_DIM  # 7
        self.charging_dim = CHARGING_DIM  # 5
        self.option_dim = OPTION_DIM  # 3
        self.output_dim = DQN_OUTPUT_DIM  # 7+5+3 = 15
        self.num_option = option_num
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.path = OPTION_DQN_SAVE_PATH
        self.state_feature_constructor = FeatureConstructor()

        # init higher level DQN network
        self.q_network = DQN_network(self.input_dim, self.output_dim)
        self.target_q_network = DQN_target_network(self.input_dim, self.output_dim)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.lr_scheduler = StepLR(optimizer=self.optimizer,step_size=1000, gamma=0.99) # 1.79 e-6 at 0.5 million step.
        self.train_step = 0
        # self.load_network()
        self.q_network.to(self.device)
        self.target_q_network.to(self.device)

        self.decayed_epsilon = self.start_epsilon
        # init option network
        self.record_list = []
        self.global_state_dict = OrderedDict()
        self.time_interval = int(0)
        self.global_state_capacity = 5*1440 # we store 5 days' global states to fit replay buffer size.
        self.with_option = isoption
        self.with_charging = ischarging
        self.local_matching = islocal
        self.hex_diffusion = hex_diffusion

        self.h_network_list = []
        self.load_option_networks(self.num_option)
        self.middle_terminal = self.init_terminal_states()


    # def load_network(self, RESUME = False):
    #     if RESUME:
    #         lists = os.listdir(self.path)
    #         lists.sort(key=lambda fn: os.path.getmtime(self.path + "/" + fn))
    #         newest_file = os.path.join(self.path, lists[-1])
    #         path_checkpoint = newest_file
    #         checkpoint = torch.load(path_checkpoint)
    #
    #         self.q_network.load_state_dict(checkpoint['net'])
    #         self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    #
    #         self.train_step = checkpoint['step']
    #         self.copy_parameter()
    #         # self.optimizer.load_state_dict(checkpoint['optimizer'])
    #         print('Successfully load saved network starting from {}!'.format(str(self.train_step)))

    def load_option_networks(self,option_num):
        for option_net_id in range(option_num):
            h_network = OptionNetwork(self.input_dim,1+6+5)
            checkpoint = torch.load(H_AGENT_SAVE_PATH + 'ht_network_option_%d_1_0_1_11520.pkl'%(option_net_id))  # lets try the saved networks after the 14th day.
            h_network.load_state_dict(checkpoint['net'])  # , False
            self.h_network_list.append(h_network.to(self.device))
            print('Successfully load H network {}, total option network num is {}'.format(option_net_id,len(self.h_network_list)))

    def init_terminal_states(self):
        """
        we initial a dict to check the sets of terminal hex ids by hour by option id
        :param oid: ID for option network
        :return:
        """
        middle_terminal = defaultdict(list)
        for oid in range(self.num_option):
            with open(TERMINAL_STATE_SAVE_PATH+'term_states_%d.csv'%oid,'r') as ts:
                next(ts)
                for lines in ts:
                    line = lines.strip().split(',')
                    hr, hid = line  # option_network_id, hour, hex_ids in terminal state
                    middle_terminal[(oid,int(hr))].append(hid)
        return middle_terminal

    def get_actions(self, states, num_valid_relos, assigned_option_ids, global_state):
        """
        option_ids is at the first three slots in the action space, so action id <3 means the corresponding h_network id
        :param global_states:
        :param states: tuple of (tick, hex_id, SOC) and SOC is 0 - 100%
        :param num_valid_relos: only relocation to ADJACENT hexes / charging station is valid
        :states:
        :return: action ids ranges from (0,14) , converted action ids has converted the option ids to hte action ids that are selected by corresponding option networks
        """
        with torch.no_grad():
            self.decayed_epsilon = max(self.final_epsilon, (self.start_epsilon - self.train_step * (
                    self.start_epsilon - self.final_epsilon) / self.epsilon_steps))

            state_reps = np.array([self.state_feature_constructor.construct_state_features(state) for state in states])
            hex_diffusions = np.array(
                [np.tile(self.hex_diffusion[state[1]], (1, 1, 1)) for state in states])  # state[1] is hex_id
            
            mask = self.get_action_mask(states, num_valid_relos)  # mask for unreachable primitive actions

            option_mask = self.get_option_mask(states) # if the state is considered as terminal, we dont use it..
            # terminate_option_mask = torch.from_numpy(option_mask).to(dtype=torch.bool, device=self.device) # the DQN need a tensor as input, so convert it.

            if True:            
                full_action_values = np.random.random(
                    (len(states), self.output_dim))  # generate a matrix with values from 0 to 1
                for i, state in enumerate(states):
                    if assigned_option_ids[i] != -1:
                        full_action_values[i][assigned_option_ids[i]] = 10 # a large enough number to maintain that option if it's terminal state, we next mask it with -1.
                    full_action_values[i][:self.option_dim] = np.negative(option_mask[i,:self.option_dim]) # convert terminal agents to -1
                    full_action_values[i][(self.option_dim+num_valid_relos[i]):(self.option_dim+self.relocation_dim)] = -1 # mask unreachable neighbors.
                    if state[-1] > HIGH_SOC_THRESHOLD:
                        full_action_values[i][(self.option_dim+self.relocation_dim):] = -1  # no charging, must relocate
                    elif state[-1] < LOW_SOC_THRESHOLD:
                        full_action_values[i][:(self.option_dim+self.relocation_dim)] = -1  # no relocation, must charge
                action_indexes = np.argmax(full_action_values, 1).tolist()
                # # hard inplace the previously assigned options. 
                # action_indexes[np.where(assigned_option_ids!=-1)] = assigned_option_ids[np.where(assigned_option_ids!=-1)] 
            # after getting all action ids by DQN, we convert the ones triggered options to the primitive action ids.
            converted_action_indexes = self.convert_option_to_primitive_action_id(action_indexes,state_reps,global_state,hex_diffusions,mask)

        return np.array(action_indexes), np.array(converted_action_indexes)-self.option_dim

    def convert_option_to_primitive_action_id(self,action_indexes,state_reps,global_state,hex_diffusions,mask):
        """
        we convert the option ids, e.g., 0,1,2 for each H network, to the generated primitive action ids
        :param action_indexes:
        :param state_reps:
        :param global_state:
        :param hex_diffusions:
        :param mask:
        :return:
        """
        ids_require_option= defaultdict(list)
        for id, action_id in enumerate(action_indexes):
            if action_id < self.num_option:
                ids_require_option[action_id].append(id)
        for option_id in range(self.num_option):
            if ids_require_option[option_id]:
                full_option_values = self.h_network_list[option_id].forward(
                    torch.from_numpy(state_reps[ids_require_option[option_id]]).to(dtype=torch.float32, device=self.device),
                    torch.from_numpy(np.concatenate(
                        [np.tile(global_state, (len(ids_require_option[option_id]), 1, 1, 1)), hex_diffusions[ids_require_option[option_id]]],
                        axis=1)).to(dtype=torch.float32, device=self.device))
                # here mask is of batch x 15 dimension, we omit the first 3 columns, which should be options.
                primitive_action_mask = mask[ids_require_option[option_id],self.option_dim:]  # only primitive actions in option generator
                full_option_values[primitive_action_mask] = -9e10
                option_generated_premitive_action_ids = torch.argmax(full_option_values, dim=1).tolist()  # let option network select primitive action
                action_indexes[ids_require_option[option_id]] = option_generated_premitive_action_ids+self.option_dim # 12 to 15
                # cover the option id with the generated primitive action id
        return action_indexes

    def add_global_state_dict(self, global_state_list):
        for tick in global_state_list.keys():
            if tick not in self.global_state_dict.keys():
                self.global_state_dict[tick] = global_state_list[tick]
        if len(self.global_state_dict.keys()) > self.global_state_capacity: #capacity limit for global states
            for _ in range(len(self.global_state_dict.keys())-self.global_state_capacity):
                self.global_state_dict.popitem(last=False)

    def add_transition(self, state, action, next_state, reward, terminate_flag, time_steps, valid_action):
        self.memory.push(state, action, next_state, reward, terminate_flag, time_steps, valid_action)

    def batch_sample(self):
        samples = self.memory.sample(self.batch_size)  # random.sample(self.memory, self.batch_size)
        return samples
        # state, action, next_state, reward = zip(*samples)
        # return state, action, next_state, reward

    def get_main_Q(self, local_state, global_state):
        return self.q_network.forward(local_state, global_state)

    def get_target_Q(self, local_state, global_state):
        return self.target_q_network.forward(local_state, global_state)

    def copy_parameter(self):
        self.target_q_network.load_state_dict(self.q_network.state_dict())

    def soft_target_update(self, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(self.target_q_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


    def train(self, record_hist):
        self.train_step += 1
        if len(self.memory) < self.batch_size:
            print('batches in replay buffer is {}'.format(len(self.memory)))
            return

        transitions = self.batch_sample()
        batch = self.memory.Transition(*zip(*transitions))

        global_state_reps = [self.global_state_dict[int(state[0] / 60)] for state in
                             batch.state]  # should be list of np.array

        global_next_state_reps = [self.global_state_dict[int(state_[0] / 60)] for state_ in
                                  batch.next_state]  # should be list of np.array

        state_reps = [self.state_feature_constructor.construct_state_features(state) for state in batch.state]
        next_state_reps = [self.state_feature_constructor.construct_state_features(state_) for state_ in
                           batch.next_state]

        hex_diffusion = [np.tile(self.hex_diffusion[state[1]],(1,1,1)) for state in batch.state]
        hex_diffusion_ = [np.tile(self.hex_diffusion[state_[1]],(1,1,1)) for state_ in batch.next_state]

        state_batch = torch.from_numpy(np.array(state_reps)).to(dtype=torch.float32, device=self.device)
        action_batch = torch.from_numpy(np.array(batch.action)).unsqueeze(1).to(dtype=torch.int64, device=self.device)
        reward_batch = torch.from_numpy(np.array(batch.reward)).unsqueeze(1).to(dtype=torch.float32, device=self.device)
        time_step_batch = torch.from_numpy(np.array(batch.time_steps)).unsqueeze(1).to(dtype=torch.float32, device=self.device)

        next_state_batch = torch.from_numpy(np.array(next_state_reps)).to(device=self.device, dtype=torch.float32)
        global_state_batch = torch.from_numpy(np.concatenate([np.array(global_state_reps),np.array(hex_diffusion)],axis=1)).to(dtype=torch.float32, device=self.device)
        global_next_state_batch = torch.from_numpy(np.concatenate([np.array(global_next_state_reps), np.array(hex_diffusion_)],axis=1)).to(dtype=torch.float32,
                                                                                        device=self.device)

        q_state_action = self.get_main_Q(state_batch, global_state_batch).gather(1, action_batch.long())
        # add a mask
        all_q_ = self.get_target_Q(next_state_batch, global_next_state_batch)
        option_mask = self.get_option_mask(batch.next_state)
        mask_ = self.get_action_mask(batch.next_state, batch.valid_action_num)  # action mask for next state
        all_q_[option_mask] = -9e10
        all_q_[mask_] = -9e10
        maxq = all_q_.max(1)[0].detach().unsqueeze(1)
        y = reward_batch + maxq*torch.pow(self.gamma,time_step_batch)
        loss = F.smooth_l1_loss(q_state_action, y)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), self.clipping_value)
        self.optimizer.step()
        self.lr_scheduler.step()

        self.record_list.append([self.train_step, round(float(loss),3), round(float(reward_batch.view(-1).mean()),3)])
        self.save_parameter(record_hist)
        print('Training step is {}; Learning rate is {}; Epsilon is {}:'.format(self.train_step,self.lr_scheduler.get_lr(),round(self.decayed_epsilon,4)))

    def get_action_mask(self, batch_state, batch_valid_action):
        """
        the action space: the first 3 is for h_network slots, then 7 relocation actions,and 5 nearest charging stations.
        :param batch_state: state
        :param batch_valid_action: info that limites to relocate to reachable neighboring hexes
        :return:
        """
        mask = np.zeros((len(batch_state), self.output_dim))  # (num_state, 15)
        for i, state in enumerate(batch_state):
            mask[i][(self.option_dim+ batch_valid_action[i]):(self.option_dim+self.relocation_dim)] = 1  # limited to relocate to reachable neighboring hexes
            if state[-1] > HIGH_SOC_THRESHOLD:
                mask[i][(self.option_dim+self.relocation_dim):] = 1  # no charging, must relocate
            elif state[-1] < LOW_SOC_THRESHOLD:
                mask[i][:(self.option_dim+self.relocation_dim)] = 1  # no relocation, must charge

        mask = torch.from_numpy(mask).to(dtype=torch.bool, device=self.device)
        return mask

    def get_option_mask(self,states):
        """
        self.is_terminal is to judge if the state is terminal state with the info of hour and hex_id
        :param states:
        :return:
        """
        terminate_option_mask = np.zeros((len(states),self.output_dim))
        for oid in range(self.num_option):
            terminate_option_mask[:,oid] = self.is_terminal(states,oid)  # set as 0 if not in terminal set
        for oid in range(self.num_option,self.option_dim):
            terminate_option_mask[:,oid] = 1 # mask out empty options
        return terminate_option_mask

    def is_terminal(self,states,oid):
        """

        :param states:
        :return: a list of bool
        """
        return [1 if state in self.middle_terminal[(oid,int(state[0] // (60 * 60) % 24))] else 0 for state in states]

    def is_initial(self,states,oid):
        """

        :param states:
        :return: a list of bool
        """
        return [1 if state not in self.middle_terminal[(oid,int(state[0] // (60 * 60) % 24))] else 0 for state in states]


    def save_parameter(self, record_hist):
        # torch.save(self.q_network.state_dict(), self.dqn_path)
        if self.train_step % SAVING_CYCLE == 0:
            checkpoint = {
                "net": self.q_network.state_dict(),
                # 'optimizer': self.optimizer.state_dict(),
                "step": self.train_step,
                "lr_scheduler": self.lr_scheduler.state_dict()
            }
            if not os.path.isdir(self.path):
                os.mkdir(self.path)
            # print('the path is {}'.format('logs/dqn_model/duel_dqn_%s.pkl'%(str(self.train_step))))
            torch.save(checkpoint, 'logs/test/cnn_dqn_model/dqn_with_option_%d_%d_%d_%d_%s.pkl' % (self.num_option,bool(self.with_option),bool(self.with_charging),bool(self.local_matching),str(self.train_step)))
            # record training process (stacked before)
            for item in self.record_list:
                record_hist.writelines('{},{},{}\n'.format(item[0], item[1], item[2]))
            print('Training step: {}, replay buffer size:{}, epsilon: {}, learning rate: {}'.format(self.record_list[-1][0],len(self.memory), self.decayed_epsilon,self.lr_scheduler.get_lr()))
            self.record_list = []


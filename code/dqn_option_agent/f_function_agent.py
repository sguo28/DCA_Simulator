from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from config.setting import LEARNING_RATE, GAMMA, REPLAY_BUFFER_SIZE, BATCH_SIZE, RELOCATION_DIM, CHARGING_DIM, \
    INPUT_DIM, OPTION_DIM, FINAL_EPSILON, CLIPPING_VALUE, START_EPSILON, EPSILON_DECAY_STEPS, F_AGENT_SAVE_PATH
from .f_approximator import F_Network, Target_F_Network
from .dqn_option_feature_constructor import FeatureConstructor
from torch.optim.lr_scheduler import StepLR
from .replay_buffers import TrajReplayMemory
from collections import deque


class F_Agent:
    """
    todo 1: logic is to first train options
    todo 2: enlarge the action space to 12+10, additional 10 are options that are sampled ~epsilon greedy policy. [done]
    todo 3: embed an Option-DQN module to generate the 10 options.
    todo 4: carefully design the training module of Option-DQN, with the training loss being delta f_value, refer to deep covering option.
    todo 5: how to covert option-id to hex_id, so that we can correctly interpret them and dispatch the vehicle to right place.
    """
    def __init__(self,hex_diffusion, device, isoption=False,islocal=True,ischarging=True):
        self.learning_rate = LEARNING_RATE  # 1e-4
        self.gamma = GAMMA
        self.start_o_epsilon = START_EPSILON
        self.final_o_epsilon = FINAL_EPSILON
        self.epsilon_o_steps = EPSILON_DECAY_STEPS
        self.traj_memory = [TrajReplayMemory(REPLAY_BUFFER_SIZE) for _ in range(24)] # 24 hours
        self.option_batch_size = BATCH_SIZE
        self.clipping_value = CLIPPING_VALUE
        self.input_dim = INPUT_DIM
        self.option_dim = OPTION_DIM
        self.device = device
        self.path = F_AGENT_SAVE_PATH
        self.state_feature_constructor = FeatureConstructor()
        self.decayed_o_epsilon = self.start_o_epsilon
        # init option network
        self.option_network = F_Network()  # output a policy
        self.option_target_network = Target_F_Network()
        self.option_optimizer = torch.optim.Adam(self.option_network.parameters(), lr=self.learning_rate)
        self.option_lr_scheduler = StepLR(optimizer=self.option_optimizer, step_size=1000, gamma=0.99)
        self.option_train_step = 0
        self.option_network.to(self.device)
        self.option_target_network.to(self.device)

        self.record_list = []
        self.global_state_dict = defaultdict()
        self.time_interval = int(0)
        self.global_state_capacity = 5*1440 # we store 5 days' global states to fit replay buffer size.
        self.with_option = isoption
        self.with_charging = ischarging
        self.local_matching = islocal
        self.hex_diffusion = hex_diffusion
        self.option_queue = deque()
        self.init_func = None
        self.term_func = None
        self.init_dist = 0.05
        self.term_dist = 0.05
        self.upper_threshold = 1e5
        self.lower_threshold = 1e5


    def get_f_value(self, hex_ids):
        hex_diffusions = [np.tile(self.hex_diffusion[int(hex_id)], (1, 1, 1)) for hex_id in hex_ids]  # state[1] is hex_id
        return self.option_network.forward(torch.from_numpy(np.array(hex_diffusions)).to(dtype=torch.float32, device=self.device))

    def is_initial(self,state):
        if self.init_func is None:
            return True
        else:
            f_value = self.get_f_value(state[1])
            return self.init_func(f_value)

    def is_terminal(self,state):
        if self.term_func is None:
            return True
        else:
            f_value = self.get_f_value(state[1])
            return self.term_func(f_value)


    def train_option_network(self,hr):
        self.option_train_step += 1
        if len(self.traj_memory[hr]) < self.option_batch_size:
            print('batches in option replay buffer is {}'.format(len(self.traj_memory[hr])))
            return

        transitions = self.traj_memory[hr].sample(self.option_batch_size)
        traj_batch = self.traj_memory[hr].Transition(*zip(*transitions))

        f_values = torch.from_numpy(np.array(self.get_f_value([cur_hex for cur_hex in traj_batch.current_hex]))).to(dtype=torch.int64, device=self.device)
        # add a mask
        f_values_ = torch.from_numpy(np.array(self.get_f_value([next_hex for next_hex in traj_batch.next_hex]))).to(dtype=torch.int64, device=self.device)
        eta = 1.0 # lagrangian multiplier
        loss = 0.5 * F.mse_loss(f_values_ - f_values) + eta*((f_values_.pow(2) - 1)*(f_values.pow(2)-1) + f_values_.pow(2)*f_values.pow(2))
        self.option_optimizer.zero_grad()
        loss.backward()
        self.option_optimizer.step()
        self.option_lr_scheduler.step()

    # def sample_f_value(self, f_values,init_percentile, term_percentile):
    #
    #     f_sort = np.sort(f_values)
    #     init_threshold = f_sort[int(len(self.option_batch_size)*init_percentile)]
    #     term_threshold = f_sort[int(len(self.option_batch_size)*term_percentile)]
    #     return init_threshold, term_threshold

    def add_hex_pair(self,hour,current_hex,next_hex):
        self.traj_memory[int(hour)].push(current_hex,next_hex)

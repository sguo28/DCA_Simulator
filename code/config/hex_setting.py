## Configuration setting

### data paths
import os

db_dir = "../data/db.sqlite3"
DB_HOST_PATH = ''.join(['sqlite:///', db_dir])
OSRM_HOSTPORT = os.getenv("OSRM_HOSTPORT", "localhost:5000")
DEFAULT_LOG_DIR = "../logs/tmp"
DATA_DIR = "../data"

CS_SHP_PATH = '../data/NYC_shapefiles/cs_snap_unique.shp'  # '../data/NYC_shapefiles/cs_snap_lonlat.shp'
HEX_SHP_PATH = '../data/NYC_shapefiles/snapped_clustered_hex.shp'  # '../data/NYC_shapefiles/intersected_hex.shp'  #
TRIP_FILE = '../data/trip_od_hex.csv'
TRAVEL_TIME_FILE = '../data/trip_time_od_hex.csv'
CS_DATA_PATH = "../data/top_used_evse.csv" # updated 6/16/2022 # '../data/evcs_snap_unique.csv'
CMS_HOP_DIST_PATH = "../data/evcs_hop_distance.pkl"
ALL_HEX_ROUTE_FILE = '../data/parsed_routes.pkl'
HEX_ROUTE_FILE = '../data/hex_routes.pkl'  # the route between necessary pair of hex zone
MODEL_SAVE_PATH = 'logs/dqn_model'  # 'logs/dqn_model/dqn.pkl'
CNN_SAVE_PATH = 'logs/test/cnn_dqn_model'  # 'logs/dqn_model/dqn.pkl'
HRL_SAVE_PATH = 'logs/test/hrl_dqn_model'  # 'logs/dqn_model/dqn.pkl'
HEX_DIFFUSION_PATH = '../data/hex_diffusion.pkl' # return a list of diffusion by querying with hex_id.

H_AGENT_SAVE_PATH = 'logs/test/h_agent/'
F_AGENT_SAVE_PATH = 'logs/test/f_agent/'
OPTION_DQN_SAVE_PATH = 'logs/test/agent_with_option/'
TERMINAL_STATE_SAVE_PATH  = 'logs/test/terminal_states/'
### state-action space dimension
NUM_REACHABLE_HEX =  1347
NUM_NEAREST_CS = 5
DIM_OF_CHARGING = NUM_NEAREST_CS
LEVEL_OF_SOC = 11

### Attack model: EV and EVCS:
INIT_VEH_I = 0
INIT_EVCS_I = 0.1
DELAYED_PENALTY_TIME = 10 # 20 min more charging time due to delay. to be tuned.
DELAYED_PENALTY_TIME_STD = 2.5 # min, easily learned from historical charging data.
ATTACK_TYPE = 0 # -1 if we don't consider cyberattack; 0 is delaying charging, 1 is denial-of-service.
EVCS_EV_TRANSMISSION_RATE = 0.05
EV_EVCS_TRANSMISSION_RATE = EVCS_EV_TRANSMISSION_RATE
CMS_TRANSMISSION_RATE = 0.01
CMS_EVCS_TRANSMISSION_RATE = 0.01 # after an attack transmitted through all communication layers, the probabability of a sucessful attack.
# TIME_FOR_DETECTION = 1e9 # unit: sec, we don't consider detection as it is based on detection algorithm.
TIME_FOR_INSPECTION = 30 # unit: min (following time step)
TIME_FOR_REPAIR = 30 # unit: min
TRAINED_ATTACK_DETECTION_MODEL_PATH= lambda x: f"../data/redo_trained_isolated_forest_model_{x}.sav" # 0.05 is suggested in reference.
### Vehicle settings for simulator

# N_VEHICLES = int(1500)
# N_DUMMY_VEHICLES = int(0)
# N_DQN_VEHICLES = int(1500)
# DEMAND_SCALE = 0.25 #0.1 for high demand and 0.08 for low demand
MAX_WAIT_TIME = 600  #use 600 for undersupply and 300 for oversupply
MAX_MATCH_TIME = 1200 #use 480
ALL_SUPERCHARGING = True  # mask level 2 type to super-charging.
MAX_CHARGING_WAIT_TIME = 90*60 # one hour, unit: sec
QUEUE_LENGTH_THRESHOLD = 10  # 5 * 40 min if DC-fast
QUICK_END_CHARGE_PENALTY  = 50  # equivalent to weighted summation of 200 min staying at charging station (0.5*200min) + charge cost (5*$12)

### config for training DQN with CNN
LEARNING_RATE = 0.005# 1e-4

REPLAY_BUFFER_SIZE = int(100000)
BATCH_SIZE = 256
FINAL_EPSILON = 0.05
START_EPSILON = 1
EPSILON_DECAY_STEPS = int(10*1440)# 250*1440 #2e5
CLIPPING_VALUE = 2.0
RELOCATION_DIM = 7
CHARGING_DIM = 0
OPTION_DIM = 0
MAX_OPTION=3
CUDA=0
INPUT_DIM = 2+1 #2 + 1
OUTPUT_DIM = RELOCATION_DIM + CHARGING_DIM
DQN_OUTPUT_DIM = RELOCATION_DIM + CHARGING_DIM + OPTION_DIM*MAX_OPTION
PER_TICK_DISCOUNT_FACTOR = 0.99
GAMMA = PER_TICK_DISCOUNT_FACTOR

USE_RANDOM=0
LIMIT_ONE_NET=0

### config for training DQN with HRL
HRL_CHARGING_DIM = 12
HRL_RELOCATION_DIM = 7
HRL_OUTPUT_DIM = HRL_RELOCATION_DIM + HRL_CHARGING_DIM

STORE_TRANSITION_CYCLE = 60 * 50  # dump transitions to DQN every 5 min
TRAINING_CYCLE = 1 * 60  # min
UPDATE_CYCLE = 100*60  # update per 1000 tick
SAVING_CYCLE = 60 * 24 * 3 # save per day
DQN_RESUME = False
CNN_RESUME = False
HRL_RESUME = False

### config for CNN module

HEX_INPUT_CHANNELS = 1
HEX_OUTPUT_CHANNELS = 3
HEX_KERNEL_SIZE = 1
HEX_STRIDE = 4

NUM_BATCHES = 1
NUM_CHANNELS = 3
MAP_WIDTH = 46
MAP_HEIGHT = 54

# MIN_LON = -74.0387
# MIN_LAT = 40.5623  # this is the LAT in ODD COLUMN
# DELTA_LON = 0.074
# DELTA_LAT = 0.066

### config for agents

HIGH_SOC_THRESHOLD = 0.5
INIT_SOC_STD = 0.5
LOW_SOC_THRESHOLD = 0.2
OFF_DURATION = 20 * 60  # 20min
SPEED = 5  # m/s
IDLE_DURATION = 5  # min
TARGET_SOC = 0.78
TARGET_SOC_STD = 0.02
TOLERANCE_FOR_QUEUING = 1 # can wait if 2 EVs are queued.

#### PARAMETER FOR REWARD CALCULATION
BETA_EARNING = 3
BETA_CHARGE_COST = 1
BETA_RANGE_ANXIETY = 0
SOC_PENALTY = 100
BETA_DIST = 0.1
BETA_TIME = 0.1 / 60

### Time config for simulator

GLOBAL_STATE_UPDATE_CYCLE = 60 * 5
TIMESTEP = 60  # sec
ENTERING_TIME_BUFFER = 0 # 60*60*1 = 1 hour

NUM_SEEDS = 3
REJECT_TIME = 30 * 60  # 30 min
START_OFFSET = int(0)  # simulation start datetime offset (days)"
SIM_DAYS = int(28)  # simulation days: 504000 ticks
START_TIME = int(0)  # int(1464753600)  #  + 3600 * 5simulation start datetime (unixtime)")
N_EPISODE=int(1) # number of episode

#### Speed up the simulation
SIM_ACCELERATOR = float(1)  # accelerate consumption speed
CHARGE_ACCELERATOR = float(1)  # todo:accelerate charging speed

### Unit convert
MILE_PER_METER = 0.000621371
SEC_PER_MIN = 60
MIN_PER_HOUR = 60

### Customer
WAIT_COST = 0.5/60 # usd per sec
TOTAL_COST_PER_MILE = 0.4815  # operation and maintenance ,unit:USD
DRIVER_TIME_VALUE = 30 / 3600  # $30/hr
SERVICE_PRICE_PER_MILE = 1.103 #usd per mile
SERVICE_PRICE_PER_MIN = 0.502/60 # usd per sec


## Setting for test scenarios
TEST_START=15
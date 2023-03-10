import numpy as np
from config.hex_setting import NUM_REACHABLE_HEX
class FeatureConstructor:
    """
    transform tick, SOC, hex_id to representative format
    tick: hour of day, day of week
    SOC: 0 - 1
    hex_id: one-hot encoding and 1 by N vector

    one-hot encoding for hexagon needs additional padding and transformation: https://github.com/ai4iacts/hexagdly
    """

    def __init__(self):
        self.hex_vector = [0] * NUM_REACHABLE_HEX
        # self.hex_coord = None

    def construct_state_features(self,state):
        """
        To construct state features:
        time feature: sin and cos: time of day  # deleted --day of week--
        soc: discretized soc to 10 levels
        hex_id: hold for one-hot encoding inside DQN network
        :param state:
        :return:
        """
        tick, hex_id, soc = state
        state_features = []
        time_features = self.construct_time_features(tick)
        # hex_one_hot = self.construct_one_hot_encoding_map(hex_id)
        soc_level = np.digitize(soc,[0,0.1,0.2,0.25,0.3,0.4,0.5,0.65,0.8,1.0]) # level = 0 only means SOC<0 (never >1)
        #  [0.8-1, 0.65-0.8, 0.5-0.65, 0.4-0.5, 0.3-0.4, 0.25-0.3, 0.2-0.25])
        state_features += [hex_id]
        state_features += [soc_level]
        state_features += time_features
        return state_features

    def construct_map_vector(self, hex_id):
        """
        todo: try to use function in torch to construct one-hot encoding location.
        :param hex_id:
        :return:
        """
        hex_vector = self.hex_vector
        hex_vector[hex_id] = 1
        return hex_vector

    # def construct_one_hot_encoding_map(self, hex_id):
    #     initial_map = np.zeros((NUM_BATCHES, NUM_CHANNELS, MAP_WIDTH, MAP_HEIGHT))
    #     for ib, batch in enumerate(initial_map):
    #         for ic, channel in enumerate(batch):
    #             location_flag = self.hex_dot_location(hex_id)
    #             initial_map[ib,ic,:,:] += location_flag
    #     return initial_map

    # def hex_dot_location(self, hex_id, value=1.0):
    #     # d = np.zeros((HEX_MAP_WIDTH,HEX_MAP_HEIGHT))
    #     # lon, lat = self.hex_coord[hex_id]
    #     # loc_x, loc_y = self.convert_lonlat2xy(lon,lat)
    #     # d[loc_x, loc_y] = value
    #     # return d.transpose()

    def construct_time_features(self, tick):
        hourofday = tick % (60 * 60 * 24) / 24.0 * 2 * np.pi
        # dayofweek = tick % (60 * 60 * 24 * 7) / 7.0 * 2 * np.pi
        return [np.sin(hourofday), np.cos(hourofday)]  #, np.sin(dayofweek), np.cos(dayofweek)]

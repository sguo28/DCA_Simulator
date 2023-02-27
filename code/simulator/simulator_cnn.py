import pickle
from scipy.spatial import cKDTree
import random
import geopandas as gpd
import pandas as pd
import numpy as np
np.set_printoptions(precision=5)
from scipy.special import softmax
from collections import deque
import warnings
warnings.simplefilter("ignore")
from queue import PriorityQueue
from config.hex_setting import NUM_REACHABLE_HEX, NUM_NEAREST_CS, ENTERING_TIME_BUFFER, HEX_ROUTE_FILE, \
    ALL_HEX_ROUTE_FILE, STORE_TRANSITION_CYCLE, SIM_DAYS, ALL_SUPERCHARGING, MAP_WIDTH, MAP_HEIGHT, NUM_CHANNELS, HEX_DIFFUSION_PATH,TERMINAL_STATE_SAVE_PATH, \
    OPTION_DIM, INIT_VEH_I, INIT_EVCS_I, CMS_HOP_DIST_PATH, TRAINED_ATTACK_DETECTION_MODEL_PATH,\
    CMS_TRANSMISSION_RATE, TARGET_SOC, TARGET_SOC_STD, HIGH_SOC_THRESHOLD, INIT_SOC_STD,LOW_SOC_THRESHOLD
from novelties import agent_codes, status_codes, vectorized_random_choice, largest_indices
from .models.charging_pile.charging_pile import ChargingStation
from .models.vehicle.vehicle import Vehicle
from .models.vehicle.vehicle_state import VehicleState
from .models.zone.hex_zone import hex_zone
from .models.zone.matching_zone_sequential import matching_zone
from .models.attack_detection.anomaly_detection import AnomalyDetection


class Simulator(object):
    def __init__(self, start_time, timestep, config):
        self.reset_time(start_time, timestep)
        self.last_vehicle_id = 1
        self.vehicle_queue = PriorityQueue() # sorted by arrival time
        self.route_cache = {}
        self.current_dummyV = 0
        self.current_dqnV = 0
        # containers as dictionaries
        self.config = config
        self.n_vehicles = self.config["N_VEHICLES"]
        self.match_zone_collection = []
        self.hex_zone_collection = {}
        # DQN for getting actions and dumping transitions
        self.all_transitions = []
        self.prime_transitions = []
        self.f_transitions=[]
        self.fo_transitions=[]
        self.charging_station_collections = []
        self.trajectory_transitions=[]
        self.snapped_hex_coords_list = [[0,0]]
        self.num_match = 0
        self.total_num_arrivals = 0
        self.total_num_removed_pass = 0
        self.total_num_served_pass = 0
        self.total_num_longwait_pass = 0
        self.total_idled_vehicles = 0
        self.global_state_tensor = np.zeros((NUM_CHANNELS, MAP_HEIGHT, MAP_WIDTH))  # t, c, w, h
        self.global_state_buffer=dict()
        self.demand_scale = self.config["DEMAND_SCALE"]
        self.with_charging = config["IS_CHARGING"]
        self.local_matching = config["IS_LOCAL"]
        self.with_option = config["IS_OPTION"]
        self.use_detector = self.config["USE_DETECTOR"]
        self.N_EVSE_AMPLIFICATION = self.config["N_EVSE_AMPLIFICATION"]
        self.detection_technique = self.config["DETECTION_TECHNIQUE"]
        self.sensitivity_level = self.config["SENSITIVITY_LEVEL"]
        self.start_day_of_attack = self.config["START_DAY_OF_ATTACK"]
        self.detection_algorithm = AnomalyDetection(method=self.detection_technique,sensitivity_level=self.sensitivity_level)
        self.hex_diffusions = None
        self.charging_profile = [] # init charging profile data.
        self.historical_charging_profile = [] # init historical charging profile data.

    def init(self, file_hex, trip_file, cs_data_file, travel_time_file, n_nearest=NUM_NEAREST_CS):
        '''
        :param file_hex:
        :param file_charging:
        :param trip_file:
        :param travel_time_file:
        :param n_nearest: we consider top *5* nearest charging stations.
        :return:
        '''
        #read demand
        filename = "../data/daily_trips/16.csv" # only for reading the scale.
        n = sum(1 for _ in open(filename)) - 1  # number of records in file (excludes header)
        s = int(n*self.demand_scale)  # desired sample size, 0.3 for 500 vehicles, 0.6 for 2000 vehicles.
        random.seed(10)
        skip = sorted(random.sample(range(1, n + 1), n - s))  # the 0-indexed header will not be included in the skip list

        np.random.seed(0)

        self.attack_type = -1 # inititate as no attack. # self.config["ATTACK_TYPE"]
        self.penalty_charging_duration = self.config["PENALTY_CHARGING_DURATION"]
        self.false_alarm_rate = self.config["FALSE_ALARM_RATE"]
        self.EVCS_EV_TRANSMISSION_RATE = self.config["EVCS_EV_TRANSMISSION_RATE"]
        self.EV_EVCS_TRANSMISSION_RATE = self.EVCS_EV_TRANSMISSION_RATE
        self.CMS_EVCS_TRANSMISSION_RATE = self.config["CMS_EVCS_TRANSMISSION_RATE"]
        self.TIME_FOR_REPAIR = self.config["TIME_FOR_REPAIR"]

        local_rand_seed=np.random.randint(2,7) # only consider a weekday.
        alltrips = pd.read_csv(f"../data/daily_trips/{local_rand_seed}.csv",skiprows=skip, dtype=int).groupby('origin_hid')

        # import hex file
        df = gpd.read_file(file_hex)  # tagged_cluster_hex
        self.hex_kdtree = cKDTree(df[["snap_lon", "snap_lat"]])
        # import charging station file based on top-used EVCSs.:
        cs_df = pd.read_csv(cs_data_file)
        # cs_df["num"] = (cs_df["num"]*self.N_EVSE_AMPLIFICATION).astype(int) # EVSE AMPLIFICATION: default is 1, only for tuning.

        self.charging_kdtree = cKDTree(cs_df[["snap_lon", "snap_lat"]])

        # import hop distance of the communication network:
        with open(CMS_HOP_DIST_PATH, 'rb') as f1:
            self.hop_dist = np.load(f1)
            # self.hop_dist = np.zeros((cs_df["num"].sum(),cs_df["num"].sum()))

        print("loading routing information...")
        with open(ALL_HEX_ROUTE_FILE, 'rb') as f2:
            self.hex_routes=pickle.load(f2)

        df['cluster_la'] = 0 # set a mask to combine to one matching zone.
        matchzones = np.unique(df['cluster_la'])

        hex_ids = df.index.tolist()
        print(f'{len(hex_ids)} hexagons loaded...')

        hex_coords = df[['snap_lon', 'snap_lat']].to_numpy()  # coord
        self.snapped_hex_coords_list = df[['snap_lon', 'snap_lat']].values.tolist()

        hex_to_match = df['cluster_la'].to_numpy() # corresponded match zone id
        row_col_coords = df[['row_id', 'col_id']].to_numpy()

        print("loading demand and travel time by intervals...")
        demand = self.get_processed_trip(trip_file)
        self.travel_time = self.get_travel_time(travel_time_file)

        self.hex_diffusions = self.get_hex_diffusions(HEX_DIFFUSION_PATH, row_col_coords)

        # preprocessed od time mat from OSRM engine
        od_time = 600*np.ones([NUM_REACHABLE_HEX, NUM_REACHABLE_HEX])
        for (o, d) in self.hex_routes.keys():
            od_time[o, d] = sum(self.hex_routes[(o, d)]['travel_time'])
            if od_time[o,d]==0:
                od_time[o,d]=60
            # print(o,d,self.hex_routes[(o,d)])
            # self.hex_routes[(o,d)]['travel_time']=[60]
        od_time[np.isnan(od_time)] = 1e8  # set a large enough number
        self.od_time=od_time

        epoch_length = 60 * 24 * SIM_DAYS  # this is the total number of ticks set for simulation, change this value.'
        t_unit = 60  # number of time steps per hour (60 minute per hour)

        # we initiaze the set of hexagonal zones first
        maxdemand = 0
        total_demand = 0
        charging_coords = cs_df[['snap_lon', 'snap_lat']].values.tolist()
        _,charging_hexes=self.hex_kdtree.query(charging_coords)
        self.charging_hexes=[charging_hexes[ix] for ix,num in enumerate(cs_df["num"].to_numpy()) for _ in range(int(num))]
        self.all_neighbors=np.zeros((NUM_REACHABLE_HEX,7)).astype(int) #store the index here

        # record high demand hexagons per hour.
        highdemand_hex_ids_per_hour = []
        for t in range(24):
            highdemand_hex_ids_per_hour.append(np.argsort(np.sum(demand[t, :, :], axis=1))[-500:])

        for h_idx, coords, row_col_coord, match_id in zip(hex_ids, hex_coords, row_col_coords, hex_to_match):
            _,neighbors= self.hex_kdtree.query(coords,k=7) # return the closet 6 locations + itself
            neighbors=list(neighbors)
            self.all_neighbors[h_idx,:]=neighbors
            if h_idx!=neighbors[0]:
                print('error encountered for h_idx',h_idx,neighbors)
                new_neighbors=[h_idx]+[i for i in neighbors if i!=h_idx]
                self.all_neighbors[h_idx,:]=new_neighbors
                neighbors=new_neighbors
                print('corrected for h_idx', h_idx, new_neighbors)

            _, charging_idx = self.charging_kdtree.query(coords, k=n_nearest)  # charging station id
            maxdemand = max(maxdemand,sum(demand[0,h_idx,:])/60)
            try:
                demand_df=alltrips.get_group(int(h_idx))
            except KeyError:
                demand_df=pd.DataFrame({'tick':[]})
            total_demand += sum(demand[0, h_idx, :])

            highdemand_hexs = []
            for t in range(24):
                target_hex_id = np.argsort(self.travel_time[t,h_idx, highdemand_hex_ids_per_hour[t]])[-1]
                highdemand_hexs.append(target_hex_id)
            self.hex_zone_collection[h_idx] = hex_zone(h_idx, coords, row_col_coord, hex_coords, match_id, neighbors,
                                                       charging_idx, charging_hexes, charging_coords,
                                                       demand[:, h_idx, :], self.travel_time[:, h_idx, :],
                                                       t_unit, epoch_length,local_rand_seed,demand_df, highdemand_hexs)

        hex_collects = []
        for m_idx in matchzones:
            h_ids = df[df['cluster_la'] == m_idx].index.tolist()
            hex_collects.append([self.hex_zone_collection[hid] for hid in h_ids])

        # initialize the matching zones
        self.match_to_hex = {}  # a local map of hexagons to matching zones
        [self.match_zone_collection.append(matching_zone(idx, hexs, od_time)) for idx, hexs in
         zip(matchzones, hex_collects)]
        print('matching zone initialized')

        for idx, hexs in zip(matchzones, hex_collects):
            self.match_zone_collection[idx].get_info()
            self.match_to_hex[idx] = hexs  # a local container

        # init charging station
        self.init_charging_station(cs_df)

        self.generate_veh_q()
        # # load attack detection model.
        # self.attack_detection_model = pickle.load(open(TRAINED_ATTACK_DETECTION_MODEL_PATH(self.false_alarm_rate),'rb'))

        print('initialize vehicle queue complete')

        # vehicle_hex_ids=[random.choice(hex_ids) for _ in range(N_VEHICLES)] # randomly initialize location.
        # vehicle_hex_ids = np.random.choice(hex_ids,self.n_vehicles) # randomly initialize location.
        # status = np.random.choice([0,1],self.n_vehicles,p=[1-INIT_VEH_I,INIT_VEH_I]) # initiate infectious EVs
        # vehicle_ids = np.arange(self.last_vehicle_id, self.last_vehicle_id + self.n_vehicles)
        # self.last_vehicle_id += self.n_vehicles # some vehicle takes shift, new vehicle need to initial upon that.
        # entering_time = np.random.uniform(self.__t, self.__t + ENTERING_TIME_BUFFER, self.n_vehicles)
        # [self.vehicle_queue.put((t,vid,hid,status)) for t,vid,hid,status in zip(entering_time, vehicle_ids, vehicle_hex_ids, status)]


    def enter_market(self):
        while not self.vehicle_queue.empty():
            t, vid, hid, SOC, status = self.vehicle_queue.get() # pop the earliest arrival time for each loop
            if self.__t >= t:
                self.populate_vehicle(vid, hid, SOC, status)
            else:
                self.vehicle_queue.put((t, vid, hid, SOC, status)) # if not arrived, put it back
                break

    def populate_vehicle(self, vehicle_id, vehicle_hex_id, SOC, status):
        """
        vehicle_id: unique ID
        vehicle_hex_id: origin location indexed by hexagon
        status: susceptible or infectious (we consider SI process)
        """
        agent_type = agent_codes.dqn_agent

        location = (self.hex_zone_collection[vehicle_hex_id].lon, self.hex_zone_collection[vehicle_hex_id].lat)  # update its coordinate with the centroid of the hexagon

        # append this new available vehicle to the hexagon zone
        self.hex_zone_collection[vehicle_hex_id].add_veh(Vehicle(
            VehicleState(vehicle_id, location, vehicle_hex_id, SOC, agent_type, status),

            self.with_option,self.local_matching,self.with_charging,self.__t))

    def init_charging_station(self, charging_stations_data):
        # no attack at the first week. Attack starts at week 2 (check self.attach_attack_status)
        init_status = np.zeros(charging_stations_data.shape[0])
        for idx, sp in charging_stations_data.iterrows():
            hex_id, num, ilon, ilat, evcs_id = sp.tolist() # we consider each EVSE as an individual, but under the same EVCS (id)
            hex = self.hex_zone_collection[int(hex_id)]
            hex.n_charges += num
            self.charging_station_collections.append(
                ChargingStation(n_l2=0, n_dcfast=1, lat=float(ilat), lon=float(ilon),
                                hex_id=int(hex_id), hex=hex, row_col_coord=(hex.row_id, hex.col_id),
                                status=init_status[idx], attack_type=self.attack_type, id=idx,
                                tipping_tick=self.__t,
                                penalty_charging_duration=self.penalty_charging_duration))

    def get_hex_diffusions(self, diff_file, row_col_coords):
        with open(diff_file, "rb") as f:
            hex_diffusions = pickle.load(f)  # with key: hex_id
        mat = np.zeros((NUM_REACHABLE_HEX, MAP_HEIGHT,MAP_WIDTH))

        for key_id, diffusions in hex_diffusions.items():
            for hex_id, diff in enumerate(diffusions):
                row_id, col_id = row_col_coords[hex_id]
                mat[key_id, row_id, col_id] = diff
        return mat

    def step(self):  # we use parallel update to call the step function.
        '''
        1. conduct the matching for each matching zone
        2. Update passenger status
        3. Update vehicle status
        4. Dispatch vehicles
        5. Generate new passengers
        :return:
        '''
        # print('match time:', time.time()-t1)
        # update vehicle locations
        # self.update_vehicles()  # push routes into vehicles
        # update passenger status

        # t1 = time.time()
        [mc.update_passengers() for mc in self.match_zone_collection]
        # print('update passengers time:', time.time()-t1)
        # update the demand for each matching zone
        # t1 = time.time()
        [mc.async_demand_gen(self.__t) for mc in self.match_zone_collection]
        # print('async demand time:', time.time()-t1)
        # t1 = time.time()
        # dispatched vehicles which have been attached dispatch actions.
        target_hex_ids = self.get_target_hex_ids() # list of multinomial prob generator.
        target_cs_hex_ids, target_cs_ids = self.get_target_charging_station_ids(top_k=10)
        [mc.matchingzone_dispatch(self.__t, target_hex_ids,target_cs_hex_ids, target_cs_ids) for mc in self.match_zone_collection]
        # print('dispatch time:', time.time()-t1)
        # t1 = time.time()
        [mc.matchingzone_match(self.__t,self.charging_station_collections) for mc in self.match_zone_collection]  # force this to complete: matching idled vehicles to requests.
        # print('match time:', time.time()-t1)

        self.update_vehicles()  # push routes into vehicles, there may be some vehicles need charge in last step.
        # print('update vehicle time:', time.time() - t1)
        self.enter_market()
        # update charging_station status (Cyberattack component)
        # t1 = time.time()
        if self.attack_type!=-1: # if there is an attack.
            self.update_EVCS_EV_status() # if we consider cyberattack. (self.attack_type =-1: no cyberattack)
        # print('update EVCS status time:', time.time() - t1)

        # update charging stations...
        # t1 = time.time()
        [cs.step(self.__dt, self.__t) for cs in self.charging_station_collections]
        self.vehicle_step_update(self.__dt, self.__t)  # interpolate routes and update vehicle status
        self.download_match_zone_metrics()

    def update_EVCS_EV_status(self,usb_infection_rate=0.05):
        '''
        The status update under cyberattack
        EV: Susceptible-Infectious
        EVCS: Susceptible - Infectious - Removed
        Functions:
        (1) calculate CMS_based probability and transmission
        (2) calculate EV_based probability and transmission
        '''
        infectious_flag = np.random.binomial(1,p=usb_infection_rate,size=len(self.charging_station_collections)) # every 30 min compromising with probability.
        if self.__t%1800==0:
            for cs,flag in zip(self.charging_station_collections,infectious_flag):
                if flag and cs.status==0: # EVCS changes from susceptible to infectious
                    cs.status=1 # update status to infectious
                    cs.tipping_tick = self.__t # record the most recent tick of a status-change.
                    self.attack_counter[cs.id] += 1 # update the detection counter

        # next conduct attack detection using the online charging profile as testing data (tuples of start_SOC	charging_duration	hour)
        if self.use_detector and self.penalty_charging_duration>0: # only undergo the SIRS process if we consider cyberattack.
            if int(self.__t % 3600) == 0:  # record charging profile every 60 minute.
                evcs_id_outlier, evcs_id_inlier = self.get_list_of_charging_station_to_detect()
                for cs_id in range(len(self.charging_station_collections)):
                    if cs_id in evcs_id_outlier:
                        selected_cs = self.charging_station_collections[int(cs_id)]
                        if selected_cs.status == 1:
                            selected_cs.status = 2
                            selected_cs.time_to_recover = self.TIME_FOR_REPAIR
                            selected_cs.tipping_tick = self.__t
                            self.detection_counter[1]+=1 # a successful detection
                        elif selected_cs.status == 0:
                            self.detection_counter[0]+=1 # false alarm alert
                    elif cs_id in evcs_id_inlier:
                        selected_cs = self.charging_station_collections[int(cs_id)]
                        if selected_cs.status == 1:
                            self.detection_counter[2]+=1 # fail to detect the anomaly.
                        elif selected_cs.status == 0:
                            self.detection_counter[3]+=1 # true negative
                    else:
                        # cs_id is removed for repair...
                        continue

            for cs in self.charging_station_collections:
                if cs.status == 2:
                    cs.time_to_recover-=1
                    if cs.time_to_recover<=0:
                        cs.status = 0
                        cs.tipping_tick = self.__t
                        [p.remove_charging_samples() for p in cs.piles]  # remove the historical operation data.

    def get_list_of_charging_station_to_detect(self):
        '''
        :params charging_profile: a sample recording start_SOC, charging_duration, and hour
        :return: list of detected charging station id.
        '''
        # collect the charging profile of each EVCS
        for cs in self.charging_station_collections:
            for p in cs.piles:
                if p.charging_samples and cs.status != 2:  # not removed..
                        [self.charging_profile.append([cs.id] + sample_i) for sample_i in
                         p.charging_samples]  # assigned charging duration, starting_SOC, starting hour.

        if len(self.charging_profile)==0:
            return {},{}
        else:
            # charging_profile = np.array(self.charging_profile) # ID, start SoC, Charging duration, and hour of day
            # detected_outliers = self.attack_detection_model.predict(charging_profile[:,1:])
            print(f"line 377, start detecting.... with {len(self.charging_profile)} samples")
            evcs_status = np.array([cs.status for cs in self.charging_station_collections])
            evcs_id_outlier, evcs_id_inlier, Accuracy = self.detection_algorithm.predict(self.charging_profile, evcs_status)

            return evcs_id_outlier, evcs_id_inlier

    def calculate_cms_based_prob(self):
        '''
        calculate propagation probability according to CMS communication network topology initiated as self.CMS_network.
        :return: an array of infected flag of length N_evcs.
        '''
        binary_infection_indicator = np.array([cs.status==1 for cs in self.charging_station_collections]) # first summarized the infected charging station
        # generate a matrix with diag =0 and others =1. (for the purpose of excluding the charging station itself)
        exclude_mat = np.ones_like(self.hop_dist) - np.identity(self.hop_dist.shape[0])
        probs_evcs_j_to_i = binary_infection_indicator * np.power(self.CMS_EVCS_TRANSMISSION_RATE, 2) * np.power(CMS_TRANSMISSION_RATE,self.hop_dist) * exclude_mat
        prob = 1 - np.prod(1-probs_evcs_j_to_i,axis=1)
        # then we consider EVCS -> CMS and CMS -> EVCS: np.power(CMS_EVCS_TRANSMISSION_RATE, 2)
        # and the transmission through CMS network for roaming service among charging station operators (np.power(CMS_TRANSMISSION_RATE, self.hop_dist[cs_id,]).
        # note that dist is the hop distance.
        return prob

    def calculate_ev_based_prob(self):
        '''
        calculate propagation probability between EV and EVCS.
        '''
        # if cp.assigned_vehicle.state.attack_status is infectious, the charging pile is likely to be infected.
        prob = [1- np.prod([1-self.EV_EVCS_TRANSMISSION_RATE*(cp.assigned_vehicle.state.attack_status==1) for cp in cs.piles if cp.occupied==True]) for cs in self.charging_station_collections]
        return np.array(prob)



    def update(self):
        self.__update_time()
        # print('demand gen time', time.time() - t1)

        # if self.__t % 3600 == 0:
        #     self.logger.info("Elapsed : {}".format(get_local_datetime(self.__t)))  # added origin UNIX time inside func.

        # print('Iteration {} completed, push time={:.3f}'.format(tick / 60, t_p_update))
        # finally identify new vehicles, and update location of existing vehicles
        # the results is a list of list of dictionaries.

    def download_match_zone_metrics(self):
        metrics = [m.get_metrics() for m in self.match_zone_collection]
        self.num_match = sum([item[0] for item in metrics])
        self.total_num_arrivals = sum([item[1] for item in metrics])
        self.total_num_longwait_pass = sum([item[2] for item in metrics])
        self.total_num_served_pass = sum([item[3] for item in metrics])
        self.total_idled_vehicles = sum([item[4] for item in metrics])
        # total_assigned=sum([item[5] for item in metrics])
        # total_v=sum([item[6] for item in metrics])
        # print(self.num_match,self.total_num_arrivals,self.total_num_served_pass,self.total_idled_vehicles,total_assigned,total_v)

    def update_vehicles(self):
        '''
        1. loop through all hexagones and update the vehicle status
        2. add veh to charging station
        3. do relocation: attach vehicle's action id
        :return:
        '''
        # add vehicles to charging stations and remove from the hex zone

        vehs_to_update = [veh for hex in self.hex_zone_collection.values() for veh in hex.vehicles.values()]
        # veh_hex_to_update = [veh for veh in vehs_to_update if veh.state.status == status_codes.V_CRUISING and veh.state.real_time_location]
        if vehs_to_update:
            _, real_time_hex_id = self.hex_kdtree.query([veh.state.real_time_location for veh in vehs_to_update]) # a cruising vehicle's real-time location can be tracked.

            for hex_id, veh in zip(real_time_hex_id,vehs_to_update):
                veh.state.current_hex = hex_id

        [vehicle.update_info(self.hex_zone_collection, self.hex_routes, self.snapped_hex_coords_list, self.__t) for vehicle in vehs_to_update]

        [self.charging_station_collections[vehicle.get_assigned_cs_id()].add_arrival_veh(vehicle) for vehicle in vehs_to_update if vehicle.state.status == status_codes.V_WAITPILE]

    def get_local_states(self):
        state_batches = []
        num_valid_relocations = []
        assigned_option_ids=[]
        empty_flag = True
        for hex in self.hex_zone_collection.values():
            for vehicle in hex.vehicles.values():
                if vehicle.state.agent_type == agent_codes.dqn_agent and vehicle.state.status == status_codes.V_IDLE:
                    state_batches.append(vehicle.dump_states(self.__t))  # (tick, hex_id, SOC)
                    num_valid_relocations.append(len([0] + self.hex_zone_collection[vehicle.get_hex_id()].neighbor_hex_id))
                    assigned_option_ids.append(vehicle.assigned_option)
        if state_batches: # if state batches is not empty
            empty_flag = False
        return state_batches, num_valid_relocations,assigned_option_ids, empty_flag

    def get_local_infos(self):
        state_batches = []
        num_valid_relocations = []
        excluded_od_pairs = []
        for hex in self.hex_zone_collection.values():
            for vehicle in hex.vehicles.values():
                if vehicle.state.agent_type == agent_codes.dqn_agent and vehicle.state.status == status_codes.V_IDLE:
                    state_batches.append(vehicle.dump_states(self.__t))  # (tick, hex_id, SOC)
                    num_valid_relocations.append(len([0] + self.hex_zone_collection[vehicle.get_hex_id()].neighbor_hex_id))
                if vehicle.require_online_query:
                    excluded_od_pairs += [vehicle.non_included_od_pair]
        return state_batches, num_valid_relocations, excluded_od_pairs



    def get_global_state(self):
        self.store_global_states()
        # global_state_slice = np.zeros((NUM_CHANNELS, MAP_WIDTH, MAP_HEIGHT))  # t, c, w, h
        # global_state_slice[2] = 1e6  # queue length
        # for hex_i in self.hex_zone_collection.values():
        #     self.global_state_tensor[0:2, hex_i.x, hex_i.y] = [sum(hex_i.arrivals[-15:]), len(hex_i.vehicles.keys())]
        #
        # for cs in self.charging_station_collections:
        #     self.global_state_tensor[t, 2, cs.x_coord, cs.y_coord] += len(cs.queue) / self.hex_zone_collection[
        #         cs.hex_id].n_charges
        global_state_slice = self.global_state_buffer[self.__t // 60]

        return global_state_slice  # take the last slice that was stored at the start of this tick.

    # def get_utility_mat(self):
    #     '''
    #     :return: utility matrix of shape (n_hex, n_hex)
    #     '''
    #     sd_pattern = self.get_supply_demand_pattern() # get the latest state
    #     utility_mat = sd_pattern/self.travel_time
    #     return utility_mat

    def get_target_hex_ids(self,top_k=10):
        sd_pattern = self.get_supply_demand_pattern() # np.ones(NUM_REACHABLE_HEX)# a vector of length n_hex
        # add a noise:
        sd_pattern = sd_pattern + np.random.normal(0,1,len(sd_pattern))
        # print("sd_pattern",sd_pattern,np.sum(sd_pattern))
        # print("about time",np.min(self.travel_time[self.__t//(60*60)%24,:,:]), "hour",self.__t//(60*60)%24)
        util_mat = np.divide(sd_pattern,self.travel_time[self.__t//(60*60)%24,:,:]) # self.get_utility_mat() # demand/supply matrix # check how the supply is calculated.
        top_k_per_row = np.argpartition(util_mat, -top_k)[:,-top_k:] # add 0 because it originally returns list of list.
        # print("top_k_per_row",top_k_per_row,"util_mat",util_mat)
        idx_row = np.arange(util_mat.shape[0])
        soft_util_mat = softmax(util_mat[idx_row[:, None],top_k_per_row],axis=1) # softmax along the row after selecting top_k hexes
        # print(soft_util_mat[:2],"soft_util_mat.shape[:2]")
        row_ids = vectorized_random_choice(soft_util_mat.T, np.arange(soft_util_mat.shape[0]))
        target_hex_ids = top_k_per_row[idx_row,row_ids]
        return target_hex_ids

    def get_target_charging_station_ids(self,top_k=5):
        '''
        :param top_k: K charging stations with the highest utility
        :return: target_charging_station_ids and list of hex_ids
        function: process an utility matrix with |N_HEX| rows and |N_CS| columns.
        '''
        charging_pattern = self.get_charging_pattern()
        normal_operation_mask = np.array([cs.status==2 for cs in self.charging_station_collections])
        time_to_cs = self.od_time[:,self.charging_hexes]
        # time_to_cs[time_to_cs==0] = 0.1
        # print("shape of time_to_cs",time_to_cs/60, "charging_pattern",60*np.log(charging_pattern+1))

        util_mat = - 60*np.log(charging_pattern+1) - time_to_cs/60 - 120*normal_operation_mask

        top_k_per_row = np.argpartition(util_mat, -top_k)[:, -top_k:]  # add 0 because it originally returns list of list.

        idx_row = np.arange(util_mat.shape[0])
        soft_util_mat = softmax(util_mat[idx_row[:, None], top_k_per_row],
                                axis=1)  # softmax along the row after selecting top_k hexes

        row_ids = vectorized_random_choice(soft_util_mat.T, np.arange(soft_util_mat.shape[0]))
        target_cs_ids = top_k_per_row[idx_row, row_ids]
        target_hex_ids = [self.charging_hexes[i] for i in target_cs_ids]
        return target_hex_ids, target_cs_ids

    def get_charging_pattern(self):
        '''
        get average waiting time per charging station
        '''
        # charging_pattern = np.array([np.mean(cs.get_average_waiting_time()) for cs in self.charging_station_collections])
        charging_pattern = np.array([cs.get_queue_length() for cs in self.charging_station_collections])
        return charging_pattern



    def dump_global(self):
        global_dict=self.global_state_buffer
        return global_dict

    def store_global_states(self):
        """
        1st layer: # of request per hex_zone
        2nd layer: # of available vehicle per hex_zone
        3rd layer: # of queueing vehicle per hex_zone
        :return:
        """
        t=self.__t//60
        tmp_state=np.array(self.global_state_tensor)
        for hex_i in self.hex_zone_collection.values():
            tmp_state[0:2, hex_i.row_id, hex_i.col_id] = [sum(hex_i.arrivals[-15:]), len(hex_i.vehicles)]

        for cs in self.charging_station_collections:
            tmp_state[2, cs.row_id, cs.col_id] += len(cs.queue) / self.hex_zone_collection[
                cs.hex_id].n_charges
        self.global_state_buffer[t]=tmp_state # store the current state (per tick)

    def get_supply_demand_pattern(self):
        # for h in self.hex_zone_collection.values():
        #     for v in h.vehicles.values():
        #         print(f"check if we can get the vehicle id {v.id}")
        #         break
        #     break
        return np.array([sum(h.arrivals[-30:])/(1+len(h.vehicles.values())) for h in self.hex_zone_collection.values()]) # /(1+len(h.vehicles))

    def attach_actions_to_vehs(self, action_select, action_to_execute,opts=None,contd_options=None):
        veh_agents = [vehicle for hex in self.hex_zone_collection.values() for vehicle in hex.vehicles.values() if
                      vehicle.state.agent_type == agent_codes.dqn_agent and \
                      vehicle.state.status == status_codes.V_IDLE]
        if opts is None:
            for veh, a_sel,a_exe in zip(veh_agents, action_select,action_to_execute):
                veh.send_to_dispatching_pool(a_sel,a_exe) # and interpret the actions at hex_zone dispatching function.
        else:
            for veh, a_sel,a_exe, opt,ctd_opts in zip(veh_agents,action_select,action_to_execute,opts,contd_options):
                veh.send_to_dispatching_pool(a_sel,a_exe,opt,ctd_opts)

    def vehicle_step_update(self, timestep, tick):
        [m.matchingzone_update_vehicles(timestep, tick) for m in self.match_zone_collection]

    # def match_zone_step_wrapper(self, zone):
    #     '''
    #     This is a wrapper to be fed to the parallel pool in each iteration
    #     '''
    #     tick = self.__t - self.start_time
    #     t1 = time.time()
    #     zone.step(tick)  # call the step function for the matching zone
    #     return time.time() - t1

    def __update_time(self):
        self.__t += self.__dt

    def store_transitions_from_veh(self):
        """
        vehicle.dump_transition() returns a list of list. [[s,a,s_next,r]]
        """

        for hex in self.hex_zone_collection.values():
            for vehicle in hex.vehicles.values():
                self.all_transitions += vehicle.dump_transitions()

    def store_prime_action_from_veh(self):
        """
        vehicle.dump_transition() returns a list of list. [[s,a,s_next,r]]
        """

        for hex in self.hex_zone_collection.values():
            for vehicle in hex.vehicles.values():
                self.prime_transitions += vehicle.dump_prime_transitions()

    def store_f_action_from_veh(self):
        """
        vehicle.dump_transition() returns a list of list. [[s,a,s_next,r]]
        """
        for hex in self.hex_zone_collection.values():
            for vehicle in hex.vehicles.values():
                self.f_transitions += vehicle.dump_f_transitions()
                self.fo_transitions+=vehicle.dump_fo_transitions()

    def store_trajectory_from_veh(self):
        for hex in self.hex_zone_collection.values():
            for vehicle in hex.vehicles.values():
                self.trajectory_transitions += vehicle.dump_trajectories()

    def last_step_transactions(self,tick):
        #store all the transactions from the vehicles:
        for hex in self.hex_zone_collection.values():
            for vehicle in hex.vehicles.values():
                self.all_transitions += vehicle.dump_last(tick)

    def dump_transition_to_dqn(self):
        """
        convert transitions to batches of state, action, next_state, and off-duty flag.
        :return:
        """
        state, action, next_state, reward, terminate_flag, time_steps, num_valid_relos_ = None, None, None, None, None, None, None
        all_transitions = np.array(self.all_transitions, dtype=object)
        if len(all_transitions) > 0:
            # print('First row of Transitions are:',all_transitions[1])
            [state, action, next_state, reward, terminate_flag, time_steps] = [all_transitions[:, i] for i in range(6)]
        if state is not None:
            num_valid_relos_ = [len([0] + self.hex_zone_collection[state_i[1]].neighbor_hex_id) for state_i in
                                next_state]  # valid relocation for next state.
        return state, action, next_state, reward, terminate_flag, time_steps, num_valid_relos_

    def dump_prime_action_to_dqn(self):
        """
        convert transitions to batches of state, action, next_state, and off-duty flag.
        :return:
        """
        state, action, next_state,trip_flag, time_steps, num_valid_relos_ = None, None, None, None, None, None
        all_transitions = np.array(self.prime_transitions, dtype=object)
        if len(all_transitions) > 0:
            # print('First row of Transitions are:',all_transitions[1])
            [state, action, next_state,trip_flag, time_steps] = [all_transitions[:, i] for i in range(5)]
        if state is not None:
            num_valid_relos_ = [len([0] + self.hex_zone_collection[state_i[1]].neighbor_hex_id) for state_i in
                                next_state]  # valid relocation for next state.
        return state, action, next_state, trip_flag, time_steps, num_valid_relos_

    def dump_f_transitions(self):
        state,next_state,on_opt=None,None,None
        all_transitions=np.array(self.f_transitions,dtype=object)
        if len(all_transitions)>0:
            [state,next_state,on_opt]=[all_transitions[:,i] for i in range(3)]
        return state,next_state,on_opt

    def dump_fo_transitions(self):
        state,next_state=None,None
        all_transitions=np.array(self.fo_transitions,dtype=object)
        if len(all_transitions)>0:
            [state,next_state]=[all_transitions[:,i] for i in range(2)]
        return state,next_state

    def dump_trajectories(self):
        trajectory=None
        all_transitions=self.trajectory_transitions
        if len(all_transitions)>0:
            trajectory=self.trajectory_transitions
        return trajectory


    def get_current_time(self):
        return self.__t

    def summarize_metrics(self, charging_dest_file, cruising_od_file, matching_od_file,charging_file,num_attack_file, current_day):
        '''
        note: vehicle in charging status cannot be queried directly since they are in the charging station collection.
        '''
        vehicles_on_duty = [vehicle for hex in self.hex_zone_collection.values() for vehicle in hex.vehicles.values()]
        vehicles_queuing_at_station = []# [list(cs.queue) for cs in self.charging_station_collections]
        vehicles_charging_at_station = []
        occupancy_rates = []
        for cs in self.charging_station_collections:
            vehicles_queuing_at_station+=list(cs.queue)
            num_occupied = 0
            for p in cs.piles:
                if p.occupied == True:
                    vehicles_charging_at_station.append(p.assigned_vehicle)
                    num_occupied+=1
            occupancy_rates.append(num_occupied/(cs.num_dcfc_pile+cs.num_l2_pile))

        status_ids,num_veh_per_status = np.unique([veh.state.status for veh in vehicles_on_duty+vehicles_queuing_at_station+vehicles_charging_at_station],return_counts=True)

        full_status_count = np.zeros(10)
        # print("num_veh_per_status:",num_veh_per_status)
        full_status_count[status_ids]+=num_veh_per_status

        num_scheduled_evs = sum([len(cs.queue) for cs in self.charging_station_collections])
        average_reduced_SOC = np.mean([veh.mileage_per_charge_cycle for veh in vehicles_on_duty if veh.state.status == status_codes.V_WAITPILE])

        n_matches = self.num_match
        average_cumulated_earning = np.mean([veh.total_earnings for veh in vehicles_on_duty])
        average_idle_time = np.mean([veh.idle_time for veh in vehicles_on_duty])
        total_removed_passengers = self.total_num_longwait_pass + self.total_num_served_pass
        total_num_longwait_pass = self.total_num_longwait_pass
        total_num_served_pass = self.total_num_served_pass

        veh_attack_status_ids, num_veh_per_attack_status = np.unique([veh.state.attack_status for veh in vehicles_on_duty+vehicles_charging_at_station+vehicles_queuing_at_station],return_counts=True)
        veh_attack_status = np.zeros(2)
        veh_attack_status[veh_attack_status_ids]+=num_veh_per_attack_status
        veh_0,veh_1 = veh_attack_status[0], veh_attack_status[1]
        # veh_0 = sum([veh.state.attack_status == 0 for veh in vehicles_on_duty+vehicles_charging_at_station+vehicles_queuing_at_station])
        # veh_1 = sum([veh.state.attack_status == 1 for veh in vehicles_on_duty+vehicles_charging_at_station+vehicles_queuing_at_station])

        evcs_attack_status_ids, num_evcs_per_attack_status = np.unique([evcs.status for evcs in self.charging_station_collections],return_counts=True)
        evcs_attack_status = np.zeros(3)
        evcs_attack_status[np.array(evcs_attack_status_ids,dtype=np.int64)]+=num_evcs_per_attack_status
        evcs_0,evcs_1,evcs_2 = evcs_attack_status[0],evcs_attack_status[1],evcs_attack_status[2]

        queuing_time = []
        if int(self.__t % 300) ==0: # record charging profile every minute.
            # [charging_file.writelines(f'{veh.state.id},{veh.charging_wait},{veh.state.attack_status},{self.charging_station_collections[veh.state.assigned_charging_station_id].status},{self.__t},{veh.state.assigned_charging_station_id},{veh.state.SOC},{veh.state.target_SOC},{veh.state.status}\n') for veh in vehicles_on_duty if veh.state.status in [status_codes.V_WAYTOCHARGE,status_codes.V_WAITPILE]]
            for cs in self.charging_station_collections:
                for p in cs.piles:
                    if p.occupied == True:
                        veh = p.assigned_vehicle # track the vehicle id that is charging.
                        queuing_time.append(veh.charging_wait) # output EVCS dynamics
                        charging_file.writelines(f'{veh.state.id},{p.time_to_finish},{veh.charging_wait},{veh.state.attack_status},{cs.status},{self.__t},{veh.state.assigned_charging_station_id},{veh.state.SOC},{veh.state.target_SOC},{veh.state.status}\n')
                    if current_day < self.start_day_of_attack: # collect normal operation charging profile for model training.
                        if p.charging_samples: # if the charging samples are not empty, then record the charging profile.
                            [self.historical_charging_profile.append([cs.id]+sample_i) for sample_i in p.charging_samples]
                            p.remove_charging_samples()

            [charging_dest_file.writelines('{},{}\n'.format(veh.state.id, ','.join(str(item) for item in od))) for veh in vehicles_on_duty for od in veh.charging_od_pairs]
            [cruising_od_file.writelines('{},{}\n'.format(veh.state.id,','.join(str(item) for item in od))) for veh in vehicles_on_duty for od in veh.repositioning_od_pairs]
            [matching_od_file.writelines('{},{}\n'.format(veh.state.id,','.join(str(item) for item in od))) for veh in vehicles_on_duty for od in veh.matching_od_pairs]
            # [demand_supply_gap_file.writelines('{},{},{}\n'.format(self.__t,hex.hex_id, (hex.arrivals[-1] - len(hex.vehicles.keys())) )) for hex in self.hex_zone_collection.values()]

        if int(self.__t % 3600) ==0:
            [num_attack_file.writelines(f"{self.__t//60},{cid},{times_attack}\n") for cid, times_attack in enumerate(self.attack_counter)]

            print(f"current tick is {self.__t//60}, recording data: number of cruising {full_status_count[1]}, scheduled:{num_scheduled_evs}, and charging:{full_status_count[6]}")
            print(f"\n number of cruising {full_status_count[1]}, assigned:{full_status_count[3]}, and occupied:{full_status_count[2]}")
            if self.attack_type >=0 and self.penalty_charging_duration!=0:
                print(f"current tick is {self.__t//60}, EV_S: {veh_0}, EV_I: {veh_1}, EVCS_S: {evcs_0}, EVCS_I: {evcs_1}......")

        [veh.reset_stored_od_pairs() for veh in vehicles_on_duty]

        return full_status_count, n_matches, average_idle_time, average_reduced_SOC, \
               total_removed_passengers,total_num_longwait_pass, total_num_served_pass, average_cumulated_earning,\
               veh_0, veh_1, evcs_0, evcs_1, evcs_2,  self.detection_counter, occupancy_rates, queuing_time

    def reset_storage(self):
        self.all_transitions = []
        self.prime_transitions=[]
        self.f_transitions=[]
        self.fo_transitions=[]
        self.trajectory_transitions=[]
        self.global_state_buffer=dict()
        self.detection_counter = np.zeros(4) # 3 status in total.
        self.charging_profile = []
        self.attack_counter = np.zeros(len(self.charging_station_collections))

    def attach_attack_status(self,day):
        '''
        first_day_of_attack: the first day of attack, default is 7 (the end the of thefirst week)
        '''
        if day == self.start_day_of_attack:# first week as a warm up period.
            self.attack_type = self.config["ATTACK_TYPE"]  # start attack after warm-up.
            if self.attack_type == 0:
                # init_status = np.random.binomial(1, INIT_EVCS_I, len(self.charging_station_collections))  # randomly select infectious charging station
                init_status = np.ones(len(self.charging_station_collections)) # all infected
                for attack_status, cs in zip(init_status, self.charging_station_collections):
                    cs.update_attack_status(attack_status,attack_type=self.attack_type)
                # start training the detection algorithm using one-week data
                print(f"line 809 start training using {len(self.historical_charging_profile)} pieces of charging logs.....")

                self.detection_algorithm.train(self.historical_charging_profile)
                self.historical_charging_profile = [] # reset it.
                self.charging_profile = [] # reset it.

        elif day==0: # as a test (initialize).
            status_S = 0
            attack_type = -1
            for cs in self.charging_station_collections:
                cs.update_attack_status(status_S,attack_type=attack_type) # all safe.

    def reload_demand(self,seed):
        #reset random seed
        local_rand_seed=((seed//5)*7+seed%5+2)%28 # repetitive random seed (2 weeks) (e.g. 14 =0)
        random.seed(local_rand_seed)
        alltrips = self.get_all_trips(local_rand_seed)

        #reset hex zones
        for hex in self.hex_zone_collection.values():
            h_idx=hex.hex_id
            try:
                demand_df=alltrips.get_group(int(h_idx))
            except KeyError:
                demand_df=pd.DataFrame({'tick':[]})
            hex.all_trips=demand_df.groupby('tick')
            hex.seed=local_rand_seed

    def get_all_trips(self,local_rand_seed):
        #reset random seed
        filename = f"../data/daily_trips/{local_rand_seed}.csv"

        n = sum(1 for _ in open(filename)) - 1  # number of records in file (excludes header)
        s = int(n*self.demand_scale)  # desired sample size
        skip = sorted(random.sample(range(1, n + 1), n - s))  # the 0-indexed header will not be included in the skip list
        alltrips = pd.read_csv(filename,skiprows=skip, dtype=int)
        print('Total number of trips in this episode: {}'.format(alltrips.shape[0]))
        alltrips=alltrips.groupby('origin_hid')
        return alltrips

    def reset(self,start_time,timestep,seed=None):
        #restart the simulator, but assuming the time of previous end time
        self.reset_time(start_time, timestep)
        self.last_vehicle_id = 1
        self.vehicle_queue = PriorityQueue()
        # DQN for getting actions and dumping transitions
        self.all_transitions = [] #clear the memories
        self.num_match = 0
        self.total_num_arrivals = 0
        self.total_num_removed_pass = 0
        self.total_num_served_pass = 0
        self.total_num_longwait_pass = 0
        self.total_idled_vehicles = 0
        self.current_dummyV = 0
        self.current_dqnV = 0
        self.detection_counter = np.zeros(4) # 3 status in total.
        self.attack_counter = np.zeros(len(self.charging_station_collections))

        #reset random seed
        local_rand_seed=(seed//5)*7+seed%5+2 # repetitive random seed (2 weeks) (e.g. 14 =0)
        random.seed(local_rand_seed)

        alltrips = self.get_all_trips(local_rand_seed)

        #reset hex zones
        for hex in self.hex_zone_collection.values():
            h_idx=hex.hex_id
            try:
                demand_df=alltrips.get_group(int(h_idx))
            except KeyError:
                demand_df=pd.DataFrame({'tick':[]})
            hex.all_trips=demand_df.groupby('tick')
            hex.seed=local_rand_seed
            hex.reset()

        #### note: adding back if we need to reset the fleet and charging stations. ####
        init_status = np.zeros(len(self.charging_station_collections)) # first 1 is no-attack period.
        #reset charging stations
        for cs, status in zip(self.charging_station_collections,init_status):
            cs.reset(status)

        self.generate_veh_q()
        print('simulator reset complete')

    def generate_veh_q(self):
        hex_ids=[hex.hex_id for hex in self.hex_zone_collection.values()]
        vehicle_hex_ids = np.random.choice(hex_ids,self.n_vehicles) # randomly initialize location.
        status_list = np.random.binomial(1,INIT_VEH_I,self.n_vehicles) # initiate infectious EVs
        vehicle_ids = np.arange(self.last_vehicle_id, self.last_vehicle_id + self.n_vehicles)
        SOC_list = np.round(np.maximum(LOW_SOC_THRESHOLD,np.minimum(1,np.random.normal(HIGH_SOC_THRESHOLD, INIT_SOC_STD, self.n_vehicles))),3)
        self.last_vehicle_id += self.n_vehicles # some vehicle takes shift, new vehicle need to initial upon that.
        entering_time = np.zeros(self.n_vehicles)#np.random.uniform(self.__t, self.__t + ENTERING_TIME_BUFFER, self.n_vehicles)
        [self.vehicle_queue.put((t,vid,hid,SOC,status)) for t,vid,hid,SOC, status in zip(entering_time, vehicle_ids, vehicle_hex_ids, SOC_list, status_list)]

    def reset_time(self, start_time=None, timestep=None):
        if start_time is not None:
            self.__t = start_time
            self.start_time = start_time
        if timestep is not None:
            self.__dt = timestep

    def get_processed_trip(self, filename):
        '''
        :param filename: trip time OD data or trip count OD data, preprocessed from taxi-records-201605
        :return: 3d numpy array of time OD or trip count OD
        '''
        nhex = NUM_REACHABLE_HEX
        # process the line based file into a hour by
        data = np.zeros((24, nhex, nhex))
        with open(filename, 'r') as f:
            next(f)
            for lines in f:
                line = lines.strip().split(',')
                h, o, d, t = line[1:]  # hour, oridin, dest, trip_time/num of trip # t is the number of trips
                data[int(h), int(o), int(d)] = float(t)*self.demand_scale
        return data

    def get_travel_time(self, filename):
        '''
        :param filename: trip time OD data or trip count OD data, preprocessed from taxi-records-201605
        :return: 3d numpy array of time OD or trip count OD
        '''
        nhex = NUM_REACHABLE_HEX
        # process the line based file into a hour by
        data = np.zeros((24, nhex, nhex))
        with open(filename, 'r') as f:
            next(f)
            for lines in f:
                line = lines.strip().split(',')
                h, o, d, t = line[1:]  # hour, oridin, dest, trip_time/num of trip
                data[int(h), int(o), int(d)] = float(t)
        data[data==0.0] = 60.0
        return data

    def read_terminal(self):
        middle_terminal = dict()
        for oid in range(OPTION_DIM):
            with open('saved_f/term_states_%d.csv' % oid, 'r') as ts:
                next(ts)
                for lines in ts:
                    line = lines.strip().split(',')
                    hr, hid = line  # option_network_id, hour, hex_ids in terminal state
                    if (oid, int(hr)) in middle_terminal.keys():
                        middle_terminal[(oid, int(hr))].append(int(hid))
                    else:
                        middle_terminal[(oid, int(hr))] = [int(hid)]
        return middle_terminal


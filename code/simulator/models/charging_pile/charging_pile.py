from collections import deque
import numpy as np
from config.hex_setting import CHARGE_ACCELERATOR, QUEUE_LENGTH_THRESHOLD, PER_TICK_DISCOUNT_FACTOR, \
    DELAYED_PENALTY_TIME, EVCS_EV_TRANSMISSION_RATE, DELAYED_PENALTY_TIME_STD, NUM_NEAREST_CS, \
    MAX_CHARGING_WAIT_TIME
from novelties import status_codes

class ChargingPile:
    def __init__(self, type, location, hex_id, hex):
        """
        :param type: Level 2 or DC-fast
        :param location: lon lat
        :param hex_id: hex_id
        :param hex: the object hex
        metrics:
            rate: charging speed
            unit_time price by type: $1.5/hr or 0.30 per min, Source: EVgo in NYC https://account.evgo.com/signUp
        """
        self.occupied = False
        self.time_to_finish = 0
        self.assigned_vehicle = None  # vehicle agent
        self.type = type
        self.location = location
        self.hex_id = hex_id
        self.hex = hex
        self.served_num = 0
        self.charging_samples = deque(maxlen=15) # the last 15 charging events
        if self.type == status_codes.SP_DCFC:
            self.rate = CHARGE_ACCELERATOR * (0.7*220)/(30*60)  # mile per sec
            self.unit_time_price = 18 / (60*60)  # 18 USD per hour
        else:
            self.rate = CHARGE_ACCELERATOR * 25 / (60 * 60)  # mile per sec
            self.unit_time_price = 1.5 / (60*60)  # 1.5 USD per hour
        self.last_start_charging_SOC = 0
        self.last_charging_duration = 0
        self.last_start_charging_time = 0
        self.last_queuing_time = 0

    def reset(self):
        self.occupied = False
        self.time_to_finish = 0
        self.assigned_vehicle = None  # vehicle agent
        self.served_num = 0

    def assign_vehicle(self, veh):
        self.occupied = True
        self.time_to_finish = (veh.get_target_SOC() - veh.get_SOC()) * veh.get_mile_of_range() / self.rate
        self.assigned_vehicle = veh
        veh.start_charge()

    def get_cp_location(self):
        return self.location

    def get_cp_hex_id(self):
        return self.hex_id

    def step(self, time_step, tick, infectious):
        """
        :param time_step: 60 seconds per minute
        :param tick: current tick
        :return:
        """
        if self.time_to_finish > 0:
            self.time_to_finish -= time_step
            # if infectious:
            #     self.keep_attacking_ev()
            if self.time_to_finish <= 0:
                # charging has been completed!
                self.assigned_vehicle.end_charge(tick, self.unit_time_price) # cost per unit time by charging type
                self.hex.add_veh(self.assigned_vehicle)  # add the vehicle back, also dump charging info.
                self.dump_charging_info() # dump charging info to the simulator, e.g, SOC, duration, hour, etc.
                self.occupied = False
                self.time_to_finish = 0
                self.assigned_vehicle = None # reset.
                self.served_num += 1

    def keep_attacking_ev(self):
        pass
        # if self.assigned_vehicle.state.attack_status ==0:
        #     self.assigned_vehicle.state.attack_status = np.random.binomial(1, EVCS_EV_TRANSMISSION_RATE) # at every time step.

    def save_charging_info(self,tick):
        '''
        save charging info, e.g, SOC, duration, hour, etc.
        proceed after charging is completed.
        '''
        self.last_start_charging_SOC = self.assigned_vehicle.state.SOC
        self.last_charging_duration = self.time_to_finish
        self.last_start_charging_time = tick
        self.last_queuing_time = self.assigned_vehicle.charging_wait

    def dump_charging_info(self):
        '''
        save charging info, e.g, SOC, duration, hour, etc.
        proceed after charging is completed.
        '''

        self.charging_samples.append([self.last_start_charging_SOC, self.last_charging_duration/60, self.last_start_charging_time/60%1440//60])

    def get_charging_samples(self):
        return self.charging_samples

    def remove_charging_samples(self):
        self.charging_samples = deque(maxlen=15)

    def add_charging_time(self, time):
        '''
        add charging time to the charging pile. triggered if the charging pile is compromised.
        '''
        self.time_to_finish += time

class ChargingStation:
    def __init__(self, n_l2=1, n_dcfast=1, lat=None, lon=None, hex_id=None, hex=None, row_col_coord=None, status=0,attack_type=0,id=0,tipping_tick=0,penalty_charging_duration=0,time_to_recover=0):
        self.location = float(lon), float(lat)
        # initial the charging piles for the charging station
        self.piles = [ChargingPile(type=status_codes.SP_LEVEL2, location=self.location, hex_id=hex_id, hex=hex) for _ in
                      range(n_l2)] + \
                     [ChargingPile(type=status_codes.SP_DCFC, location=self.location, hex_id=hex_id, hex=hex) for _ in
                      range(n_dcfast)]
        # self.available_piles=self.piles
        self.waiting_time = deque(maxlen=15) # the last 15 served passengers
        self.charging_time = []
        self.queue = deque()  # waiting queue for vehicle
        self.virtual_queue = []
        self.time_to_cs = []
        self.hex_id = hex_id
        self.hex = hex
        self.row_id, self.col_id = row_col_coord
        self.num_l2_pile = n_l2
        self.num_dcfc_pile = n_dcfast
        self.status = status # S-I-D-R (susceptible, infectious, detected, removed)
        self.attack_type = attack_type
        self.id = id
        self.tipping_tick = tipping_tick
        self.penalty_charging_duration = penalty_charging_duration
        self.time_to_recover = time_to_recover

    def reset(self,status):
        self.waiting_time = deque(maxlen=15) # the last 15 served passengers
        self.charging_time = []
        self.queue = deque()  # waiting queue for vehicle
        self.virtual_queue = []
        self.time_to_cs = []
        self.status = status # randomly generate status (as initial status)
        [p.reset() for p in self.piles] #reset the status of all charging piles

    def update_attack_status(self,attack_status,attack_type):
        self.status = attack_status
        self.attack_type = attack_type

    def get_cs_location(self):
        return self.location

    def get_cs_hex_id(self):
        return self.hex_id

    def update_available_piles(self):
        # set the list of available charging piles
        self.available_piles = [p for p in self.piles if p.occupied == False]

    def update_remaining_capacity(self):
        self.remaining_capacity = len(self.available_piles) - len(self.queue)

    def step(self, time_step, tick):
        '''
        First update each pile, then find available charging piles, then match, then update queue
        :return:
        '''
        # update the status
        [p.step(time_step, tick,self.status==1) for p in self.piles]  # update the status of each charging pile
        self.update_available_piles()  # update available piles
        self.update_remaining_capacity() # update the remaining capacity of each charging pile
        # update waiting time of each vehicle in the queue
        # assign waiting vehicles to each pile
        # must have both vehicle and pile available to proceed
        if self.status == 2: # removed and repairing
            self.available_piles = []  # no available piles
            self.relocate_evs(tick) # clear up evs in queue since the EVCS is not available
        elif self.status !=0 and self.attack_type == 1: # infected by Denial-of-Service attack.
            self.available_piles = []  # no available piles
            self.relocate_evs(tick) # clear up evs in queue since the EVCS is not available
        else: # (1) susceptible or (2) infectious but delayed charging.
            # first generate an array of random number (at most the fleet size)
            np.random.seed(self.id) # charging station id
            add_charging_time = np.maximum(0,np.random.normal(self.penalty_charging_duration,DELAYED_PENALTY_TIME_STD,size=len(self.queue)))*60 # SECOND_PER_MINUTE
            idx = 0
            while len(self.queue) > 0 and len(self.available_piles) > 0:
                veh = self.queue.popleft()
                pile = self.available_piles.pop()
                pile.assign_vehicle(veh) # add veh to pile, and the veh is removed from the queue in CS
                if self.status==0 or self.attack_type==-1: # suspectible
                    self.waiting_time.append(veh.charging_wait)  # total waiting time of the vehicle
                    self.charging_time.append(pile.time_to_finish)  # total charging time
                    pile.save_charging_info(tick)
                else: # delayed service: EV delayed a pre-determined duration (e.g., 10 min)
                    self.waiting_time.append(veh.charging_wait + add_charging_time[idx])  # total waiting time of the vehicle
                    self.charging_time.append(pile.time_to_finish + add_charging_time[idx])  # total charging time
                    ### EV will not be infected.
                    # if veh.state.attack_status == 0: # EV is S status
                    #     veh.state.attack_status = np.random.binomial(1,EVCS_EV_TRANSMISSION_RATE) # EV is infected with a probability.
                    veh.add_charging_wait_time(add_charging_time[idx])
                    pile.add_charging_time(add_charging_time[idx])
                    pile.save_charging_info(tick)
                idx+=1 # iterate the added charging time

        # for unmatched vehicles, update the waiting time of vehicles in the charging queue
        list_of_veh = list(self.queue)
        for v in list_of_veh:
            v.charging_wait += 60 # second per minute
            if v.charging_wait >= MAX_CHARGING_WAIT_TIME: # e.g., one hour.
                self.relocate_evs(tick)  # clear up evs in queue since the EVCS is not available

                # self.queue.remove(v)
                # self.virtual_queue.append(v)
                # self.time_to_cs.append(v.charging_wait)
                # self.waiting_time.append(v.charging_wait)
                # self.charging_time.append(0)
                # self.relocate_evs(tick)


    def relocate_evs(self,tick):
        '''
        relocate EVs that are currently waiting or scheduled to other nearby EVCSs.
        '''
        while self.queue: # while queue is not empty
            veh = self.queue.pop()
            self.hex.add_veh(veh)  # add the vehicle back, rematch with another charging station again.
            veh.charging_wait = 0 # reset the waiting time
            veh.DoS_end_charge(tick)
            veh.softmax_sending_to_charging_station_pool()
            # cid is one of the nearest charging station.
            # target is the coordinates (may be not useful)
            # tick is the current tick
            # action id: the index of action (recorded for sampling purpose)

    def add_arrival_veh(self, veh):
        '''
        :param veh: vehicle object (class)
        '''
        # if len(self.queue)/(self.num_dcfc_pile+self.num_l2_pile) <= QUEUE_LENGTH_THRESHOLD: # we seperate dcfc and l2, so either of the number must be 0.
        self.queue.append(veh)
        veh.charging_wait = 0  # no wait at the beginning
        # print(veh.state.origin_hex,veh.state.current_hex, veh.state.destination_hex, veh.state.per_tick_coords, self.hex_id,self.location)
        self.hex.remove_veh(veh)  # remove this vehicle from the hex zone
        # else:
        #     veh.quick_end_charge()  # quick pop out the vehicle if the queue is long.

    def get_queue_length(self):
        return len(self.queue)

    def get_average_waiting_time(self):
        if len(self.waiting_time) < 1:
            return [0.0]
        else:
            return np.mean(self.waiting_time)

    def get_served_num(self):
        return sum([cp.served_num for cp in self.piles])

#### in your funciton, you can define your charging repository as a list of length N, N = total number of stations

# from simulator.services.routing_service import RoutingEngine
from config.hex_setting import OFF_DURATION, RELOCATION_DIM
from novelties import status_codes
from simulator.models.customer.customer import Customer
from simulator.models.customer.request import request
from collections import defaultdict
import numpy as np
import contextlib
from novelties.pricing.price_calculator import calculate_price
def weighted_random(w, n):
    cumsum = np.cumsum(w)
    rdm_unif = np.random.rand(n)
    return np.searchsorted(cumsum, rdm_unif)


@contextlib.contextmanager
def local_seed(seed):
    # this defines a local random seed funciton, and let the simulator to resume previous random seed
    state = np.random.get_state()
    np.random.seed(seed)  # set seed
    try:
        yield
    finally:
        np.random.set_state(state)  # put the state back on

class hex_zone:

    def __init__(self, hex_id, coord,row_col_coord, coord_list, match_zone, neighbors, charging_station_ids, charging_hexes,
                 charging_coords, od_split, trip_time, t_unit, epoch_length,seed,demand_df, highdemand_hexs):
        """
        hex_id: id of the hexagon zone in the shapefile
        coord: lon and lat values
        arrival_rate: number of arrivals per tick
        neighbors: adjacent hexagons' ids
        charging_station_ids: nearest 5 charging station ids
        charging_coords: list of coordinates of the 5 charging stations
        epoch_length: total ticks per epoch of simulation: 60 * 24 * SIM_DAYS
        """
        self.seed=seed
        self.hex_id = hex_id
        self.match_zone_id = match_zone
        self.lon, self.lat = coord
        self.row_id, self.col_id = row_col_coord
        self.coord_list = coord_list  # this is the list for all the lon lat coordinates of the hexagons
        od_split = np.reshape(od_split, (od_split.shape[0], od_split.shape[-1]))
        trip_time = np.reshape(trip_time, (trip_time.shape[0], trip_time.shape[-1]))  # remove one of the dimension
        self.arrival_rate = np.sum(od_split,
                                   axis=-1).flatten() / t_unit  # now this becomes a  hour by 1 array,and we convert this to each tick of demand!
        self.next_arrivals = None
        # 1 by N matrix
        self.od_ratio = od_split
        self.trip_time = trip_time

        # the following two defines the actions
        self.neighbor_hex_id = neighbors  # length may vary
        self.highdemand_hexs = highdemand_hexs  # list of list (per hour interval)
        self.nearest_cs = charging_station_ids
        self.charging_hexes = charging_hexes
        self.charging_station_loc = charging_coords
        self.n_charges=0 #number of charging stations
        self.passengers = defaultdict()
        self.vehicles = defaultdict()
        self.served_num = 0
        self.removed_passengers = 0
        self.served_pass = 0
        self.total_served=0
        self.longwait_pass = 0
        self.veh_waiting_time = 0
        self.total_pass = 0  # this also servers as passenger id

        self.t_unit = t_unit  # number of ticks per hour
        self.epoch_length = epoch_length
        self.q_network = None
        self.narrivals = 0
        self.next_narrivals = 0
        self.all_trips=demand_df.groupby('tick')
        # initialize the demand for each hexagon zone
        self.init_demand()
        self.initial_arrivals = self.arrivals
        self.initial_destinations = self.destinations

    def reset(self):
        #reinitialize the status of the hex zones
        self.passengers.clear()
        self.vehicles.clear()
        self.served_num = 0
        self.removed_passengers = 0
        self.served_pass = 0
        self.longwait_pass = 0
        self.veh_waiting_time = 0
        self.narrivals = 0
        self.next_narrivals = 0
        # initialize the demand for each hexagon zone
        self.init_demand()

    def init_demand(self):
        '''
        todo: generate all the initial demand for each hour. Fix a local random generator to reduce randomness
        :return:
        '''
        # copy the arrival rate list multiple times!
        self.arrivals=[]
        self.destinations=[]
        for tick in range(1440):
            try:
                td = self.all_trips.get_group(tick)['destination_hid'].tolist()
            except KeyError: # no demand at this tick
                td = []
            self.arrivals.append(len(td))
            self.destinations.append(td)
        self.arrivals.append(0)
        self.destinations.append([])
        self.arrivals.reverse()
        self.destinations.reverse()

    def add_veh(self, veh):
        '''
        add and remove vehicles by its id
        id contained in veh.state
        :param veh: a class vehicle
        :return:
        '''
        self.vehicles[veh.state.vehicle_id] = veh

    def remove_veh(self, veh):
        self.vehicles.pop(veh.state.vehicle_id)  # remove the vehicle from the list

    def demand_generation(self,tick):
        hour = tick // (self.t_unit * 60) % 24

        if len(self.destinations)==1: # a new day, the list is initialized with a length of 1441 ([0]).
            self.init_demand()

        destinations=self.destinations.pop()

        self.arrivals.pop()
        for i in range(len(destinations)):
            # r={'id':self.total_pass,'origin_id':self.hex_id, 'origin_lat':self.lat, 'origin_lon':self.lon, \
            #    'destination_id':destinations[i], 'destination_lat':self.coord_list[destinations[i]][1], 'destination_lon':self.coord_list[destinations[i]][0], \
            #        'trip_time':self.trip_time[hour,destinations[i]],'request_time':tick}
            # r=request(self.total_pass, self.hex_id, (self.lon,self.lat,), destinations[i], self.coord_list[destinations[i]],self.trip_time[hour,destinations[i]],tick)
            self.passengers[(self.hex_id, self.total_pass)] = Customer(
                request(self.total_pass, self.hex_id, (self.lon, self.lat), destinations[i],
                        self.coord_list[destinations[i]], self.trip_time[hour, destinations[i]],
                        tick))  # hex_id and pass_id create a unique passenger identifier
            self.total_pass += 1

    def remove_pass(self, pids):  # remove passengers
        '''
        Remove passengers by key_id
        :return:
        '''
        [self.passengers.pop(pid) for pid in pids]

    def update_passengers(self):
        """
        code for updating the passenger status / or remove them if picked up
        """
        remove_ids = []
        self.longwait_pass=0
        self.served_pass=0

        for pid in self.passengers.keys():
            if self.passengers[pid].matched==True: # we have matched passengers to requests.
                self.served_pass +=1
                remove_ids.append(pid)
            elif self.passengers[pid].waiting_time >= self.passengers[pid].max_tolerate_delay:  # remove passengers after 10 ticks.
                self.longwait_pass +=1
                remove_ids.append(pid)
            else:
                self.passengers[pid].waiting_time += self.t_unit  # update waiting time

        self.removed_passengers += len(remove_ids)
        self.remove_pass(remove_ids)

    def remove_matched(self):
        remove_ids = []
        self.longwait_pass=0
        self.served_pass=0
        for pid in self.passengers.keys():
            if self.passengers[pid].matched==True:
                self.served_pass +=1

        self.removed_passengers += len(remove_ids)
        self.remove_pass(remove_ids)

    def hexzone_dispatch(self, tick, target_hex_id, target_cs_hex_id, target_cs_id):
        """
        Dispatch the vehicles. This step follows from matching step
        :param tick: current tick (min)
        :param target_hex_id: the hex_id of the target hexagon
        :return:
        """
        for vehicle in self.vehicles.values():
            if vehicle.state.status == status_codes.V_TOBEDISPATCHED:
                action_id = vehicle.state.dispatch_action_id # the action to execute, negative if to charge.
                offduty = False  # we don't consider the rest here.
                if offduty:
                    off_duration = np.random.randint(OFF_DURATION/2, OFF_DURATION * 3 / 2)
                    vehicle.take_rest(off_duration)
                else: # on-duty    # Get target destination and key to cache
                    target_coord, charge_flag, destination_hex_id = self.convert_action_to_destination(action_id,target_hex_id, target_cs_hex_id)
                    vehicle.state.origin_hex = vehicle.state.hex_id
                    vehicle.state.need_route = True
                    if charge_flag:
                        vehicle.state.destination_hex = destination_hex_id
                        vehicle.heading_to_charging_station(target_cs_id, target_coord, tick, action_id)
                    else:
                        vehicle.state.destination_hex = destination_hex_id
                        if vehicle.state.hex_id == destination_hex_id:
                            action_id = 0 # stay in the same hexagon
                        vehicle.cruise(target_coord, action_id, tick, destination_hex_id) # todo: add the global state here.

    def convert_action_to_destination(self, action_id,target_hex_id, target_cs_hex_id):
        '''
        action_id: action ids from 0-11, pre-derived from DQN
        return:
        target: lon and lat
        charge_flag: 1 if need to charge.
        target_hex_id: hex_id
        cid: charging station id, return None if not charge.
        '''
        if action_id>=0:
            lon, lat = self.coord_list[target_hex_id]
            charge_flag = False
            target_hex = target_hex_id
        else: # charging actions
            lon, lat = self.coord_list[target_cs_hex_id]
            charge_flag = True
            target_hex = target_cs_hex_id
        target = (lon, lat)

        return target, charge_flag, target_hex

    def get_num_arrivals(self):
        return self.narrivals

    def get_next_num_arrivals(self):
        return self.next_narrivals

    def get_num_removed_pass(self):
        return self.removed_passengers

    def get_num_served_pass(self):
        return self.served_pass

    def get_num_longwait_pass(self):
        return self.longwait_pass

    def get_stay_idle_veh_num(self):
        idle_vehicles = {key: vehicle for key, vehicle in self.vehicles.items() if
                        vehicle.state.status in [status_codes.V_STAY, status_codes.V_IDLE]}
        return len(idle_vehicles)

    def get_passenger_num(self):
        passenger_dict = {key: passengers for key, passengers in self.passengers.items()}
        return len(passenger_dict)



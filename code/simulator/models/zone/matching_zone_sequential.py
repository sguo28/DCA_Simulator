from novelties import status_codes
from collections import defaultdict
from config.hex_setting import REJECT_TIME, SEC_PER_MIN, MAX_MATCH_TIME, MAX_WAIT_TIME, NUM_NEAREST_CS, \
    HIGH_SOC_THRESHOLD,LOW_SOC_THRESHOLD, TOLERANCE_FOR_QUEUING
import numpy as np

class matching_zone(object):
    def __init__(self, m_id, hex_zones, time_od):
        """
        m_id: matching zone id
        hex_zones: the list of hex zone objects
        """
        self.matching_zone_id = m_id
        self.hex_zones = hex_zones
        self.reject_wait_time = REJECT_TIME * SEC_PER_MIN  # sec
        self.local_hex_collection = {hex.hex_id: hex for hex in hex_zones}  # create a a local hex
        self.num_matches = 0
        self.time_od = time_od
        self.charging_prob = lambda x: np.exp(-LOW_SOC_THRESHOLD*np.log(1e-4)/(HIGH_SOC_THRESHOLD-LOW_SOC_THRESHOLD)+np.log(1e-4)/(HIGH_SOC_THRESHOLD-LOW_SOC_THRESHOLD)*x)

    def get_local_collection(self):
        return self.local_hex_collection

    def get_info(self):
        print('Match zone id: {}, number of hexs:{}'.format(self.matching_zone_id, len(self.hex_zones)))

    def matchingzone_dispatch(self, tick, target_hex_ids, target_cs_hex_ids, target_cs_ids):
        '''
        Call dispatch for each hex zones
        :param tick:
        :return:
        '''
        [h.hexzone_dispatch(tick, hid,cs_hid, cs_id) for h,hid,cs_hid, cs_id in zip(self.hex_zones,target_hex_ids,target_cs_hex_ids, target_cs_ids)] # zone level.

    def update_passengers(self):
        '''
        Call update passenger in each hex zones
        :return:
        '''
        [h.update_passengers() for h in self.hex_zones]

    def matchingzone_update_vehicles(self,timestep,timetick):
        '''
        call step function for each vehicles
        :return:
        '''
        all_veh=[veh for hex in self.hex_zones for veh in hex.vehicles.values()]
        for veh in all_veh:
            veh.step(timestep,timetick,self.hex_zones)

    def async_demand_gen(self, tick):
        # do the demand generation for all hex zones in the matching zone
        [h.demand_generation(tick) for h in self.hex_zones]
        return True

    def get_vehicles_by_hex(self):
        '''
        return: list of vehicle_dict per hex
        '''
        veh_dict = [hex.vehicles for hex in self.hex_zones]
        return veh_dict

    def get_vehicles_by_hex_list(self):
        """
        :return:
        """
        veh_dict = [hex.vehicles.values() for hex in self.hex_zones]
        return veh_dict

    def set_vehicles_by_hex(self, new_veh,timestep,timetick):
        # reset the new collection of vehicles for each hex areas in the matching zone
        # update the vehicle at the same time as push
        # make sure the order in new_veh is the same as the hex zone orders in each matching zone
        for i in range(len(new_veh)):
            self.hex_zones[i].vehicles = new_veh[i]
            for veh in self.hex_zones[i].vehicles.values():
                veh.step(timestep,timetick)
        # print('total vehicles deployed to zone {} is {}'.format(self.matching_zone_id,nvs))

    def get_arrivals_length(self):
        return len(self.hex_zones[0].arrivals)

    def get_all_veh(self):
        '''
        :return: all vehicles in the hex areas inside the matching zone
        '''
        all_vehs = defaultdict()
        for hex_zone in self.hex_zones:
            all_vehs.update(hex_zone.vehicles)
        return all_vehs

    def get_all_passenger(self):
        '''
        :return: all available passengers in the list
        todo: consider sorting the passengers based on their time of arrival?
        '''
        available_pass = defaultdict()
        for hex_zone in self.hex_zones:
            local_availables = {key: value for (key, value) in hex_zone.passengers.items() if value.matched == False}
            available_pass.update(local_availables)
        return available_pass

    def get_served_num(self):
        return sum([h.served_num for h in self.hex_zones])

    def get_veh_waiting_time(self):
        '''
        todo: this function makes no sense [# this is the waiting time for a charging pile]
        :return:
        '''
        return sum([h.veh_waiting_time for h in self.hex_zones])

    def matchingzone_match(self,tick,charging_stations):
        '''
        Perform the matching here.
        :return:
        '''
        # get all vehicles and passengers first
        all_passengers = self.get_all_passenger()
        all_veh = self.get_all_veh()
        self.matching_algorithms(all_passengers, all_veh,charging_stations, tick)

    def matching_algorithms(self, passengers, vehicles, charging_stations, tick):
        '''
        passengers: the set of available Customer objects
        vehicles: the set of vehicle objects
        match available vehicles with passengers
        Change the status for passengers and vehicles
        :return:
        no return here. We will change the mark for each passengers and drivers as they are matched
        '''
        self.num_matches=0
        vehicles_to_match = {}
        vehicles_to_charge = {}
        if passengers and vehicles: # both are not empty
            for veh_id, veh in vehicles.items():
                if veh.state.status in [status_codes.V_IDLE, status_codes.V_STAY, status_codes.V_CRUISING]:
                    if not self.check_SOC(veh): # no need to charge
                        vehicles_to_match.update({veh_id:veh})
                    else:
                        vehicles_to_charge.update({veh_id:veh})
        elif vehicles:# if no demand, have vehicles: check if vehicles need to charge
            for veh_id, veh in vehicles.items():
                if veh.state.status in [status_codes.V_IDLE, status_codes.V_STAY, status_codes.V_CRUISING]:
                    if self.check_SOC(veh): # check if SOC is below a threshold
                        vehicles_to_charge.update({veh_id:veh})

        if vehicles_to_match:
            self.match_with_requests(vehicles_to_match, passengers,tick) # otherwise there were issues with an empty dataframe.
        if vehicles_to_charge:
            self.matching_algorithm_with_charging_station(charging_stations,vehicles_to_charge)
            # self.match_with_charging_stations(vehicles_to_charge) # randomly select a charging station

    def match_with_charging_stations(self,vehicles):
        '''
        vehicles: set of vehicles that need to charge
        '''
        random_charging_actions = -1-np.random.randint(NUM_NEAREST_CS,size=len(vehicles)) # randomly choose a charging station
        for veh, charging_action_id in zip(list(vehicles.values()),random_charging_actions):
            veh.softmax_sending_to_charging_station_pool()
            # veh.send_to_charging_station_pool(veh.assigned_action,charging_action_id)

    def matching_algorithm_with_charging_station(self,charging_stations,vehicles):
        '''
        charging_stations: set of charging stations
        vehicles: the set of vehicle objects
        match available vehicles with passengers
        Change the status for passengers and vehicles
        :return:
        no return here. We will change the mark for each passengers and drivers as they are matched
        '''
        v_hex_id = [veh.state.current_hex for veh in vehicles.values()]
        c_hex_id = [cs.hex_id for cs in charging_stations]
        remaining_capacity = [cs.remaining_capacity + TOLERANCE_FOR_QUEUING*len(cs.piles) for cs in charging_stations] # how many more EVs can be accomodated.
        # print(f"check remaining capacity: {remaining_capacity[:5]} (first 5)")

        time_od_mat = self.time_od[v_hex_id,:][:,c_hex_id]
        assignments = self.veh2cs_greedy_matching(time_od_mat,remaining_capacity,method="min_time_first")
        for veh,(_,c_id) in zip(vehicles.values(),assignments):
            # print(f"vehicle {veh.state.id} is assigned to charging station {c_id}")
            veh.send_to_charging_station_pool(veh.assigned_action,-1-c_id) # -1-c_id is to differentiate the charging station from normal relocating.

    def veh2cs_greedy_matching(self,time_od_mat,remaining_capacity,method="min_time_first"):
        '''
        :param time_od_mat:
        :param remaining_capacity:
        :param method: "min_time_first" or "max_capacity_first" (within a distance) or "utility_greedy"
        :return:
        '''
        assignments = []
        if method == "min_time_first":
            weight_mat = time_od_mat
            weight_mat[weight_mat>=60*60] = np.nan # set the time to infinity for those that are too far away.
        elif method == "max_capacity_first":
            weight_mat = remaining_capacity
        elif method == "utility_greedy": # we will consider waiting time (recent 15 minutes) and distance, and use a softmax function to select the best one.
            weight_mat = time_od_mat/remaining_capacity # todo: check if this is correct
        else:
            raise ValueError("method not supported")
        # print(f"weight_mat: {weight_mat[:5]} (first 5)")
        for v_id in range(weight_mat.shape[0]):
            while True:
                c_id = np.nanargmin(weight_mat[v_id,:])
                # print(weight_mat[v_id,c_id],remaining_capacity[c_id],weight_mat.shape)
                if remaining_capacity[c_id] > 0: # if the charging station is not full, we finish the matching
                    break
                elif np.max(remaining_capacity*weight_mat[v_id,:])>0: # there is an available charging station within a certain distance, we keep searching.
                    weight_mat[v_id,c_id] = np.nan
                    continue
                else: # all charging stations are full
                    c_id = np.nanargmin(weight_mat[v_id,:]) # go back to the first available charging station
                    break
            remaining_capacity[c_id] -= 1  # scheduled.
            assignments.append([v_id, c_id])

        return assignments

    def check_SOC(self,veh):
        '''
        :param veh: a class of vehicle
        return: 0: no charge, 1: need to charge
        '''
        charge_flag = 0
        if veh.state.SOC<=LOW_SOC_THRESHOLD: # must charge
            charge_flag = 1
        elif veh.state.SOC<HIGH_SOC_THRESHOLD: # a linear probability
            charge_flag = np.random.binomial(1,self.charging_prob(veh.state.SOC))
        return bool(charge_flag)

    def match_with_requests(self, vehicles, passengers, tick):
        """
        :param vehicles:
        :param passengers:
        :return:
        ====
        match requests to vehicles
        """
        v_hex_id = [veh.state.current_hex for veh in vehicles.values()]

        vehs = list(vehicles.values())

        customers = np.array([cs for cs in passengers.values()])
        wait_time = np.array([cs.waiting_time for cs in customers])
        inds = np.flip(np.argsort(wait_time))  # highest to lowest

        # sort the customers from longest waiting to shortest waiting
        # sorted_customer = customers[inds]

        sorted_customer=customers # do not sort by waiting time

        r_hex_id = [c.request.origin_id for c in sorted_customer]
        # requests = [c for c in sorted_customer]

        od_mat = self.time_od[v_hex_id,:][:,r_hex_id]
        # update the time with already matching time
        # for cid,cs in enumerate(requests):
        #     od_mat[:,cid]+=cs.waiting_time
        assignments = self.assign_nearest_vehicle(od_mat,wait_time)

        for [v_id, r_id] in assignments:
            vehicle = vehs[v_id]
            customer = sorted_customer[r_id]
            customer.matched = True
            vehicle.customer = customer  # add matched passenger to the current on
            vehicle.state.need_route = True
            # vehicle.state.current_hex = customer.get_origin()
            # note: the function "heading_to_customer" changes the status to "assigned", which is updated by tick followed by the status "picked up".
            vehicle.heading_to_customer(customer.get_origin_lonlat(),customer.get_origin(),tick)
        self.num_matches +=len(assignments)  # record nums of getting matched

    # Returns list of assignments
    def assign_nearest_vehicle(self,od_mat,wait_time):
        """
        :param:vehicles: vehicles in idled status
        :param: od_tensor: a V by R time matrix (vehicle and request)
        :return: matched od pairs: vid, rid (row and col numbers of od_mat)
        """
        assignments = []
        time = od_mat

        for rid in range(time.shape[1]): #loop through all columns
            if time[:,rid].min()+wait_time[rid]>=MAX_MATCH_TIME:
                continue
            vid=time[:,rid].argmin()
            time[vid,:]=1e9#np.NaN #mask vehicle out
            assignments.append([vid,rid])

        return assignments

    def get_metrics(self):
        num_arrivals = sum([h.get_num_arrivals() for h in self.hex_zones])
        num_long_wait_pass = sum([h.get_num_longwait_pass() for h in self.hex_zones])
        num_served_pass = sum([h.get_num_served_pass() for h in self.hex_zones])
        num_idle_vehs = len([veh for h in self.hex_zones for veh in h.vehicles.values() if veh.state.status == status_codes.V_IDLE])
        num_assign_vehs = len([veh for h in self.hex_zones for veh in h.vehicles.values() if veh.state.status == status_codes.V_ASSIGNED])
        total_vehs=len([veh for h in self.hex_zones for veh in h.vehicles.values()])
        return [self.num_matches, num_arrivals, num_long_wait_pass, num_served_pass, num_idle_vehs,num_assign_vehs,total_vehs]

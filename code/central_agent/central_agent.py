from novelties import status_codes
from simulator.settings import FLAGS
from simulator.models.vehicle.vehicle_repository import VehicleRepository
from simulator.models.customer.customer_repository import CustomerRepository
from simulator.models.charging_pile.charging_pile_repository import ChargingRepository
from common.geoutils import great_circle_distance
from novelties.pricing.price_calculator import calculate_price
# from tools.read_ev_station import get_ev_station

class Central_Agent(object):


    def __init__(self, matching_policy):
        self.matching_policy = matching_policy

    def get_match_commands(self, current_time, vehicles, requests,charging_stations):
        matching_commands = []
        if len(requests) > 0:
            # if FLAGS.enable_pooling:
            #     matching_commands = self.matching_policy.match_RS(current_time, vehicles, requests)
            # else:
            matching_commands = self.matching_policy.match_requests(current_time, vehicles, requests)
        if len(charging_stations)>0:
            charging_commands = self.matching_policy.match_charging_stations(current_time,vehicles,charging_stations)
            
        m_commands,c_commands = self.init_price(matching_commands,charging_commands)
            # V = defaultdict(list)
            # for command in matching_commands:
            #     # print(command)
            #     V[command["vehicle_id"]].append(command["customer_id"])
            #
            # for k in V.keys():
            #     print(k, "Customers :", V[k])

        vehicles = self.update_vehicles(vehicles, m_commands,c_commands) # here: matched veh --> assigned, to charged.

        return m_commands,c_commands, vehicles


    def update_vehicles(self, vehicles, matching_commands,charging_commands):
        vehicle_ids = [command["vehicle_id"] for command in matching_commands]
        vehicles.loc[vehicle_ids, "status"] = status_codes.V_ASSIGNED
        
        charging_vehicle_ids = [command["vehicle_id"] for command in charging_commands]
        vehicles.loc[charging_vehicle_ids, "status"] = status_codes.V_WAYTOCHARGE
        
        return vehicles


    def init_price(self, match_commands,charging_commands):
        m_commands = []
        for m in match_commands:
            vehicle = VehicleRepository.get(m["vehicle_id"])
            # vehicle.state.status = status_codes.V_ASSIGNED
            if vehicle is None:
                print("Invalid Vehicle id")
                continue
            customer = CustomerRepository.get(m["customer_id"])
            if customer is None:
                print("Invalid Customer id")
                continue

            triptime = m["duration"]

            # if FLAGS.enable_pricing:
            dist_for_pickup = m["distance"]
            dist_till_dropoff = great_circle_distance(customer.get_origin()[0], customer.get_origin()[1],
                                          customer.get_destination()[0], customer.get_destination()[1])
            total_trip_dist = dist_for_pickup + dist_till_dropoff
            [travel_price, wait_price] = vehicle.get_price_rates()
            # v_cap = vehicle.state.current_capacity
            initial_price = calculate_price(total_trip_dist, triptime, vehicle.state.mileage, travel_price,
                                wait_price, vehicle.state.gas_price, vehicle.state.driver_base_per_trip)
            m["init_price"] = initial_price
            m_commands.append(m)
        
        # matching_charging_station
        c_commands = [] # charging command
        for c in charging_commands:
            vehicle = VehicleRepository.get(c["vehicle_id"])
            if vehicle is None:
                print("Invalid Vehicle id")
                continue
            charging_pile = ChargingRepository.get_charging_station(c["customer_id"])

            # charging_pile = df_charging_piles["customer_id"] # return to that row
            if charging_pile is None:
                print("Invalid Customer id")
                continue
            charging_incentives = 0 # charging_pile.get_incentive()

            triptime = c["duration"]
            # if FLAGS.enable_pricing:
            dist_to_cs = c["distance"]
            total_trip_dist = dist_to_cs
            [travel_price, wait_price] = vehicle.get_price_rates()
            # v_cap = vehicle.state.current_capacity
            initial_price = calculate_price(total_trip_dist, triptime, vehicle.state.mile_of_range, travel_price,
                                wait_price, vehicle.state.full_charge_price, vehicle.state.driver_base_per_trip)
            c["init_price"] = initial_price + charging_incentives
            c_commands.append(c)

        return m_commands,c_commands
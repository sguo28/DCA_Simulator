import argparse
import glob
import os.path

from common.time_utils import get_local_datetime
from config.hex_setting import HEX_SHP_PATH, CS_SHP_PATH, NUM_NEAREST_CS, TRIP_FILE, TRAVEL_TIME_FILE, CS_DATA_PATH,\
    TIMESTEP, START_OFFSET, SIM_DAYS, START_TIME, OPTION_DIM, LEARNING_RATE, RELOCATION_DIM, CHARGE_ACCELERATOR,N_EPISODE,\
    TIME_FOR_REPAIR,EVCS_EV_TRANSMISSION_RATE, EV_EVCS_TRANSMISSION_RATE, CMS_TRANSMISSION_RATE, CMS_EVCS_TRANSMISSION_RATE,\
    NUM_SEEDS
from novelties.config_parser import config_parser # organize the config file of different scenarios
from simulator.simulator_cnn import Simulator
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# ---------------MAIN FILE---------------
def main(args,HEX_SHP_PATH,CS_DATA_PATH, NUM_NEAREST_CS, TRIP_FILE, TRAVEL_TIME_FILE,TIMESTEP, START_OFFSET, SIM_DAYS,
         START_TIME, RELOCATION_DIM):

    if SIM_DAYS > 0:
        start_time = START_TIME + int(60 * 60 * 24 * START_OFFSET)
        print("Simulate Episode Start Datetime: {}".format(get_local_datetime(start_time)))
        end_time = start_time + int(60 * 60 * 24 * SIM_DAYS)
        print("Simulate Episode End Datetime : {}".format(get_local_datetime(end_time)))

        config = config_parser(args) # contains false_alarm_rate, penalty_charging_duration, attack_type
        ttrial = int(args.trial)
        if args.test:
            root_path0 = "test_ieee"
        else:
            root_path0 = "ieee"

        list_of_sensitivity_level = config["SENSITIVITY_LEVEL_GRID_SEARCH"]

        for sensitivity_level in list_of_sensitivity_level:
            print(f"SENSITIVITY_LEVEL: {sensitivity_level}, DETECTION_TECHNIQUE: {config['DETECTION_TECHNIQUE']}")
            config["SENSITIVITY_LEVEL"] = sensitivity_level
            root_path = f"tf_logs/{root_path0}/N_{args.n_vehicles}_A_{args.attack_type}_T_{args.penalty_charging_duration}_D_{int(args.use_detector)}_DT_{args.detection_technique}_S_{sensitivity_level}/"
            root_save_path = f"logs/{root_path0}/"
            if os.path.exists(root_save_path) is False:
                os.mkdir(root_save_path)

            if os.path.exists(root_path) is False:
                os.mkdir(root_path)
            completed_seeds = len(glob.glob(root_path + "*")) if not args.test else 0 # e.g., if finished 1, then completed_seeds = 1, start from 2
            print(f"Completed Seeds: {completed_seeds}")
            if completed_seeds >= NUM_SEEDS:
                print("All seeds are completed, exit")
                continue
            START_SEED = completed_seeds # index starts from 0

            simulator = Simulator(start_time, TIMESTEP, config) # option, local, charging indicators are embedded in the config file
            simulator.init(HEX_SHP_PATH, TRIP_FILE, CS_DATA_PATH, TRAVEL_TIME_FILE, NUM_NEAREST_CS)
            for seeds in range(START_SEED,NUM_SEEDS):  # 5 randmon seeds
                trial = ttrial + seeds
                n_steps = int(3600 * 24 / TIMESTEP)  # number of time ticks per day
                save_path = f"{args.n_vehicles}_{args.attack_type}_{int(args.use_detector)}_{args.penalty_charging_duration}_{args.detection_technique}_{sensitivity_level}"
                writer = SummaryWriter(root_path)

                with open(f'{root_save_path}/parsed_results_trial_{trial}_{save_path}.csv','w') as f,\
                        open(f'{root_save_path}/parsed_results_system_trial_{trial}_{save_path}.csv','w') as f1,\
                        open(f'{root_save_path}/parsed_results_veh_status_trial_{trial}_{save_path}.csv','w') as f2,\
                        open(f'{root_save_path}/parsed_results_cyberattack_status_trial_{trial}_{save_path}.csv','w') as f3,\
                        open(f'{root_save_path}/parsed_results_repetitive_attack_trial_{trial}_{save_path}.csv','w') as f4,\
                        open(f'{root_save_path}/target_charging_stations_trial_{trial}_{save_path}.csv','w') as g, \
                        open(f'{root_save_path}/training_hist_trial_{trial}_{save_path}.csv','w') as h, \
                        open(f'{root_save_path}/demand_supply_gap_trial_{trial}_{save_path}.csv','w') as l1, \
                        open(f'{root_save_path}/cruising_od_trial_{trial}_{save_path}.csv','w') as m1, \
                        open(f'{root_save_path}/matching_od_trial_{trial}_{save_path}.csv','w') as n1, \
                        open(f'{root_save_path}/charging_od_trial_{trial}_{save_path}.csv','w') as c1:
                    # record system dynamics
                    f.writelines('''time,num_idle,num_serving,num_charging,num_cruising,num_assigned,num_waitpile,num_tobedisptached,num_waytocharge,average_idle_time,num_matches,longwait_pass,served_pass,removed_pass,consumed_SOC_per_cycle,total_system_revenue\n''')
                    # record EV and EVSE status:
                    f1.writelines('''time,revenue_per_veh,fulfillment_rate,occupancy_rate,n_matches\n''')
                    f2.writelines('''time,num_idle,num_cruising,num_occupied,num_assigned,num_waytocharge,num_charging,num_waitpile,num_tobedispatched\n''')
                    f3.writelines(('''time,ev0,ev1,evcs0,evcs1,evcs2,success_detection,false_alarm,fail_to_detect,true_negative\n'''))
                    f4.writelines('''time,cs_id,times_attack\n''')
                    # record destination charging stations
                    g.writelines('''tick,cs_id,destination_cs_id\n''')
                    # record training dynamics
                    h.writelines('''step,loss,reward,learning_rate,sample_reward,sample_SOC\n''')
                    # record supply-demand gap
                    l1.writelines('''step,hex_zone_id,demand_supply_gap\n''')
                    # record cruising trip
                    m1.writelines('''vehicle_id,tick,origin_hex,destination_hex,dist,time,waiting_time,action_id\n''')
                    # record matching trip
                    n1.writelines('''vehicle_id,tick,origin_hex,destination_hex,dist,time,waiting_time,action_id\n''')
                    # record charging information
                    c1.writelines('''vehicle_id,charging_duration,queuing_time,veh_infected,cs_infected,tick,cs_id,start_SOC,end_SOC,status\n''')

                    for episode in range(N_EPISODE):
                        simulator.reset(start_time = episode * (end_time - start_time), timestep=TIMESTEP, seed=trial) # use the weekday data.
                        for day in range(SIM_DAYS): # usually > 7 days.
                            print(f"############################ DAY {day} of {SIM_DAYS} SUMMARY ################################")
                            # need to reset attack status for charging station here.
                            simulator.attach_attack_status(day) # reset attack status for charging station
                            simulator.reload_demand(trial)
                            for i in range(n_steps):
                                tick = simulator.get_current_time()
                                local_state_batches, num_valid_relos, assigned_option_ids, empty_flag = simulator.get_local_states()

                                if not empty_flag:
                                    # local_state_batches, num_valid_relos, assigned_option_ids = simulator.get_local_states()
                                    # selected action is only used for training DQN
                                    # action to execute is to actually implement.
                                    action_selected = np.random.randint(RELOCATION_DIM, size=len(local_state_batches),dtype=int) # randomly select an action for psuedo-training.
                                    action_to_execute = np.random.randint(RELOCATION_DIM, size=len(local_state_batches),dtype=int)  # since we don't have options here,randomly select one.
                                    simulator.attach_actions_to_vehs(action_selected, action_to_execute)

                                simulator.step()
                                simulator.update()  # update time, get metrics.

                                (veh_count_by_status, n_matches,average_idle_time, average_reduced_SOC,
                                 total_passengers, total_num_longwait_pass, total_num_served_pass,
                                 average_cumulated_earning, veh_0,veh_1,evcs_0,evcs_1,evcs_2, detector_metrics,
                                 occupancy_rates, queuing_time) \
                                    = simulator.summarize_metrics(g, m1, n1,c1,f4, day)

                                writer.add_scalar('cyberattack_status/ev_S', veh_0,global_step=tick//60)
                                writer.add_scalar('cyberattack_status/ev_I', veh_1,global_step=tick//60)
                                writer.add_scalar('cyberattack_status/evcs_S', evcs_0, global_step=tick // 60)
                                writer.add_scalar('cyberattack_status/evcs_I', evcs_1, global_step=tick // 60)
                                writer.add_scalar('cyberattack_status/evcs_R', evcs_2, global_step=tick // 60)

                                writer.add_scalar('detection_rate/success_detection',detector_metrics[0],global_step=tick//60)
                                writer.add_scalar('detection_rate/false_alarm',detector_metrics[1],global_step=tick//60)
                                writer.add_scalar('detection_rate/fail_to_detect',detector_metrics[2],global_step=tick//60)

                                f1.writelines(f"{tick},{average_cumulated_earning},{total_num_served_pass/(total_passengers+1)},{(veh_count_by_status[2])/args.n_vehicles},{n_matches}\n")

                                f3.writelines(f"{tick},{veh_0},{veh_1},{evcs_0},{evcs_1},{evcs_2},{detector_metrics[0]},{detector_metrics[1]},{detector_metrics[2]},{detector_metrics[3]}\n")


                                simulator.reset_storage()
                        print(f"############################ DAY {day} of {SIM_DAYS} END ################################")
                        simulator.reset_storage()
                writer.close()

if __name__ == '__main__':
    # initial parameters
    arg = argparse.ArgumentParser("Start running")
    arg.add_argument("--islocal", "-l", default=1, type=bool, help="choose local matching instead of global matching")
    arg.add_argument("--isoption", "-o", default=0, type=bool, help="set number of options")
    arg.add_argument("--ischarging", "-c", default=1, type=bool, help="choose charging option or not")
    arg.add_argument("--trial", "-t", default=0, type=int, help="sequence of running")
    arg.add_argument("--penalty_charging_duration", "-pc", default=0, type=float, help="delay of charging time")
    arg.add_argument("--attack_type", "-a", default=0, type=int, help="if launch attack or not")
    arg.add_argument("--false_alarm_rate", "-far", default=0.05, type=float, help="false alarm rate")
    arg.add_argument("--evcs2ev", "-cs2ev", default=0.05, type=float, help="transmission rate between EVCS and EV")
    arg.add_argument("--evcs2cms", "-cs2cms", default=0.01, type=float, help="transmission rate between EVCS and CMS")
    arg.add_argument("--MTTR", "-r", default=120, type=float, help="mean time to repair")
    arg.add_argument("--use_detector", "-d", default=0, type=int, help="detect or not")
    arg.add_argument("--n_vehicles", "-n_veh", default=1400, type=int, help="number of vehicles")
    arg.add_argument("--n_evse", "-n_evse", default=int(1.0), type=float, help="number of EVSE amplification")
    arg.add_argument("--demand_scale", "-ds", default=0.25, type=float, help="demand scale")

    arg.add_argument("--detection_technique", "-dt", default=0, type=int, help="detection technique")
    arg.add_argument("--sensitivity_level", "-sl", default=0.05, type=float, help="sensitivity level")
    arg.add_argument("--start_day_of_attack", "-sdoa", default=1, type=int, help="start day of attack")

    arg.add_argument("--test", "-test", default=0, type=bool, help="test or not")

    args = arg.parse_args()
    # start simulation.
    main(args,HEX_SHP_PATH,CS_DATA_PATH,  NUM_NEAREST_CS, TRIP_FILE, TRAVEL_TIME_FILE,TIMESTEP, START_OFFSET, SIM_DAYS,
         START_TIME, RELOCATION_DIM)
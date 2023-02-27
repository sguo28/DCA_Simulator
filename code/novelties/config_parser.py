def config_parser(args):
    '''
    :param args: information in argparser (input)
    :return:
    '''
    detection_technique_code = {0:"IF", 1:"KL_div", 2:"Kmeans", 3:"MixGaussian", 4:"Spectral"}
    sensitivity_level_grid_search = {0:[0.15], # checked! # higher: more outliers
                                     1:[0.8], # checked! # higher: less outliers
                                     2:[round(i,2) for i in [0.2]],
                                     3:[0.25,0.35,0.45], #[0.45], # higher: more outliers
                                     4: [0.3,0.4,0.5,0.6,0.7,0.8]} # checked! # higher: less outliers

    test = args.test
    config = {} # initialize a dictionary
    config["PENALTY_CHARGING_DURATION"] = args.penalty_charging_duration
    config["ATTACK_TYPE"] = args.attack_type
    config["FALSE_ALARM_RATE"] = args.false_alarm_rate
    config["EVCS_EV_TRANSMISSION_RATE"] = args.evcs2ev
    config["CMS_EVCS_TRANSMISSION_RATE"] = args.evcs2cms
    config["TIME_FOR_REPAIR"] = args.MTTR
    config["USE_DETECTOR"] = args.use_detector
    config["N_VEHICLES"] = args.n_vehicles if not test else 100
    config["N_EVSE_AMPLIFICATION"] = args.n_evse if not test else 0.5
    config["DEMAND_SCALE"] = args.demand_scale if not test else 0.1

    print(f"Test: {bool(args.test)}, we have {config['N_VEHICLES']} vehicles and {config['N_EVSE_AMPLIFICATION']} EVSEs")
    config["IS_LOCAL"] = "l" if args.islocal else "nl"
    config["IS_OPTION"] = "o" if args.isoption else "no"
    config["IS_CHARGING"] = "c" if args.ischarging else "nc"
    config["DETECTION_TECHNIQUE"] = detection_technique_code[int(args.detection_technique)]
    config["SENSITIVITY_LEVEL"] = args.sensitivity_level
    config["START_DAY_OF_ATTACK"] = args.start_day_of_attack
    config["SENSITIVITY_LEVEL_GRID_SEARCH"] = sensitivity_level_grid_search[int(args.detection_technique)]

    return config



'''
arg.add_argument("--evcs2ev", "-ev2cs", default=0.05, type=float, help="transmission rate between EVCS and EV")
    arg.add_argument("--evcs2cms", "-cs2cms", default=0.01, type=float, help="transmission rate between EVCS and CMS")
    arg.add_argument("--MTTR", "-r", default=30, type=float, help="mean time to repair")
    
'''

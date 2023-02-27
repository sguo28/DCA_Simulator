import pickle
from simulator.services.osrm_engine import OSRMEngine
import pandas as pd
import ast


def get_fhv_routes(df):
    '''
    input file is preprocessed data from FHV trajectory data
    input data's info: column: driver_id, sequence of epoch (datetime) and coord (list of list).
    to store OSRM info (duration, distance, trajectory) to dict
    the OSRM info: a dict with key being (driver_id, day, shift), with value being per step duration, disctance, and coord
                    id is the unique driver id, d is day of the month, shift considers online/offline transition.
                    per step duration/distance/coord is the intersection on the road
    :return:  no return
    '''
    engine = OSRMEngine()
    """

    :param engine: OSRM engine
    
    :param datapath: location of preprocessed data
    :return:
    """
    coord_ods = []
    df['od_coord'] = df['od_coord']  # .apply(ast.literal_eval) # string to list
    serving_ods = df['od_coord'].to_list()  # coord
    results = engine.route_hex(serving_ods, decode=True, annotations=True)

    results_df = pd.DataFrame(results)
    results_df.columns = ['trajectory', 'distance', 'duration','node_id','precise_speed','precise_duration','precise_distance']

    return results_df


if __name__ == '__main__':
    day_list = [('04-16-2017',),
                ('04-20-2017',),
                ('04-22-2017',),
                ('04-18-2017',),
                ('04-12-2017',),
                ('04-24-2017',),
                ('04-19-2017',),
                ('04-21-2017',),
                ('04-17-2017',),
                ('04-23-2017',)]
    for i in range(len(day_list)):
        df = pd.read_json('/home/sguo/workspace/nyc_fhv_trajectories/data/serving_od_coord_{}.json'.format(day_list[i][0]))
        fhv_route_df = get_fhv_routes(df)
        combined_df = pd.concat([df[['vehicle_id', 'od_epoch', 'od_coord']], fhv_route_df], axis=1)
        combined_df.to_json('/home/sguo/workspace/nyc_fhv_trajectories/data/combined_serving_od_coord_{}.json'.format(day_list[i][0]))
        print('processed data on {}'.format(day_list[i][0]))

    # df = pd.read_json('/home/sguo/workspace/nyc_fhv_trajectories/data/serving_od_coord_test.json')
    # fhv_route_df = get_fhv_routes(df)
    # combined_df = pd.concat([df[['vehicle_id', 'od_epoch', 'od_coord']], fhv_route_df], axis=1)
    # combined_df.to_json('/home/sguo/workspace/nyc_fhv_trajectories/data/combined_serving_od_coord_test.json')
    # print('processed data on {}'.format(day_list[i][0]))

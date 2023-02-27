import numpy as np
import sys
import pickle
import os
from scipy.spatial import KDTree
import geopandas as gpd
import pandas as pd
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
# import pandas as pd
sys.path.append('/../data')
# sys.path.append('C:/Users/17657/PycharmProjects/Deep_Pool')
from simulator.services.osrm_engine import OSRMEngine


#### NOTICE
# I replace the warning which raises as show 'No Routes found' by returning None, to make sure the for loop can continued.
# check the package: "/home/sguo/anaconda3/envs/myenv/lib/python3.7/site-packages/osrm.py", line 142

def create_routes_between_hexes(engine,n_charge=5):
    '''
    We extract only the feasible od pairs under one of the three conditions:
    1. Neighboring zones of each zone
    2. Pairwise zones within each matching zone
    3. Each hex zone to each charging zone.
    :param engine:
    :param hex_coords:
    :return:
    '''
    routes = dict()
    od_pairs=[]
    df = gpd.read_file('../../data/NYC_shapefiles/snapped_clustered_hex.shp')  # NYC_shapefiles/test_feasible_hex.shp')
    df['m_id'] = 0 # mask the cluster label
    # df = df.set_index('hex_id')
    # print(df.index.to_numpy()[-1])
    hex_ids=df.index.to_list()
    hex_coords = df[['snap_lon', 'snap_lat']].to_dict()  # coord
    hex_coords_list = df[['snap_lon', 'snap_lat']].to_numpy()
    # hex_coords = pd.Series([df.snap_lon, df.snap_lat], index=df.index).to_dict()
    hex_to_match = df['m_id'].to_numpy()  # corresponded match zone id

    charging_stations = gpd.read_file('../../data/NYC_shapefiles/cs_snap_concat.shp')  # point geometry # 'data/NYC_shapefiles/processed_cs.shp'
    charging_stations = charging_stations[charging_stations['hex_id'].isin(hex_ids)]
    # print(charging_stations[charging_stations.hex_id == 1325])
    # num	lat	lon	hex_id	type	geometry
    charging_hex_ids=charging_stations['hex_id'].to_numpy()
    charging_kdtree = KDTree(charging_stations[['lon', 'lat']])

    #add od pairs between each hex and [neighboring zones + nearest 5 charging stations]
    for h_idx, coords, match_id in zip(hex_ids,hex_coords_list,hex_to_match):
        neighbors = df[df.geometry.touches(df.geometry[h_idx])].index.tolist()  # len from 0 to 6
        _, charging_idx = charging_kdtree.query(coords, k=n_charge)  # charging station id
        nearest_cs_hex_ids = charging_hex_ids[charging_idx]
        for target_hid in neighbors+nearest_cs_hex_ids.tolist()+[h_idx]:
            if (h_idx,target_hid) not in routes.keys():
                routes[(h_idx,target_hid)]=dict()
                # print(h_idx, target_hid)
                od_pairs.append([[hex_coords['snap_lon'][h_idx],hex_coords['snap_lat'][h_idx]],
                [hex_coords['snap_lon'][target_hid], hex_coords['snap_lat'][target_hid]]]
                )


    # pairwise distance inside each matching zone
    for i in np.unique(hex_to_match):
        tdf=df[df['m_id']==i]
        for h1 in tdf.index.tolist():
            for h2 in tdf.index.tolist():
                if (h1, h2) not in routes.keys():
                    routes[(h1, h2)] = dict()
                    od_pairs.append([[hex_coords['snap_lon'][h1],hex_coords['snap_lat'][h1]],
                                     [hex_coords['snap_lon'][h2],hex_coords['snap_lat'][h2]]])

    # od pair for all possible trips
    with open('../../data/trip_od_hex.csv', 'r') as f:
        next(f)
        for lines in f:
            line = lines.strip().split(',')
            h, o, d, t = line[1:]  # hour, oridin, dest, trip_time/num of trip
            o,d=int(o),int(d)
            if (o, d) not in routes.keys():
                if o in hex_ids and d in hex_ids:
                    routes[(o, d)] = dict()
                    od_pairs.append([[hex_coords['snap_lon'][o],hex_coords['snap_lat'][o]],
                                         [hex_coords['snap_lon'][d],hex_coords['snap_lat'][d]]])

    # od_pairs=od_pairs[:100]

    print('number of od pairs to query: {}'.format(len(od_pairs)), len(routes.keys()))
    if (382,467) in routes.keys():
        print('routes included!!!')
    else:
        print('routes not included!!')


    results= engine.route_hex(od_pairs, decode=False) #results is a list of [trajectory, time, distance]

    for r,key in zip(results,routes.keys()):
        routes[key]['route']=r[0]; routes[key]['travel_time']=r[1]; routes[key]['distance']=r[2]

    return routes




# to test funstions
if __name__ == '__main__':
    #
    # tt_map = pickle.load(open(os.path.join('../data/hex_tt_map.pkl'), 'rb'))
    # print(len(tt_map))
    # x,y,z= engine.route_hex([[[-73.776954, 40.758245],[-73.920914, 40.817326]]])[0]
    #
    # print(x,y,z)
    #
    engine = OSRMEngine()
    routes = create_routes_between_hexes(engine)
    with open("../../data/all_routes.pkl", "wb") as f:
        pickle.dump(routes, f)

    print('data processing completed')

import numpy as np
import argparse
import sys
import pickle
import os
from scipy.spatial import KDTree
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
import geopandas as gpd
# import pandas as pd
sys.path.append('/../data')
# sys.path.append('C:/Users/17657/PycharmProjects/Deep_Pool')
from simulator.services.osrm_engine import OSRMEngine

# state_space = [(x, y) for x in range(MAP_WIDTH) for y in range(MAP_HEIGHT)]
# action_space = [(ax, ay) for ax in range(-MAX_MOVE, MAX_MOVE + 1)
#                 for ay in range(-MAX_MOVE, MAX_MOVE + 1)]

#
# def create_reachable_map(engine):
#     lon0, lat0 = convert_xy_to_lonlat(0, 0)
#     lon1, lat1 = convert_xy_to_lonlat(1, 1)
#     d_max = great_circle_distance(lat0, lon0, lat1, lon1) / 2.0
#
#     points = []
#     for x, y in state_space:
#         lon, lat = convert_xy_to_lonlat(x, y)
#         points.append((lat, lon))
#
#     nearest_roads = engine.nearest_road(points)
#     reachable_map = np.zeros((MAP_WIDTH, MAP_HEIGHT), dtype=np.float32)
#     for (x, y), (latlon, d) in zip(state_space, nearest_roads):
#         if d < d_max:
#             reachable_map[x, y] = 1
#
#     return reachable_map

#
# def create_tt_tensor(engine, reachable_map):
#     origin_destins_list = []
#     for x, y in state_space:
#         origin = convert_xy_to_lonlat(x, y)[::-1]
#         destins = [convert_xy_to_lonlat(x + ax, y + ay)[::-1] for ax, ay in action_space]
#         origin_destins_list.append((origin, destins))
#     tt_list = engine.eta_one_to_many(origin_destins_list)
#
#     a_size = MAX_MOVE * 2 + 1
#     tt_tensor = np.full((MAP_WIDTH, MAP_HEIGHT, a_size, a_size), np.inf)
#     for (x, y), tt in zip(state_space, tt_list):
#         tt_tensor[x, y] = np.array(tt).reshape((a_size, a_size))
#         for ax, ay in action_space:
#             x_, y_ = x + ax, y + ay
#             axi, ayi = ax + MAX_MOVE, ay + MAX_MOVE
#             if x_ < 0 or x_ >= MAP_WIDTH or y_ < 0 or y_ >= MAP_HEIGHT or reachable_map[x_, y_] == 0:
#                 tt_tensor[x, y, axi, ayi] = float('inf')
#         # if reachable_map[x, y] == 1:
#         #     tt_tensor[x, y, MAX_MOVE, MAX_MOVE] = 0
#     tt_tensor[np.isnan(tt_tensor)] = float('inf')
#     return tt_tensor
#
#
# def create_routes(engine, reachable_map):
#     routes = {}
#     for x, y in state_space:
#
#         print(x, y)
#         #if(x ==0 and y==14):
#         #    continue
#
#         origin = convert_xy_to_lonlat(x, y)[::-1]
#         od_list = [(origin, convert_xy_to_lonlat(x + ax, y + ay)[::-1]) for ax, ay in action_space]
#         tr_list, _ = zip(*engine.route(od_list, decode=False))
#         routes[(x, y)] = {}
#         for a, tr in zip(action_space, tr_list):
#             routes[(x, y)][a] = tr
#     return routes
# od_pairs = [[origin_coord, destination_coord] for idx, origin_coord in enumerate(hex_coords) for destination_coord in
#             hex_coords[0:idx]]
# od_hex_ids = [(origin_id, destination_id) for idx, origin_id in enumerate(hex_ids) for destination_id in hex_ids[0:idx]]


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
    charging_stations = gpd.read_file('../../data/NYC_shapefiles/cs_snap_concat.shp')  # point geometry # 'data/NYC_shapefiles/processed_cs.shp'
    charging_zones=charging_stations['hex_id'].tolist()
    charging_kdtree = KDTree(charging_stations[['lon', 'lat']])
    df = gpd.read_file('../../data/NYC_shapefiles/tagged_clustered_hex.shp')
    hex_ids=df.index.tolist()
    hex_coords = df[['lon', 'lat']].to_numpy()  # coord
    hex_to_match = df['cluster_la'].to_numpy()  # corresponded match zone id

    #add od pairs between each hex and [neighboring zones + nearest 5 charging stations]
    for h_idx, coords, match_id in zip(hex_ids,hex_coords,hex_to_match):
        neighbors = df[df.geometry.touches(df.geometry[h_idx])].index.tolist()  # len from 0 to 6
        _, charging_idx = charging_kdtree.query(coords, k=n_charge)  # charging station id
        for n in neighbors+charging_idx.tolist()+[h_idx]:
            if (h_idx,n) not in routes.keys():
                routes[(h_idx,n)]=dict()
                od_pairs.append([hex_coords[h_idx],hex_coords[n]])


    # pairwise distance inside each matching zone
    for i in np.unique(hex_to_match):
        tdf=df[df['cluster_la']==i]
        for h1 in tdf.index.tolist():
            for h2 in tdf.index.tolist():
                if (h1, h2) not in routes.keys():
                    routes[(h1, h2)] = dict()
                    od_pairs.append([hex_coords[h1], hex_coords[h2]])

    # od pair for all possible trips
    with open('../../data/trip_od_hex.csv', 'r') as f:
        next(f)
        for lines in f:
            line = lines.strip().split(',')
            h, o, d, t = line[1:]  # hour, oridin, dest, trip_time/num of trip
            o,d=int(o),int(d)
            if (o, d) not in routes.keys():
                routes[(o, d)] = dict()
                od_pairs.append([hex_coords[o], hex_coords[d]])

    print('number of od pairs to query: {}'.format(len(od_pairs)), len(routes.keys()))
    if (382,467) in routes.keys():
        print('routes included!!!')
    else:
        print('routes not included!!')


    results= engine.route_hex(od_pairs, decode=False) #results is a list of [trajectory, time, distance]

    for r,key in zip(results,routes.keys()):
        routes[key]['route']=r[0]; routes[key]['travel_time']=r[1]; routes[key]['distance']=r[2]

    return routes


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument("data_dir", help = "data directory")
#     # parser.add_argument("--route", action='store_true', help="whether compute route or not")
#     args = parser.parse_args()
#
#     engine = OSRMEngine()
#
#     print("create reachable map")
#     reachable_map = create_reachable_map(engine)
#     np.save("{}/reachable_map".format(args.data_dir), reachable_map)
#
#     print("create tt map")
#     tt_tensor = create_tt_tensor(engine, reachable_map)
#     np.save("{}/tt_map".format(args.data_dir), tt_tensor)
#
#     print("create routes")
#     # if args.route:
#     routes = create_routes(engine, reachable_map)
#     pickle.dump(routes, open("{}/routes.pkl".format(args.data_dir), "wb"))


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
    with open("../../data/hex_routes.pkl", "wb") as f:
        pickle.dump(routes, f)

    print('data processing completed')


import csv
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from common.mesh import convert_lonlat_to_xy, convert_xy_to_lonlat
from novelties import status_codes
# from tools.read_ev_station import get_processed_charging_piles
    
def get_charging_piles():
    chargingpiles = csv.DictReader(open('data/processed_cs.csv'))
    list_ev_chargers = pd.DataFrame(chargingpiles)

    r_latlon = list_ev_chargers[["Longitude","Latitude"]].apply(pd.to_numeric)

    # print(list_ev_chargers.iloc[[2]].ZIP)
    c_latlon = pd.DataFrame()
    
    for r_id, row in r_latlon.iterrows():
        try: 
            x,y = convert_lonlat_to_xy(row.Longitude, row.Latitude)
            c_lon, c_lat = convert_xy_to_lonlat(x,y)
            for _ in range(int(list_ev_chargers.iloc[[r_id]].EV_Level2)):
                c_latlon = c_latlon.append([[c_lon,c_lat,status_codes.SP_LEVEL2,status_codes.CP_AVAILABLE,0.0]])
            for _ in range(int(list_ev_chargers.iloc[[r_id]].EV_DC_Fast)):
                c_latlon = c_latlon.append([[c_lon,c_lat,status_codes.SP_DCFC,status_codes.CP_AVAILABLE,0.0]])
        except:
            ValueError
    c_latlon.columns = ["c_lon", "c_lat","type","flag","incentive"]
    c_latlon.index = [i for i in range(len(c_latlon.index))]
    return c_latlon


# get_charging_piles().to_json("../data/processd_cp.json")

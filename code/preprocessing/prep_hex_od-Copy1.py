import pandas as pd
import geopandas as gpd

from scipy.spatial import KDTree


# first get cooresponding HEX_ID of origin and destination
taxi_records = pd.read_csv('data/trip_records/hex_trips_2016-05.csv')
# taxi_gdf_o = gpd.GeoDataFrame(taxi_records,geometry=gpd.points_from_xy(taxi_records.origin_lon,taxi_records.origin_lat))

polygons = gpd.read_file('data/NYC_shapefiles/reachable_hexes.shp') # or insert the full-action-space hex shapefile and only select 'lon' != -1

# polygons = polygons.to_crs({'init':'epsg:4326'}) # lon,lat unit: feet to degree

polygons['CELL_ID']=polygons.index

hex_tree = KDTree(polygons[['lon','lat']]) # hex's centroid lon, lat
_,ohex_ids = hex_tree.query(taxi_records[['origin_lon','origin_lat']])
_,dhex_ids = hex_tree.query(taxi_records[['destination_lon','destination_lat']])
taxi_records['o_hex_id'] = ohex_ids
taxi_records['d_hex_id'] = dhex_ids
start_time = taxi_records['request_datetime'].iloc[0]
end_time = taxi_records['request_datetime'].iloc[-1]
# print('start time',start_time)
# print('request time', taxi_records['request_datetime'])
taxi_records['hour'] = ((taxi_records['request_datetime'] - start_time)//(60*60))%24
# time period is the time horizon for our study, e.g., 30 days.
time_period = (end_time- start_time)//(60*60*24)
print('time period',time_period,end_time, start_time)
within_trip_od = taxi_records[['hour','o_hex_id','d_hex_id','trip_time']]

trip_count = within_trip_od.groupby(['hour','o_hex_id','d_hex_id'])['trip_time'].count().reset_index() # here we count trip time to see the number of incurred trips. 
trip_count.columns = ['h', 'o', 'd', 'n']

trip_count['n'] = trip_count['n']/time_period
print(trip_count['n'].sum())
trip_count.to_csv('data/trip_od_hex.csv',index_label=False)


trip_duration = within_trip_od.groupby(['hour','o_hex_id','d_hex_id'])['trip_time'].mean().reset_index() # here mean is over every od pairs per hour.

trip_duration.columns = ['h', 'o', 'd', 't']
trip_duration['t'] =  trip_duration['t']#/60
trip_duration.to_csv('data/trip_time_od_hex_sec.csv',index_label=False)




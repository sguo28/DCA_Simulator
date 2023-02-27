import pandas as pd
import geopandas as gpd
import numpy as np

# this is grid based process
df = pd.read_csv('data/EV_facility/alt_fuel_stations.csv')
df = df[['Fuel Type Code','State','EV Level2 EVSE Num', 'EV DC Fast Count','Latitude','Longitude']].replace(np.nan, 0)
df = df[df['Fuel Type Code']=='ELEC']
df_ny = df[df['State']=='NY']

gdf_ny = gpd.GeoDataFrame(df_ny,geometry=gpd.points_from_xy(df_ny.Longitude,df_ny.Latitude))

polygons = gpd.GeoDataFrame.from_file('data/NYC_shapefiles/selected_hexagon.shp')

polygons = polygons.to_crs({'init':'epsg:4326'})

pts = gdf_ny.copy()
pts.crs = polygons.crs

pts_within = gpd.sjoin(pts,polygons,how="left", op="within")
pts_within = pts_within[pts_within['Id'].notnull()]
df_within = pd.DataFrame(pts_within)
#df_within.rename(columns={('Fuel Type Code', 'State', 'EV Level2 EVSE Num', 'EV DC Fast Count','Latitude', 'Longitude', 'geometry', 'index_right', 'Id', 'GRID_ID','lat', 'lon'):('Fuel Type Code', 'State', 'EV_Level2', 'EV_DC_Fast','Latitude', 'Longitude', 'geometry', 'index_right', 'Id', 'GRID_ID','lat', 'lon')},inplace = True)
df_within.columns=['Fuel Type Code', 'State', 'EV_Level2', 'EV_DC_Fast','Latitude', 'Longitude', 'geometry', 'index_right', 'Id', 'GRID_ID','lat', 'lon']
df_within.to_csv('data/processed_cs_v2.csv')

import pandas as pd
import geopandas as gpd
from scipy.spatial import KDTree
# df=gpd.read_file('data/NYC_shapefiles/tagged_clustered_hex.shp')
df=gpd.read_file('data/NYC_shapefiles/snapped_clustered_hex.shp')

hxt= KDTree(df[['snap_lon','snap_lat']])
pcs = pd.read_csv('data/snap_processed_cs.csv')

coord = pcs[['Longitude','Latitude']].to_numpy()

_, hex_id = hxt.query(coord)

pcs['hex_id']=hex_id

pcs_true = pcs[['EV_Level2', 'EV_DC_Fast', 'Latitude', 'Longitude','hex_id']]

pcs_true.to_csv('data/cs_true_lonlat.csv',index=False)
pcs[['lon','lat']] = df.loc[hex_id,['snap_lon','snap_lat']]
pcs_snap = pcs[['EV_Level2', 'EV_DC_Fast', 'snap_lon','snap_lat','hex_id']]

# pcs_snap.to_csv('data/cs_snap_lonlat.csv',index=False)


pcs_l2 = pcs_snap.loc[pcs['EV_Level2']!=0]
pcs_l2=pcs_l2[['EV_Level2','snap_lon','snap_lat'  ,'hex_id']]

pcs_l2.insert(pcs_l2.shape[1] ,'type',0)
pcs_dc = pcs_snap.loc[pcs['EV_DC_Fast']!=0]
pcs_dc=pcs_dc[['EV_DC_Fast','snap_lon','snap_lat' ,'hex_id']]

pcs_dc.insert(pcs_dc.shape[1] ,'type',1)
pcs_l2.columns = pcs_dc.columns = ['num','snap_lon','snap_lat' ,'hex_id','type']
new_df = pd.concat([pcs_l2,pcs_dc],axis=0)
new_df.reset_index()

new_df.to_csv('data/process_cs_concat.csv',index=False)

df = pd.read_csv('data/process_cs_concat.csv')
df_1 = gpd.GeoDataFrame(df,geometry= gpd.points_from_xy(df['lon'],df['lat']))
df_1.to_file('data/NYC_shapefiles/cs_snap_concat.shp')

# df=gpd.read_file('data/NYC_shapefiles/processed_cs.shp')
# df.head()
# df = pd.read_csv('data/cs_snap_lonlat.csv')
# df_1 = gpd.GeoDataFrame(df,geometry= gpd.points_from_xy(df['lon'],df['lat']))
# df_1.to_file('data/NYC_shapefiles/cs_snap_lonlat.shp')

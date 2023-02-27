from scipy.spatial import KDTree
import geopandas as gpd

import geopandas as gpd

polygons = gpd.read_file('data/NYC_shapefiles/tagged_clustered_hex.shp') 
polygons = polygons[polygons['tagged_lat']!=-1]
print('reachable_nrow:',polygons.shape[0])

polygons.to_file('data/NYC_shapefiles/reachable_hexes.shp')


'''
reachable_hex = action_space_hex[dist_list<0.0001] # select the hex within distance of 2.5*edge length, unit: meter
rids = reachable_hex.index.tolist()
action_space_hex['tagged'] = 0
action_space_hex.loc[rids,['tagged']] =1
dissolve_hex = action_space_hex.copy()
did_hex = dissolve_hex.dissolve(by= 'tagged')
# full_action_hex = 
did_hex.head

from scipy.spatial import KDTree
import geopandas as gpd

polygons = gpd.read_file('data/NYC_shapefiles/tagged_clustered_hex.shp')
polygons = polygons[polygons['tagged_lat']!=-1]
print('reachable_nrow:',polygons.shape[0])

polygons = polygons.to_crs({'init':'epsg:4326'})
print(polygons.columns)
tagged_hex_kdtree=KDTree(polygons[['lon','lat']])

action_space_hex = gpd.read_file('data/NYC_shapefiles/to_select_action_space.shp')

action_space_hex = action_space_hex.to_crs({'init':'epsg:4326'})

action_space_hex['lon'],action_space_hex['lat']=action_space_hex['geometry'].centroid.x,action_space_hex['geometry'].centroid.y

dist_list,ids = tagged_hex_kdtree.query(action_space_hex[['lon','lat']])

import matplotlib.pyplot as plt

plt.hist(dist_list)
plt.show()
action_space_hex = action_space_hex[dist_list<2*840] # select the hex within distance of 2.5*edge length, unit: meter
action_space_hex[['lat','lon','tagged']][dist_list<0.0001] = -1, -1,0

print(action_space_hex.sort_values('lon'))
'''

'''
nb_list = []
for i in ids[dist_list<0.0001]:
    neighbour_ids = action_space_hex[action_space_hex.geometry.touches(action_space_hex.geometry[i])].index.tolist() 
    nb_list.append(neighbour_ids)
results_union = set().union(*nb_list)
action_space_id = np.unique(list(results_union))
action_space_hex = action_space_hex.iloc[action_space_id]

'''

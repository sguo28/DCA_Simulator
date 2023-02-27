import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns; sns.set()

df = pd.read_csv('data/trip_records/hex_trips_2016-05.csv')
gdf = gpd.GeoDataFrame(df,geometry=gpd.points_from_xy(df.origin_lon,df.origin_lat))

polygons = gpd.GeoDataFrame.from_file('data/NYC_shapefiles/selected_hexagon.shp')

polygons = polygons.to_crs({'init':'epsg:4326'})
taxi_zones = gpd.GeoDataFrame.from_file('data/NYC_shapefiles/taxi_zones.shp')

pts = gdf.copy()
pts.crs = polygons.crs


pts_within = gpd.sjoin(pts,polygons,how="left", op="within")

hex_count = pts_within.groupby('GRID_ID')['id'].count() #.to_csv('data/hex_count.csv')
hex_count.columns = ["GRID_ID","NUM_ORDER"]
polygons = polygons.merge(hex_count,on = "GRID_ID")
polygons.rename(columns={('Id', 'GRID_ID', 'lat', 'lon', 'geometry', 'id'):('Id', 'GRID_ID', 'lat', 'lon', 'geometry', 'num_orders')},inplace = True)
polygons.to_file('data/NYC_shapefiles/hex_num_trips.shp')
print(polygons)

kmeans = KMeans(n_clusters = 12, max_iter=1000, init ='k-means++')
X_weighted = pd.DataFrame(polygons)
lat_long = X_weighted[['lon','lat']]
lot_size = X_weighted.id
weighted_kmeans_clusters = kmeans.fit(lat_long, sample_weight = lot_size) # Compute k-means clustering.
X_weighted['cluster_label'] = kmeans.predict(lat_long, sample_weight = lot_size)

centers = kmeans.cluster_centers_ # Coordinates of cluster centers.

labels = X_weighted['cluster_label'] # Labels of each point

X_weighted.plot.scatter(x = 'lon', y = 'lat', c=labels, s=50, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.title('Clustering GPS Co-ordinates to Form Regions - Weighted',fontsize=18, fontweight='bold')
plt.show()
print(polygons.head)

polygons.to_file('data/NYC_shapefiles/clustered_hex.shp')
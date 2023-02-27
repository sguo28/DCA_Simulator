
import geopandas as gpd
from simulator.services.osrm_engine import OSRMEngine
import os
# sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
# from preprocessing.preprocess_nyc_dataset import extract_bounding_box, BOUNDING_BOX
from shapely.geometry import Point

def create_snapped_hex(df,engine,batch_size=2000):
    nearest_coord = []
    batch_size=min(df.shape[0],batch_size)
    for i in range(0,df.shape[0],batch_size):
        df_i = df.iloc[i:i+batch_size]
        coord = [(lon,lat) for lat, lon in zip(df_i.lat,df_i.lon)]
        nearest_coord +=[loc for loc,_ in engine.nearest_road(coord)]
    
    tagged_lat = []; tagged_lon = []; snapped_lat = []; snapped_lon = []

    nearest_coord_pts = [Point(pt[0],pt[1]) for pt in nearest_coord]
    for poly, points,lon_lat in zip(df.geometry,nearest_coord_pts,nearest_coord):
        if points.within(poly):
            print(poly.centroid)
            tagged_lat.append(lon_lat[1])
            tagged_lon.append(lon_lat[0])
            snapped_lat.append(lon_lat[1])
            snapped_lon.append(lon_lat[0])
        else:
            tagged_lat.append(-1)
            tagged_lon.append(-1)
            snapped_lat.append(lon_lat[1])
            snapped_lon.append(lon_lat[0])

    # df[df['tagged_lat']==-1].lat = df['snapped_lat']
    df['tagged_lat'],df['tagged_lon']=tagged_lat,tagged_lon     
    df['snap_lat'],df['snap_lon']=snapped_lat,snapped_lon
    
    return df

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("input_file", help="input csv file path of ride requests to be map match")
    # parser.add_argument("output_file", help="output csv file path")
    # args = parser.parse_args()
    
    engine = OSRMEngine()

    # df = pd.read_csv(args.input_file, index_col='id')
    # print("load {} rows".format(len(df)))
    # df = create_snapped_trips(df, engine)
    # print("extract {} rows".format(len(df)))
    # df.to_csv(args.output_file)

    df = gpd.read_file('../../data/NYC_shapefiles/tagged_clustered_hex.shp')

    snapped_hex = create_snapped_hex(df,engine)
    snapped_hex.to_file('../../data/NYC_shapefiles/snapped_clustered_hex.shp')
    # selected_hex = snapped_hex['tagged_lon'!=-1]

    # print((snapped_hex[snapped_hex['tagged_coord']!=-1]).shape[0],snapped_hex.shape[0])




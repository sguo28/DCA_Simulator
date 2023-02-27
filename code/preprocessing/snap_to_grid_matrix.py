# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ### Global state wrapper
# This is a wrapper for constructing global state, which consists of number of request, available vehicle, average waiting time per charging station.
#
# ### To-do list
# 1. get the hex panel, mark the (x,y) ID
# 2. merge with the hexagon cells that we use in our study
# 3. get the data for the previous hour, fit to the correspoding hexagon
# 4. dump the hourly pattern as global state, together with local state
#     - get_action()
#     - training
# 5. modify DQN module to accomodate both CNN and FC layers, resemble them together.

import geopandas as gpd
from openpyxl.utils import column_index_from_string

geo_df = gpd.read_file('../../data/NYC_shapefiles/snapped_clustered_hex.shp')


geo_df['col_id'] = geo_df.apply(lambda row: column_index_from_string(row.GRID_ID.split('-')[0]) ,axis=1)


geo_df['row_id'] = geo_df.apply(lambda row: int(row.GRID_ID.split('-')[1]) ,axis=1)


geo_df['col_id'] -= geo_df['col_id'].min()
geo_df['row_id'] -= geo_df['row_id'].min()

geo_df.to_file('../../data/NYC_shapefiles/snapped_clustered_hex.shp')


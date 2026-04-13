import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import shapely.geometry as sgeom
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

# =============================================================================
# 1. I/O DEFINITIONS
# =============================================================================
INPUT_HURDAT_FILE    = 'HURDAT2_all_since_2014_filtered.csv'
INPUT_HRDOBS_DB_FILE = 'hrdobs_inventory_db.csv'
OUTPUT_FIGURE_FILE   = 'hrdobs_figure4.pdf'

# =============================================================================
# 2. CONFIGURATION & STYLING
# =============================================================================

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['font.sans-serif'] = ['Arial Narrow']
mpl.rcParams['font.family'] = 'sans-serif'

LAND_COLOR  = (147/255, 98/255, 38/255)
OCEAN_COLOR = (57/255, 83/255, 165/255)

COLOR_MAP = {
    'TD': (66/255,  199/255, 244/255),
    'TS': (17/255,  170/255, 75/255),
    'H1': (17/255,  170/255, 75/255),
    'H2': (254/255, 230/255, 9/255),
    'H3': (254/255, 230/255, 9/255),
    'H4': (239/255, 61/255,  37/255),
    'H5': (239/255, 61/255,  37/255),
    'EX': (238/255, 193/255, 219/255),
}

DEFAULT_COLOR = (0.5, 0.5, 0.5)

LINE_WIDTH       = 3.0
MARKER_SIZE_PTS  = 0.064 * 72.0  # 4.608 points
MARKER_EDGE_WIDTH = 0.5

EXTENT = [-110, -10, 5, 45]

# =============================================================================
# 3. DATA LOADING & PARSING
# =============================================================================

# --- Track data: full HURDAT2 file ---
hurdat_cols = ['STORM ID', 'NAME', 'YEAR', 'MONTH', 'DAY', 'HOUR',
               'CAT', 'LAT_STR', 'LON_STR', 'INT', 'PRES']

hurdat = pd.read_csv(INPUT_HURDAT_FILE, names=hurdat_cols, header=0)

def parse_lat(val):
    val = str(val).strip()
    return float(val[:-1]) if val[-1] == 'N' else -float(val[:-1])

def parse_lon(val):
    val = str(val).strip()
    return -float(val[:-1]) if val[-1] == 'W' else float(val[:-1])

hurdat['LAT'] = hurdat['LAT_STR'].apply(parse_lat)
hurdat['LON'] = hurdat['LON_STR'].apply(parse_lon)
hurdat['CAT'] = hurdat['CAT'].astype(str).str.strip().str.upper()

unwanted_cats = ['ET', 'LO', 'DB']
hurdat = hurdat[~hurdat['CAT'].isin(unwanted_cats)]

# --- Observation dot data: inventory file ---
inventory = pd.read_csv(INPUT_HRDOBS_DB_FILE)
inventory['LAT'] = pd.to_numeric(inventory['Lat'], errors='coerce')
inventory['LON'] = pd.to_numeric(inventory['Lon'], errors='coerce')
inventory = inventory.dropna(subset=['LAT', 'LON'])

# =============================================================================
# 4. FIGURE & MAP SETUP
# =============================================================================

fig_width_in  = 12.73
fig_height_in = 6.51

plot_width_in  = 11.26
plot_height_in = 5.20
plot_left_in   = 0.896
plot_top_in    = 0.493

plot_bottom_in = fig_height_in - plot_top_in - plot_height_in

left   = plot_left_in / fig_width_in
bottom = plot_bottom_in / fig_height_in
width  = plot_width_in / fig_width_in
height = plot_height_in / fig_height_in

fig = plt.figure(figsize=(fig_width_in, fig_height_in))
ax = fig.add_axes([left, bottom, width, height], projection=ccrs.PlateCarree())
ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
ax.set_aspect('auto')

domain_box = sgeom.box(EXTENT[0], EXTENT[2], EXTENT[1], EXTENT[3])

# Ocean background
ax.add_geometries([domain_box], crs=ccrs.PlateCarree(),
                  facecolor=OCEAN_COLOR, edgecolor='none', zorder=0)

# Clipped land
clipped_land = [domain_box.intersection(geom) for geom in cfeature.LAND.geometries()
                if domain_box.intersects(geom)]
ax.add_geometries(clipped_land, crs=ccrs.PlateCarree(),
                  facecolor=LAND_COLOR, edgecolor='black', linewidth=0.5, zorder=1)

# Clipped lakes
clipped_lakes = [domain_box.intersection(geom) for geom in cfeature.LAKES.geometries()
                 if domain_box.intersects(geom)]
ax.add_geometries(clipped_lakes, crs=ccrs.PlateCarree(),
                  facecolor=OCEAN_COLOR, edgecolor='black', linewidth=0.5, zorder=1)

# Clipped borders
clipped_borders = [domain_box.intersection(geom) for geom in cfeature.BORDERS.geometries()
                   if domain_box.intersects(geom)]
ax.add_geometries(clipped_borders, crs=ccrs.PlateCarree(),
                  facecolor='none', edgecolor='black', linewidth=0.5, zorder=2)

# =============================================================================
# 5. GRIDLINES, TICKS, & AXIS LABELS
# =============================================================================

gl = ax.gridlines(draw_labels=False, linewidth=1.0, color='black',
                  linestyle=(0, (3, 3)), zorder=2.5)
gl.xlocator = mticker.FixedLocator([-100, -80, -60, -40, -20])
gl.ylocator = mticker.FixedLocator([10, 20, 30, 40])

ax.set_xticks([-100, -80, -60, -40, -20], crs=ccrs.PlateCarree())
ax.set_yticks([10, 20, 30, 40], crs=ccrs.PlateCarree())

ax.xaxis.set_major_formatter(LongitudeFormatter())
ax.yaxis.set_major_formatter(LatitudeFormatter())

ax.tick_params(axis='both', direction='in', width=3.0, length=8.0,
               labelsize=15, top=True, right=True,
               labeltop=True, labelright=True, pad=6)

for spine in ax.spines.values():
    spine.set_linewidth(3.0)
    spine.set_zorder(10)

fig.text(left + width/2, bottom - 0.08, 'Longitude',
         ha='center', va='center', fontname='Arial Narrow', fontsize=20, color='black')
fig.text(left - 0.07, bottom + height/2, 'Latitude',
         ha='center', va='center', rotation='vertical', fontname='Arial Narrow', fontsize=20, color='black')

# =============================================================================
# 6. PLOTTING TRACKS & OBSERVATIONS
# =============================================================================

# --- Colored HURDAT tracks ---
for storm_id, group in hurdat.groupby('STORM ID'):
    lats = group['LAT'].values
    lons = group['LON'].values
    cats = group['CAT'].values

    for i in range(len(group) - 1):
        color = COLOR_MAP.get(cats[i], DEFAULT_COLOR)
        ax.plot(lons[i:i+2], lats[i:i+2], color=color, linewidth=LINE_WIDTH,
                transform=ccrs.PlateCarree(), solid_capstyle='round', zorder=3)

# --- HRDOBS observation dots (from inventory) ---
ax.plot(inventory['LON'].values, inventory['LAT'].values,
        marker='o', markersize=MARKER_SIZE_PTS,
        markerfacecolor='white', markeredgecolor='black',
        markeredgewidth=MARKER_EDGE_WIDTH, linestyle='none',
        transform=ccrs.PlateCarree(), zorder=4)

# =============================================================================
# 7. EXPORT
# =============================================================================

output_file = OUTPUT_FIGURE_FILE
plt.savefig(output_file, format='pdf', facecolor='white', edgecolor='none')
print(f"✅ Vector map generated successfully: {output_file}")

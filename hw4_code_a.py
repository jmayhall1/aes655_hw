# coding=utf-8
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import tcpyPI
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Use PlateCarree projection (good for global lon/lat)
proj = ccrs.PlateCarree()

# ===============================
# Load Surface Data
# ===============================
surface = netCDF4.Dataset(
    '/rstor/jmayhall/aes_hw4/f8cbe82c4f23e43b708c0423b9531e9c/'
    'data_stream-moda_stepType-avgua.nc'
)

sst = np.nanmean(surface.variables['sst'][:], axis=0) - 273.15  # °C
msl = np.nanmean(surface.variables['msl'][:], axis=0) / 100     # hPa

lon = surface.variables['longitude'][:]
lat = surface.variables['latitude'][:]

# ===============================
# Load Profile Data
# ===============================
profiles = netCDF4.Dataset(
    '/rstor/jmayhall/aes_hw4/ec5bdc8f35d3e19f52f18bd14cdedb73.nc'
)

T = np.nanmean(profiles.variables['t'][:], axis=0) - 273.15  # °C

# Convert specific humidity → mixing ratio (g/kg)
q_raw = profiles.variables['q'][:]
r = q_raw / (1 - q_raw)
q = np.nanmean(r, axis=0) * 1000  # g/kg

p = profiles.variables['pressure_level'][:]  # hPa

# Convert to float64
sst = np.array(sst, dtype=np.float64)
msl = np.array(msl, dtype=np.float64)
T = np.array(T, dtype=np.float64)
q = np.array(q, dtype=np.float64)
p = np.array(p, dtype=np.float64)

ny, nx = sst.shape
nlev = len(p)

# ===============================
# Allocate Output Arrays
# ===============================
vmax = np.full((ny, nx), np.nan)
To   = np.full((ny, nx), np.nan)
eff  = np.full((ny, nx), np.nan)

# ===============================
# Flatten for Faster Loop
# ===============================
sst_flat = sst.ravel()
msl_flat = msl.ravel()

T_flat = T.reshape(nlev, -1)
q_flat = q.reshape(nlev, -1)

vmax_flat = vmax.ravel()
To_flat   = To.ravel()
eff_flat  = eff.ravel()

# Precompute valid column mask (avoids repeated .any())
valid_mask = (~np.isnan(T_flat).any(axis=0)) & (~np.isnan(q_flat).any(axis=0))

# Local binding (faster lookup)
pi_func = tcpyPI.pi

npts = sst_flat.size

# ===============================
# Compute PI
# ===============================
for k in range(npts):

    if not valid_mask[k]:
        continue

    T_prof = T_flat[:, k]
    r_prof = q_flat[:, k]

    vmax_ij, _, ifl, To_ij, _ = pi_func(
        float(sst_flat[k]),
        float(msl_flat[k]),
        p,
        T_prof,
        r_prof
    )

    if ifl == 1:
        vmax_flat[k] = vmax_ij

        # Outflow returned in Kelvin → convert to °C for plotting
        To_flat[k] = To_ij - 273.15

        # Carnot efficiency (use Kelvin)
        Ts_K = sst_flat[k] + 273.15
        eff_flat[k] = (Ts_K - To_ij) / Ts_K

# ===============================
# Plotting
# ===============================
# Shift lon from 0–360 → -180–180
lon_centered = np.where(lon > 180, lon - 360, lon)

# Get sort indices so that lon is increasing
sort_idx = np.argsort(lon_centered)

# Reorder longitude and all corresponding 2D data arrays
lon_centered = lon_centered[sort_idx]
sst = sst[:, sort_idx]
vmax = vmax[:, sort_idx]
To = To[:, sort_idx]
eff = eff[:, sort_idx]
Lon, Lat = np.meshgrid(lon_centered, lat)
xticks = np.arange(-180, 181, 30)  # positions in degrees
yticks = np.arange(-90, 91, 30)
xticklabels = [f"{x}°" for x in xticks]  # e.g., "-180°", "-150°", ..., "180°"
yticklabels = [f"{y}°" for y in yticks]  # e.g., "-90°", "-60°", ..., "90°"

# Use PlateCarree projection (good for global lon/lat)
proj = ccrs.PlateCarree()

fig, axes = plt.subplots(
    2, 2,     figsize=(18, 9),               # wider than tall
    subplot_kw={'projection': proj},
    gridspec_kw={'hspace': 0.15, 'wspace': 0.1}, sharex=True, sharey=True
)

variables = [vmax, sst, To, eff]
titles = ['Potential Intensity (m/s)',
          'Sea Surface Temperature (°C)',
          'Outflow Temperature (°C)',
          'Carnot Efficiency']
cmaps = ['gist_ncar', 'gist_ncar', 'gist_ncar', 'gist_ncar']
vmins = [0, -5, -85, -0.05]
vmaxs = [95, 40, 25, 0.4]
cb_labels = [r'Wind Speed ($m/s$)', 'Temperature (°C)', 'Temperature (°C)', 'Efficiency']

Lon2d, Lat2d = np.meshgrid(lon_centered, lat)

for i, ax in enumerate(axes.flatten()):
    # Plot the data
    im = ax.pcolormesh(Lon2d, Lat2d, variables[i],
                       vmin=vmins[i], vmax=vmaxs[i],
                       cmap=cmaps[i], shading='auto',
                       transform=ccrs.PlateCarree())

    # Add coastlines and borders
    ax.coastlines(resolution='110m', linewidth=1)
    ax.add_feature(cfeature.BORDERS, linewidth=1)

    if i == 0 or i == 2:
        ax.set_yticks(yticks, crs=ccrs.PlateCarree())
        ax.set_yticklabels(yticklabels, fontsize=10)
        ax.set_ylabel('Latitude', fontsize=18)
    if i == 2 or i == 3:
        ax.set_xticks(xticks, crs=ccrs.PlateCarree())
        ax.set_xticklabels(xticklabels, fontsize=10, rotation=0)
        ax.set_xlabel('Longitude', fontsize=18)

    # Gridlines with labels
    gl = ax.gridlines(draw_labels=False, linewidth=0.8,
                      color='black', alpha=0.7, linestyle='--', xlocs=np.arange(-180,181,30),
                      ylocs=np.arange(-90,91,30))
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size':10}
    gl.ylabel_style = {'size':10}
    gl.xformatter = ccrs.cartopy.mpl.gridliner.LONGITUDE_FORMATTER
    gl.yformatter = ccrs.cartopy.mpl.gridliner.LATITUDE_FORMATTER


    ax.set_title(titles[i], fontsize=18)
    # Set global extent to show poles
    ax.set_global()

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.9, pad=0.02)
    cbar.set_label(cb_labels[i], fontsize=18)

fig.suptitle('2025 Annual-Mean State', fontsize=24)
plt.savefig('global_mean_cartopy.png', dpi=200)
plt.show()
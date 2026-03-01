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

sst = surface.variables['sst'][:] - 273.15  # °C
msl = surface.variables['msl'][:] / 100     # hPa

lon = surface.variables['longitude'][:]
lat = surface.variables['latitude'][:]

# ===============================
# Load Profile Data
# ===============================
profiles = netCDF4.Dataset(
    '/rstor/jmayhall/aes_hw4/ec5bdc8f35d3e19f52f18bd14cdedb73.nc'
)

T = profiles.variables['t'][:] - 273.15  # °C

# Convert specific humidity → mixing ratio (g/kg)
q_raw = profiles.variables['q'][:]
r = q_raw / (1 - q_raw)
q = r * 1000  # g/kg

p = profiles.variables['pressure_level'][:]  # hPa

# Convert to float64
sst = np.array(sst, dtype=np.float64)
msl = np.array(msl, dtype=np.float64)
T = np.array(T, dtype=np.float64)
q = np.array(q, dtype=np.float64)
p = np.array(p, dtype=np.float64)

# Local binding (faster lookup)
pi_func = tcpyPI.pi
count = 0
months = ['June', 'July', 'August', 'September', 'October', 'November']
for m in range(5, 11):
    ny, nx = sst[m, :, :].shape
    nlev = len(p)

    # ===============================
    # Allocate Output Arrays
    # ===============================
    vmax = np.full((ny, nx), np.nan)

    # ===============================
    # Flatten for Faster Loop
    # ===============================
    sst_flat = sst[m, :, :].ravel()
    msl_flat = msl[m, :, :].ravel()

    T_flat = T[m, :, :, :].reshape(nlev, -1)
    q_flat = q[m, :, :, :].reshape(nlev, -1)

    vmax_flat = vmax.ravel()

    # Precompute valid column mask (avoids repeated .any())
    valid_mask = (~np.isnan(T_flat).any(axis=0)) & (~np.isnan(q_flat).any(axis=0))

    npts = sst_flat.size
    current_month = months[count]
    # ===============================
    # Compute PI
    # ===============================
    for k in range(npts):

        if not valid_mask[k]:
            continue

        T_prof = T_flat[:, k]
        r_prof = q_flat[:, k]

        vmax_ij, _, ifl, _, _ = pi_func(
            float(sst_flat[k]),
            float(msl_flat[k]),
            p,
            T_prof,
            r_prof
        )

        if ifl == 1:
            vmax_flat[k] = vmax_ij

    # ===============================
    # Plotting
    # ===============================
    # Shift lon from 0–360 → -180–180
    lon_centered = np.where(lon > 180, lon - 360, lon)

    # Get sort indices so that lon is increasing
    sort_idx = np.argsort(lon_centered)

    # Reorder longitude and all corresponding 2D data arrays
    lon_centered = lon_centered[sort_idx]
    vmax = vmax[:, sort_idx]
    Lon, Lat = np.meshgrid(lon_centered, lat)
    xticks = np.arange(-180, 181, 30)  # positions in degrees
    yticks = np.arange(-90, 91, 30)
    xticklabels = [f"{x}°" for x in xticks]  # e.g., "-180°", "-150°", ..., "180°"
    yticklabels = [f"{y}°" for y in yticks]  # e.g., "-90°", "-60°", ..., "90°"

    # Use PlateCarree projection (good for global lon/lat)
    proj = ccrs.PlateCarree()

    fig, ax = plt.subplots(
        1, 1, figsize=(12, 6),
        subplot_kw={'projection': proj},
    )
    fig.subplots_adjust(left=0.05, right=1.05, top=0.95, bottom=0.05)

    Lon2d, Lat2d = np.meshgrid(lon_centered, lat)
    # Plot the data
    im = ax.pcolormesh(Lon2d, Lat2d, vmax,
                       vmin=0, vmax=95,
                       cmap='gist_ncar', shading='auto',
                       transform=ccrs.PlateCarree())

    # Add coastlines and borders
    ax.coastlines(resolution='110m', linewidth=1)
    ax.add_feature(cfeature.BORDERS, linewidth=1)
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())
    ax.set_yticklabels(yticklabels, fontsize=10)
    ax.set_ylabel('Latitude', fontsize=18)
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

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.8, pad=0.02)
    cbar.set_label(r'Wind Speed ($m/s$)', fontsize=18)

    fig.suptitle(f'2025 {current_month} PI', fontsize=24)
    plt.savefig(f'global_{current_month}_pi.png', dpi=200)
    plt.show()
    count += 1
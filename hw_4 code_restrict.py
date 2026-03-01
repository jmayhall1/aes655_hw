# coding=utf-8
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import tcpyPI

# ===============================
# Load Surface Data
# ===============================
surface = netCDF4.Dataset(
    'C:/Users/jmayhall/Downloads/f8cbe82c4f23e43b708c0423b9531e9c/data_stream-moda_stepType-avgua.nc'
)
sst_all = surface.variables['sst'][:] - 273.15  # °C
msl_all = surface.variables['msl'][:] / 100     # hPa
lon = surface.variables['longitude'][:]
lat = surface.variables['latitude'][:]

# Convert longitude to -180 to 180 if needed
if np.max(lon) > 180:
    lon = ((lon + 180) % 360) - 180

# ===============================
# Load Profile Data
# ===============================
profiles = netCDF4.Dataset(
    'C:/Users/jmayhall/Downloads/ec5bdc8f35d3e19f52f18bd14cdedb73.nc'
)
T_all = profiles.variables['t'][:] - 273.15
q_raw_all = profiles.variables['q'][:]
r_all = q_raw_all / (1 - q_raw_all)
q_all = r_all * 1000  # g/kg
p = profiles.variables['pressure_level'][:]

# Convert to numpy arrays
sst_all = np.array(sst_all, dtype=np.float64)
msl_all = np.array(msl_all, dtype=np.float64)
T_all = np.array(T_all, dtype=np.float64)
q_all = np.array(q_all, dtype=np.float64)
p = np.array(p, dtype=np.float64)

ny, nx = sst_all.shape[1], sst_all.shape[2]
nlev = len(p)

# ===============================
# NATL Domain Mask
# ===============================
lon2d, lat2d = np.meshgrid(lon, lat)

natl_mask = (
    (lat2d >= 5) & (lat2d <= 30) &
    (lon2d >= -100) & (lon2d <= -20)
)

# ===============================
# Compute PI per month (June–Nov)
# ===============================
vmax_months = []
pi_func = tcpyPI.pi

for m in range(5, 11):  # June–November
    print(f'Computing PI for month {m+1}...')

    sst = sst_all[m, :, :]
    msl = msl_all[m, :, :]
    T = T_all[m, :, :, :]
    q = q_all[m, :, :, :]

    # Apply spatial mask
    sst = sst[natl_mask]
    msl = msl[natl_mask]
    T = T[:, natl_mask]
    q = q[:, natl_mask]

    npts = sst.size

    vmax = np.full(npts, np.nan)

    valid_mask = (~np.isnan(T).any(axis=0)) & (~np.isnan(q).any(axis=0))

    for k in range(npts):
        if not valid_mask[k]:
            continue

        T_prof = T[:, k]
        r_prof = q[:, k]

        vmax_ij, _, ifl, _, _ = pi_func(
            float(sst[k]),
            float(msl[k]),
            p,
            T_prof,
            r_prof
        )

        if ifl == 1:
            vmax[k] = vmax_ij

    vmax_months.append(vmax)

# ===============================
# Plot Violin + Mean
# ===============================
fig, ax = plt.subplots(figsize=(10,6))

vmax_data = [v[np.isfinite(v)] for v in vmax_months]
means = [np.mean(v) for v in vmax_data]

positions = np.arange(6, 12)

parts = ax.violinplot(
    vmax_data,
    positions=positions,
    showmeans=False,
    showmedians=False,
    showextrema=False
)

for pc in parts['bodies']:
    pc.set_facecolor('#1f77b4')
    pc.set_alpha(0.6)

ax.plot(positions, means, 'ro-', label='Monthly Mean')

ax.set_xticks(positions)
ax.set_xticklabels(['Jun','Jul','Aug','Sep','Oct','Nov'])
ax.set_ylabel('Potential Intensity (m/s)')
ax.set_title('NATL Potential Intensity Distribution (5°–30°N, 100°–20°W)')
ax.grid(True)
ax.legend()

plt.tight_layout()
plt.savefig('distribution_restricted.png')
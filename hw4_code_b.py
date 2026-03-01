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

sst_all = np.array(sst_all, dtype=np.float64)  # ensures ndarray
msl_all = np.array(msl_all, dtype=np.float64)
T_all = np.array(T_all, dtype=np.float64)
q_all = np.array(q_all, dtype=np.float64)
p = np.array(p, dtype=np.float64)

ny, nx = sst_all.shape[1], sst_all.shape[2]
nlev = len(p)
nmonths = 12

# ===============================
# Compute PI per month (June-Nov)
# ===============================
vmax_months = []

pi_func = tcpyPI.pi

for m in range(5, 11):  # June-Nov (0-based index)
    print(f'Computing PI for month {m+1}...')

    sst = sst_all[m, :, :]
    msl = msl_all[m, :, :]
    T = T_all[m, :, :, :]
    q = q_all[m, :, :, :]

    # Flatten for faster loop
    sst_flat = sst.ravel()
    msl_flat = msl.ravel()
    T_flat = T.reshape(nlev, -1)
    q_flat = q.reshape(nlev, -1)

    vmax_flat = np.full(sst_flat.shape, np.nan)
    valid_mask = (~np.isnan(T_flat).any(axis=0)) & (~np.isnan(q_flat).any(axis=0))

    npts = sst_flat.size

    for k in range(npts):
        if not valid_mask[k]:
            continue

        T_prof = T_flat[:, k]
        r_prof = q_flat[:, k]


        vmax_ij, _, ifl, _, _ = pi_func(float(sst_flat[k]), float(msl_flat[k]), p, T_prof, r_prof)

        if ifl == 1:
            vmax_flat[k] = vmax_ij

    vmax_months.append(vmax_flat)

# ===============================
# Plot violin + mean line
# ===============================
fig, ax = plt.subplots(figsize=(10,6))

# Prepare data: flatten spatial dims
vmax_data = [v[np.isfinite(v)] for v in vmax_months]  # remove NaNs per month
means = [np.mean(v) for v in vmax_data]

# Violin plot
parts = ax.violinplot(vmax_data, positions=np.arange(6,12), showmeans=False, showmedians=False, showextrema=False)
for pc in parts['bodies']:
    pc.set_facecolor('#1f77b4')
    pc.set_alpha(0.6)

# Overlay mean line
ax.plot(np.arange(6,12), means, color='red', marker='o', linestyle='-', label='Monthly Mean')

ax.set_xticks(np.arange(6,12))
ax.set_xticklabels(['Jun','Jul','Aug','Sep','Oct','Nov'])
ax.set_ylabel('Potential Intensity (m/s)')
ax.set_title('Tropical Cyclone Potential Intensity: June–November')
ax.grid(True)
ax.legend()

plt.tight_layout()
plt.savefig('global_dist.png')
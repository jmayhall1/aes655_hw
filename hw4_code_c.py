# coding=utf-8
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import tcpyPI
from metpy.calc import bulk_shear
from metpy.units import units


# ===============================
# Load Surface Data
# ===============================
surface = netCDF4.Dataset(
    '/rstor/jmayhall/aes_hw4/f8cbe82c4f23e43b708c0423b9531e9c/data_stream-moda_stepType-avgua.nc'
)
sst_all = surface.variables['sst'][:] - 273.15  # °C
msl_all = surface.variables['msl'][:] / 100  # hPa
lon = surface.variables['longitude'][:]
lat = surface.variables['latitude'][:]

# ===============================
# Load Profile Data
# ===============================
profiles = netCDF4.Dataset(
    '/rstor/jmayhall/aes_hw4/ec5bdc8f35d3e19f52f18bd14cdedb73.nc'
)
T_all = profiles.variables['t'][:] - 273.15
q_raw_all = profiles.variables['q'][:]
r_all = q_raw_all / (1 - q_raw_all)
q_all = r_all * 1000  # g/kg
p = profiles.variables['pressure_level'][:]
u_all = np.asarray(profiles.variables['u'][:])
v_all = np.asarray(profiles.variables['v'][:])
met_p = np.asarray(p) * units.hPa

# ===============================
# Ensure float64
# ===============================
sst_all = np.array(sst_all, dtype=np.float64)
print(sst_all.shape)
msl_all = np.array(msl_all, dtype=np.float64)
T_all = np.array(T_all, dtype=np.float64)
q_all = np.array(q_all, dtype=np.float64)
p = np.array(p, dtype=np.float64)

ny, nx = sst_all.shape[1], sst_all.shape[2]
nlev = len(p)

# ===============================
# Define MDR mask: 10-20N, 20-60W
# ===============================
# Convert longitude if necessary
lon_wrap = np.where(lon > 180, lon - 360, lon)

lon_mask = (lon_wrap >= -60) & (lon_wrap <= -20)
lat_mask = (lat >= 10) & (lat <= 20)
mdr_mask = np.outer(lat_mask, lon_mask)  # 2D mask

# ===============================
# Compute MDR Area Weights
# ===============================

# 2D latitude grid
lat2d = np.repeat(lat[:, np.newaxis], len(lon), axis=1)

# Cosine weights
weights_2d = np.cos(np.deg2rad(lat2d))

# Mask to MDR
weights_mdr = weights_2d[mdr_mask]


pi_func = tcpyPI.pi
vmax_means = []
sst_means = []
eff_means = []
vws_means = []

for m in range(0, 12):
    print(f'Computing PI for month {m + 1}...')

    sst = sst_all[m, :, :]
    msl = msl_all[m, :, :]
    T = T_all[m, :, :, :]
    q = q_all[m, :, :, :]
    u, v = u_all[m, :, :, :], v_all[m, :, :, :]
    print(mdr_mask.shape)

    # Restrict to MDR
    sst_mdr = sst[mdr_mask]
    msl_mdr = msl[mdr_mask]
    u_mdr, v_mdr = u[:, mdr_mask], v[:, mdr_mask]


    vws_mdr = np.zeros(u_mdr.shape[1])

    for k in range(u_mdr.shape[1]):
        u_prof = (u_mdr[:, k] * units('m/s')).to('knots')
        v_prof = (v_mdr[:, k] * units('m/s')).to('knots')

        u_s, v_s = bulk_shear(
            met_p, u_prof, v_prof,
            bottom=850 * units.hPa,
            depth=650 * units.hPa
        )

        vws_mdr[k] = np.sqrt(u_s.to('m/s').magnitude ** 2 + v_s.to('m/s').magnitude ** 2)

    T_flat = T[:, mdr_mask]
    q_flat = q[:, mdr_mask]

    vmax_flat = np.full(sst_mdr.shape, np.nan)
    eff_flat = np.full(sst_mdr.shape, np.nan)
    valid_mask = (~np.isnan(T_flat).any(axis=0)) & (~np.isnan(q_flat).any(axis=0))

    for k in range(sst_mdr.size):
        if not valid_mask[k]:
            continue

        T_prof = T_flat[:, k]
        r_prof = q_flat[:, k]

        vmax_ij, _, ifl, To_ij, _ = pi_func(float(sst_mdr[k]), float(msl_mdr[k]), p, T_prof, r_prof)

        if ifl == 1:
            vmax_flat[k] = vmax_ij
            # Carnot efficiency (use Kelvin)
            Ts_K = sst_mdr[k] + 273.15
            eff_flat[k] = (Ts_K - To_ij) / Ts_K

    # Compute areal mean for the month
    # ===============================
    # Area-Weighted Means
    # ===============================

    def area_weighted_mean(data, weights):
        valid = ~np.isnan(data)
        return np.sum(data[valid] * weights[valid]) / np.sum(weights[valid])


    vmax_means.append(area_weighted_mean(vmax_flat, weights_mdr))
    sst_means.append(area_weighted_mean(sst_mdr, weights_mdr))
    eff_means.append(area_weighted_mean(eff_flat, weights_mdr))
    vws_means.append(area_weighted_mean(vws_mdr, weights_mdr))
# ===============================
# Plot 4 Separate Subplots
# ===============================
months = ['Jan','Feb','Mar','Apr','May','Jun',
          'Jul','Aug','Sep','Oct','Nov','Dec']

fig, axs = plt.subplots(2, 2, figsize=(10, 6), sharex=True)

# --- PI ---
axs[0, 0].plot(months, vmax_means, marker='o')
axs[0, 0].set_ylabel('PI (m/s)')
axs[0, 0].set_title('Areal Mean Potential Intensity')
axs[0, 0].grid(True)

# --- SST ---
axs[0, 1].plot(months, sst_means, marker='o')
axs[0, 1].set_ylabel('SST (°C)')
axs[0, 1].set_title('Areal Mean SST')
axs[0, 1].grid(True)

# --- Efficiency ---
axs[1, 0].plot(months, eff_means, marker='o')
axs[1, 0].set_ylabel('Efficiency (unitless)')
axs[1, 0].set_title('Areal Mean Carnot Efficiency')
axs[1, 0].set_xlabel('Month')
axs[1, 0].grid(True)

# --- VWS ---
axs[1, 1].plot(months, vws_means, marker='o')
axs[1, 1].set_ylabel('VWS (kt)')
axs[1, 1].set_title('Areal Mean 850–200 hPa Shear')
axs[1, 1].set_xlabel('Month')
axs[1, 1].grid(True)

fig.suptitle('Atlantic MDR Monthly Areal Means', fontsize=20)
plt.tight_layout()
plt.savefig('atlantic_areal_mean_separate.png')
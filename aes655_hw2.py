# coding=utf-8
import os
import re
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = 'C:/Users/jmayhall/Downloads/AES565_HW2/'

SST_ORDER = [10, 20, 25, 30, 40]
WIND_ORDER = [0, 5, 20, 3]

def extract_number(name):
    match = re.search(r'\d+', name)
    return int(match.group()) if match else None

# ---------------------------
# Loop over top-level folders
# ---------------------------
for folder in os.listdir(BASE_DIR):
    if folder == 'input_holder':
        continue

    folder_path = os.path.join(BASE_DIR, folder)
    if not os.path.isdir(folder_path):
        continue

    # ---------------------------
    # Collect and sort subfolders
    # ---------------------------
    subfolders = [
        sf for sf in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, sf))
    ]

    if 'SST' in folder:
        order = SST_ORDER
    elif 'surfacewind' in folder:
        order = WIND_ORDER
    else:
        order = None
        new_subfolders = [subfolders[0], subfolders[1], subfolders[4], subfolders[6], subfolders[5],
                          subfolders[2], subfolders[3]]
        subfolders = new_subfolders

    if order is not None:
        subfolders = sorted(
            subfolders,
            key=lambda sf: order.index(extract_number(sf))
            if extract_number(sf) in order else 999
        )

    # ---------------------------
    # Create ONE figure per folder
    # ---------------------------
    fig, ax = plt.subplots(1, 3, figsize=(10, 8), sharey=True)

    # ---------------------------
    # Loop over subfolders
    # ---------------------------
    for sub_folder in subfolders:
        filename = os.path.join(folder_path, sub_folder, 'profile.out')
        if not os.path.isfile(filename):
            continue

        data = np.loadtxt(filename)

        pressure = data[:, 0]
        temperature = data[:, 1]
        relative_humidity = data[:, 4] * 100
        cloud_water = data[:, -1]

        label = sub_folder
        label = label.replace('output_', '')
        label = label.replace('_', ', ')
        label = label.replace('co2,', r' $CO_2$' + ',\n')
        label = label.replace('ch4', r' $CH_4$')
        label = label.replace('ms', r' $\frac{m}{s}$')
        label = label.replace('0C', r'0$^\circ C$')
        label = label.replace('5C', r'5$^\circ C$')
        label = label.replace('original, ', '')

        ax[0].plot(temperature, pressure, label=label)
        ax[1].plot(relative_humidity, pressure, label=label)
        ax[2].plot(cloud_water, pressure, label=label)

    # ---------------------------
    # Axes formatting (unchanged fonts)
    # ---------------------------
    ax[0].set_xlabel(r"Temperature ($^\circ C$)", fontsize=14)
    ax[0].set_xlim(-80, 120)
    ax[0].set_xticks([-80, -40, 0, 40, 80, 120])
    ax[0].set_ylabel("Pressure (hPa)", fontsize=14)
    ax[0].grid(True)

    ax[1].set_xlabel("Relative Humidity (%)", fontsize=14)
    ax[1].set_xlim(0, 100)
    ax[1].grid(True)

    ax[2].set_xlabel(r"Cloud Water ($\frac{g}{kg}$)", fontsize=14)
    ax[2].set_xlim(0, 350)
    ax[2].grid(True)

    ax[0].invert_yaxis()

    # ---------------------------
    # Deduplicated legend + spacing
    # ---------------------------
    handles, labels = ax[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    plt.subplots_adjust(right=0.83, top=0.92)

    fig.legend(
        unique.values(),
        unique.keys(),
        fontsize=10,
        loc='upper right',
        bbox_to_anchor=(1, 0.92),
        ncol=1,
        frameon=True,
        framealpha=0
    )
    plot_title = folder
    plot_title = plot_title.replace('RCE_surfacewind', 'Varying Surface Wind')
    plot_title = plot_title.replace('RCE_ppm', r'Varying $CO_2$ and $CH_4$')
    plot_title = plot_title.replace('RCE_SST', 'Varying SST')

    plt.suptitle(
        f"Sounding Profile: {plot_title}",
        fontsize=18
    )

    plt.savefig(os.path.join(BASE_DIR, f"{folder}_combined_img.png"))
    plt.close(fig)

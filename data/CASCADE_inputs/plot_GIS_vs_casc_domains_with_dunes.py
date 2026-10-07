# Lexi (Van Blunk) Fiegelist
# last updated 9/28/2026
# this code loads the numpy files from the GIS download and plots them with the cascade domains that were chosen using the dune_topot_extractor
# there is also now an option to plot the chosen dunes

import numpy as np
import os
from matplotlib import pyplot as plt
import pickle


def remove_water_rows(domain_array, w_elev):
    n_rows = np.shape(domain_array)[0]
    r_list = []
    for r in range(n_rows):
        if not np.all(domain_array[r, :] <= w_elev):
            r_list.append(r)
    min_r = min(r_list)
    max_r = max(r_list)
    domain = domain_array[min_r:max_r + 1, :]

    return domain


plt.rcParams["font.size"]=12
plt.ioff()
year = "2020"
version = "test"  # same version name from the dune_topo_extractor

# specify the directories where domains are saved
gis_datadir = r"C:\Users\agfig\model\final_domains\{0}_final_GIS_npys".format(
    year)  # domains from GIS that were converted into numpy arrays
casc_datadir = r"C:\Users\agfig\model\final_domains\cascade_domains\domains_{0}_{1}".format(
    year, version)  # domains that were processed using dune_top_extractor script
casc_datadir_dunes = r"C:\Users\agfig\model\final_domains\cascade_domains\dunes_{0}_{1}".format(
    year, version)  # dunes that were processed using dune_top_extractor script
save_dir = r"C:\Users\agfig\model\final_domains\cascade_domains\comparison\{0}".format(year)  # location to save the plots

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# if including the dune and interior locations that were selected
# otherwise, set dune_dict = "" and int_dict = ""
dune_dict = r"C:\Users\agfig\model\final_domains\cascade_domains\dunes_loc_{0}_{1}.pkl".format(year, version)
int_dict = r"C:\Users\agfig\model\final_domains\cascade_domains\interior_loc_{0}_{1}.pkl".format(year, version)
if dune_dict != "" and int_dict != "":
    with open(dune_dict, 'rb') as file:  # "rb" for read, binary mode
        dune_locs = pickle.load(file)
    with open(int_dict, 'rb') as file:  # "rb" for read, binary mode
        int_locs = pickle.load(file)
else:
    dune_dict = ""
    int_dict = ""

# MHW conversion
MHW = 0.421  # 0 m NAVD88 = X m MHW
berm_elev = 1.95  # m NAVD88
berm_elev_mMHW = berm_elev - MHW  # m MHW

# loop through each file and plot the matching domains
start_domain = 3  # not using domains 1 and 2
stop_domain = 25  # not using domain 26
# domains = [22, 23, 24, 25]
for d in range(start_domain, stop_domain + 1):
# for d in domains:
    gis_domain = np.load(os.path.join(gis_datadir, "domain_{0}.npy".format(d)))  # this is in m NAVD88
    casc_domain = np.load(
        os.path.join(casc_datadir, "domain_{0}_interior_{1}.npy".format(d, year)))  # this is in dam MHW
    casc_dunes = np.load(os.path.join(casc_datadir_dunes,
                                      "domain_{0}_dunes_{1}.npy".format(d, year)))  # this is in dam, height above berm

    # convert domains into m MHW
    gis_domain = gis_domain - MHW
    casc_domain = casc_domain * 10
    casc_dunes = (casc_dunes * 10) + berm_elev_mMHW

    # put domains in the same orientation
    # GIS domain has ocean on the right, cascade domain has ocean on top 
    # gis_domain = np.rot90(gis_domain)
    # UPDATE: my ocean is on bottom
    gis_domain = np.flip(gis_domain)

    # ------------------------------------------ plotting the domains ------------------------------------------------------
    xlabel = "alongshore distance (dam)"
    ylabel = "cross-shore distance (dam)"

    minz = -3
    maxz = 5

    # if plotting dunes and interior start, make 3 plots, otherwise, make 2 plots
    # no dunes or interior highlighted 
    if dune_dict == "":
        fig1 = plt.figure(figsize=[7, 8])

        # GIS plot
        ax1 = fig1.add_subplot(121)
        mat1 = ax1.matshow(
            gis_domain,
            cmap="terrain",
            vmin=minz,
            vmax=maxz,
        )
        # cbar = fig1.colorbar(mat1)
        # cbar.set_label('m MHW', rotation=270, labelpad=10)
        ax1.set_title("GIS domain")
        ax1.set_ylabel(ylabel)
        # ax1.set_xlabel(xlabel)
        plt.gca().xaxis.tick_bottom()

        # cascade domain plot with dunes
        ax2 = fig1.add_subplot(122)
        mat2 = ax2.matshow(
            np.vstack((casc_dunes, casc_domain)),
            cmap="terrain",
            vmin=minz,
            vmax=maxz,
        )
        cbar = fig1.colorbar(mat2)
        cbar.set_label('m MHW', rotation=270, labelpad=10)
        ax2.set_title("cascade domain")
        # ax2.set_ylabel(ylabel)
        # ax2.set_xlabel(xlabel)
        plt.gca().xaxis.tick_bottom()

    else:
        fig1 = plt.figure(figsize=[15, 10])
        domain_n = "domain_{0}".format(d)
        dune_locs_d = dune_locs[domain_n]
        int_locs_d = int_locs[domain_n]
        max_col = np.shape(gis_domain)[1]

        # GIS plot
        ax1 = fig1.add_subplot(131)
        mat1 = ax1.matshow(
            gis_domain,
            cmap="terrain",
            vmin=minz,
            vmax=maxz,
        )
        # cbar = fig1.colorbar(mat1)
        # cbar.set_label('m MHW', rotation=270, labelpad=10)
        ax1.set_title("GIS domain")
        ax1.set_ylabel(ylabel)
        # ax1.set_xlabel(xlabel)
        plt.gca().xaxis.tick_bottom()

        # GIS plot with dunes and interior cells marked
        # need to remove the rows with all water cells so the indeces are correct

        gis_domain_no_water = remove_water_rows(gis_domain, -3)
        # if d==23 or d==24 or d==25:
        #     max_row_to_plot = 35
        # else:
        #     max_row_to_plot = 25
        max_row_to_plot = 50
        ax3 = fig1.add_subplot(132)
        mat3 = ax3.matshow(
            gis_domain_no_water[0:max_row_to_plot],
            cmap="terrain",
            vmin=minz,
            vmax=maxz,
        )
        # cbar = fig1.colorbar(mat1)
        # cbar.set_label('m MHW', rotation=270, labelpad=10)
        ax3.scatter(np.arange(0, max_col), dune_locs_d, color="red", s=5)
        ax3.plot(np.arange(0, max_col), int_locs_d[0], color="magenta")
        ax3.set_title("GIS domain")
        # ax3.set_ylabel(ylabel)
        # ax1.set_xlabel(xlabel)
        plt.gca().xaxis.tick_bottom()

        # cascade domain plot with dunes
        ax2 = fig1.add_subplot(133)
        mat2 = ax2.matshow(
            np.vstack((casc_dunes, casc_domain)),
            cmap="terrain",
            vmin=minz,
            vmax=maxz,
        )
        cbar = fig1.colorbar(mat2)
        cbar.set_label('m MHW', rotation=270, labelpad=10)
        ax2.set_title("cascade domain")
        # ax2.set_ylabel(ylabel)
        # ax2.set_xlabel(xlabel)
        plt.gca().xaxis.tick_bottom()

    # for all plots
    fig1.text(0.5, 0.01, xlabel, ha='center', va='center')
    fig1.tight_layout()
    # plt.show()
    fig1.savefig(os.path.join(save_dir, "domains{0}_{1}.png".format(d, version)))
    plt.close(fig1)


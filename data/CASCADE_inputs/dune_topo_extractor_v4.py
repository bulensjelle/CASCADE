# ==============================================================================
# CASCADE Dune & Topography Extractor set for Masonboro Island, NC
# Received from Hannah Henry and modified by Lexi (Van Blunk) Fiegelist 
# Last updated 9/28/2026
# INPUT  : numpy array shape = (alongshore_rows, cross_shore_cols)
# OUTPUT : topography_dunes and dune, in decameters (dam)
# ==============================================================================

import os
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import pickle


def remove_water_cols(domain_array, w_elev):
    n_cols = np.shape(domain_array)[1]
    c_list = []
    for c in range(n_cols):
        if not np.all(domain_array[:, c] == w_elev):
            c_list.append(c)
    min_c = min(c_list)
    max_c = max(c_list)
    domain = domain_array[:, min_c:max_c + 1]

    return domain

def remove_water_rows(domain_array, w_elev):
    n_rows = np.shape(domain_array)[0]
    r_list = []
    for r in range(n_rows):
        if not np.all(domain_array[r, :] == w_elev):
            r_list.append(r)
    min_r = min(r_list)
    max_r = max(r_list)
    domain = domain_array[min_r:max_r + 1, :]

    return domain


def process_domain_file(
        in_path: Path,
        topo_out_dir: Path,
        dune_out_dir: Path,
        MHW_M=0.421,
        BERM_ELEV_NAVD_M=1.95,
        BEACH_START_THR_M=0.75,
        DUNE_WINDOW_PX=15,
        SENTINEL_WATER_M=-3.0,
        TOPO_ROWS=200,
        ALONG_COLS=50,
        OCEAN_LOC="bottom",
        SHIFT_CELLS=False,
        use_const_interior=False,
        year=2020,
        dune_loc_dict = {},
        interior_loc_dict = {},
        set_dune_row_start=False,
        dune_row_start=5,
        ) -> None:
    """Process a single domain elevation array and write topography and dune outputs."""
    arr = np.load(in_path).astype(float, copy=False)
    if arr.ndim != 2:
        print(f"[skip] {in_path.name}: expected 2D array, got {arr.ndim}D")
        return

    # flip domains so that the ocean is on the right since that is how the code was written and I dont have time to
    # change it right now
    if OCEAN_LOC == "left":
        arr = np.flip(arr)
    elif OCEAN_LOC == "bottom":
        arr = np.rot90(arr)
    elif OCEAN_LOC == "top":
        arr = np.flip(arr)  # now it is the same orientation as "bottom"
        arr = np.rot90(arr)

    # Subtract MHW and clamp <-1 m to -3.0 m (MHW-relative) before any slicing
    # note: lexi edited this since we are keeping the marsh cells which are between 0 and -3 m MHW
    arr = arr - MHW_M
    arr[arr < -3.0] = SENTINEL_WATER_M
    # remove columns that are all water (need to keep rows so that alongshore length is always 50)
    arr = remove_water_cols(arr, -3)

    n_along, n_cross = arr.shape
    TOPO_ROWS = n_cross

    # Messages
    if n_along < ALONG_COLS:
        print(
            f"[warn] {in_path.name}: alongshore={n_along} < {ALONG_COLS}; "
            f"trailing output cols remain sentinel."
            )
    elif n_along > ALONG_COLS:
        print(
            f"[warn] {in_path.name}: alongshore={n_along} > {ALONG_COLS}; "
            f"only first {ALONG_COLS} profiles used."
            )

    # Allocate fixed-size outputs
    topo_m = np.full((TOPO_ROWS, ALONG_COLS), fill_value=SENTINEL_WATER_M, dtype=float)
    dune_m = np.full((ALONG_COLS,), fill_value=SENTINEL_WATER_M, dtype=float)

    dune_loc_array = []  # lexi added to track chosen dune rows
    start_interior_array = np.ones([1, n_along]) * np.nan

    # Process up to ALONG_COLS alongshore profiles
    n_cols_to_fill = min(ALONG_COLS, n_along)
    for i in range(n_cols_to_fill):
        # Profile as stored (land ... ocean-right)
        prof_lr = arr[i, :]  # this is now one column with marsh/back-barrier on top, ocean on bottom
        # Ensure index 0 == ocean
        prof = np.flip(prof_lr)  # ocean is now on top

        # 1) First index where z > 0.5 m (MHW-relative) OR z > specified beach elevation
        idx = np.where(prof > BEACH_START_THR_M)[0]
        if idx.size == 0 and set_dune_row_start is False:
            # there are no cells above the beach threshold
            dune_loc_array.append(np.nan)  # just make the dune cell nan
        else:
            # either use the beach width or set dune row to locate dunes
            if set_dune_row_start:
                # set the search window based on the input dune row
                start_beach = dune_row_start
            else:
                start_beach = int(idx[0])


            # 2) 8-pixel window landward of that point OR specified dune window size
            end_beach = min(start_beach + DUNE_WINDOW_PX, prof.size)
            if end_beach <= start_beach:
                continue

            # 3) Dune elevation = max in the window
            window = prof[start_beach:end_beach]
            if window.size == 0:
                continue

            dune_elev = float(np.max(window))

            # 4) Dune location = first index in ENTIRE profile equal to dune_elev
            matches = np.where(prof == dune_elev)[0]
            if matches.size == 0:
                continue

            dune_loc = int(matches[0])
            dune_loc_array.append(dune_loc)

            # 5) Island interior starts immediately landward of the dune
            start_island = dune_loc + 1
            # start_interior_array.append(start_island)  # note this is filled left to right with ocean on top
            start_interior_array[0,i] = start_island
            use_elev = (
                prof[start_island:-1]
                if start_island < (prof.size - 1)
                else np.array([], dtype=float)
            )

            # 6) Write interior into rows (top-down); truncate/pad to TOPO_ROWS
            # if "use_constant_interior" is set to True, the topo_m array gets over-ridden below
            if use_elev.size > 0:
                rows_to_copy = min(TOPO_ROWS, use_elev.size)
                topo_m[0:rows_to_copy, i] = use_elev[:rows_to_copy]
                # remaining rows stay as sentinel

            # 7) Dune height above berm, min 0.1 m
            dune_h_m = dune_elev - (BERM_ELEV_NAVD_M - MHW_M)  # both MHW-relative
            if dune_h_m < 0.0:
                dune_h_m = 0.1
            dune_m[i] = dune_h_m

    # is use_constant_interior is True, we start the interior domain at the same row across the alongshore rather than varying by dune location
    # this helps keep the back-barrier aligned properly
    if use_const_interior:
        arr_ocean_top = np.rot90(arr)
        dune_loc_array = np.array(dune_loc_array)
        max_dune_loc = np.nanmax(dune_loc_array)
        start_island = int(max_dune_loc + 1)
        topo_m = arr_ocean_top[start_island:, :]
        start_interior_array = np.ones([1,n_along])*start_island

    # remove rows that are all water cells
    topo_m = remove_water_rows(topo_m, -3)

    # convert to decameters (dune already in height above berm)
    topo_dm = topo_m * 0.1
    dune_dm = dune_m * 0.1

    # Save
    stem = in_path.stem  # e.g., "domain_7"
    dune_loc_dict[stem] = dune_loc_array
    interior_loc_dict[stem] = start_interior_array
    topo_out = Path(topo_out_dir) / f"{stem}_interior_{year}.npy"
    dune_out = Path(dune_out_dir) / f"{stem}_dunes_{year}.npy"
    np.save(topo_out, topo_dm)
    np.save(dune_out, dune_dm)
    print(
        f"[ok] wrote {topo_out.name} (shape {topo_dm.shape}), "
        f"{dune_out.name} (len {dune_dm.size})"
        )

    return topo_m, dune_m, dune_loc_dict, interior_loc_dict


############################################################################################
# main script
############################################################################################
plt.rcParams["font.size"] = 14

# --- PATHS --------------------------------------------------------------
version = "test"  # save version to append to folder name
year = 1996
LOAD_PATH = r"OneDrive - Universiteit Utrecht/Documents/WADWAD/Models/CASCADE/CASCADE_inputs/Terschelling/"
TOPO_SAVE_PATH = r"OneDrive - Universiteit Utrecht/Documents/WADWAD/Models/CASCADE/CASCADE_inputs/Terschelling/Interiors_{0}_{1}".format(year, version)
DUNE_SAVE_PATH = r"OneDrive - Universiteit Utrecht/Documents/WADWAD/Models/CASCADE/CASCADE_inputs/Terschelling/Domains_{0}_{1}".format(year, version)
dict_save_path = r"C:\Users\agfig\model\final_domains\cascade_domains"  # used for plotting the chosen dune cells

os.makedirs(TOPO_SAVE_PATH, exist_ok=True)
os.makedirs(DUNE_SAVE_PATH, exist_ok=True)

# --- CONSTANTS ----------------------------------------------------
MHW_M = 0.82              # meters (NAVD88)
BERM_ELEV_NAVD_M = 2.0    # meters (NAVD88)
SENTINEL_WATER_M = -3.0    # meters (MHW-relative)
TOPO_ROWS = 600            # number of inland rows to write
ALONG_COLS = 50            # number of alongshore profiles
OCEAN_LOC = "top"       # "top", "bottom", "left", or "right" based on your exported GIS domains

# -------------------------------------------------
# dictionaries for dune locations and interior locations
# dictionary will contain domain as key, dune or interior cells as values
dune_dict = {}
interior_dict = {}


# load the data and make the directories for saving
load_dir = Path(LOAD_PATH)
topo_dir = Path(TOPO_SAVE_PATH)
dune_dir = Path(DUNE_SAVE_PATH)
topo_dir.mkdir(parents=True, exist_ok=True)
dune_dir.mkdir(parents=True, exist_ok=True)

# Process domain elevation arrays: domain_#.npy
names = sorted(
    [
        n for n in os.listdir(load_dir)
        if n.endswith(".npy") and n.startswith("segment_")
        ]
    )
print(f"[info] Found {len(names)} domain file(s) in {load_dir}")


for name in names:

    # --------------------------- METHODS FOR BREAKING UP THE DUNES AND INTERIOR ------------------------------
    # you can define different methods for different domains below
    # original: use a beach elevation to define where to start looking for dunes
    # BEACH_START_THR_M = 0.5    # meters (MHW-relative), strict '>' comparison
    # new options:
    # use_const_interior (True/False): define a row to start the interior rather than varying by dune location (selects the row after the most landward dune cell)
    # set_dune_row_start (True/False): instead of using beach elevation to look for dunes, specify a row in the domain (AFTER WATER CELLS HAVE BEEN REMOVED)
    # dune_row_start: domain row to use to start searching for dunes AFTER WATER CELLS HAVE BEEN REMOVED

    # ----- 2020 domains ---------------------------------------------------------------------------------------------
    if "_23" in name or "_24" in name or "_25" in name:
        BEACH_START_THR_M = 0.5
        DUNE_WINDOW_PX = 5
        use_const_interior = False 
        set_dune_row_start = False
        dune_row_start = 0  # not used since set_dune_row_start is False
    elif "_22" in name:
        BEACH_START_THR_M = 0.5
        DUNE_WINDOW_PX = 10
        use_const_interior = False
        set_dune_row_start = False
        dune_row_start = 0  # not used since set_dune_row_start is False
    elif "_20" in name or "_21" in name: 
        BEACH_START_THR_M = 0.5
        DUNE_WINDOW_PX = 4
        use_const_interior = True  
        set_dune_row_start = False
        dune_row_start = 0  # not used since set_dune_row_start is False
    elif "_19" in name:
        BEACH_START_THR_M = 0.5  # not used since set_dune_row_start is True
        DUNE_WINDOW_PX = 3
        use_const_interior = True
        set_dune_row_start = True
        dune_row_start = 17
    elif "_14" in name:
        BEACH_START_THR_M = 0.5  # not used since set_dune_row_start is True
        DUNE_WINDOW_PX = 2
        use_const_interior = True
        set_dune_row_start = True
        dune_row_start = 16
    elif "_11" in name :
        BEACH_START_THR_M = 0.5
        DUNE_WINDOW_PX = 3
        use_const_interior = True
        set_dune_row_start = False
        dune_row_start = 0  # not used since set_dune_row_start is False
    elif "_3" in name:
        BEACH_START_THR_M = 0.5  # not used since set_dune_row_start is True
        DUNE_WINDOW_PX = 3
        use_const_interior = True 
        set_dune_row_start = True
        dune_row_start = 35
    else:
        BEACH_START_THR_M = 0.5
        DUNE_WINDOW_PX = 10
        use_const_interior = True  
        set_dune_row_start = False
        dune_row_start = 0  # not used since set_dune_row_start is False


    # ------------------ create the domains --------------------------------------
    topo_domain, dune_domain, dune_dict, interior_dict = process_domain_file(
        load_dir / name,
        topo_dir,
        dune_dir,
        MHW_M=MHW_M,
        BERM_ELEV_NAVD_M=BERM_ELEV_NAVD_M,
        BEACH_START_THR_M=BEACH_START_THR_M,
        DUNE_WINDOW_PX=DUNE_WINDOW_PX,
        SENTINEL_WATER_M=SENTINEL_WATER_M,
        TOPO_ROWS=TOPO_ROWS,
        ALONG_COLS=ALONG_COLS,
        OCEAN_LOC=OCEAN_LOC,
        use_const_interior=use_const_interior,
        year=year,
        dune_loc_dict=dune_dict,
        interior_loc_dict=interior_dict,
        set_dune_row_start=set_dune_row_start,
        dune_row_start=dune_row_start,
        )

# save the dictionaries with dune and interior locations
dunes = os.path.join(dict_save_path, "dunes_loc_{0}_{1}.pkl".format(year, version))
with open(dunes, 'wb') as file:  # "wb" for write, binary mode
    pickle.dump(dune_dict, file)
    print('dunes dictionary saved successfully to file: {}'.format(dunes))
interior = os.path.join(dict_save_path, "interior_loc_{0}_{1}.pkl".format(year, version))
with open(interior, 'wb') as file:  # "wb" for write, binary mode
    pickle.dump(interior_dict, file)
    print('interior dictionary saved successfully to file: {}'.format(interior))




import os
import gc
import math
from multiprocessing import shared_memory
import sys
from typing import List
from datasets.utils import BaseProcessor
from dask import delayed

from dask.distributed import Client, LocalCluster


import numpy as np
from numba import jit
import rasterio
from rasterio.transform import from_bounds
import concurrent.futures
import xarray
import rioxarray as rio
from tqdm import tqdm

from .nc_tools import read_area_from_netcdf_list, write_to_netcdf
from .data_tools import get_cpu_ram, interpolate_array, write_geotiff

@jit(nopython=True) #type:ignore
def find_max_sum_new(suit_vals, span, harvests) -> tuple:
    """
    Finds the optimal combination of days to maximize the sum of suitability values for harvesting.

    Parameters:
    - suit_vals (list): A list containing the suit values for each day of the year.
    - span (int): The number of consecutive days required for a single harvest.
    - harvests (int): The number of harvests to perform (2 or 3).

    Returns:
    - tuple: A tuple containing two elements:
        - list: The indices of the selected days for harvesting.
        - int: The maximum sum of suit values achievable with the selected days.

    Note:
    - The function assumes that the length of suit_vals is 365, corresponding to each day of the year.
    """
    max_sum = 0
    max_indices = [-1, -1] if harvests == 2 else [-1, -1, -1]

    if harvests == 2:
        for day1 in range(365-span):
            suit1 = suit_vals[day1]
            for day2 in range((day1 + span) % 365, (day1 - span) % 365):
                suit2 = suit_vals[day2]
                curr_sum = suit1 + suit2
                if curr_sum > max_sum:
                    max_sum, max_indices = curr_sum, [day1, day2]
    else:
        for day1 in range(365-span):
            suit1 = suit_vals[day1]
            for day2 in range((day1 + span) % 365, (day1 - (2 * span)) % 365):
                suit2 = suit_vals[day2]
                for day3 in range((day2 + span) % 365, (day1 - span) % 365):
                    suit3 = suit_vals[day3]
                    curr_sum = suit1 + suit2 + suit3
                    if curr_sum > max_sum:
                        max_sum, max_indices = curr_sum, [day1, day2, day3]
        max_sum_2hv = 0
        max_indices_2hv = [-1, -1]
        for day1 in range(365-span):
            suit1 = suit_vals[day1]
            for day2 in range((day1 + span) % 365, (day1 - span) % 365):
                suit2 = suit_vals[day2]
                curr_sum = suit1 + suit2
                if curr_sum > max_sum:
                    max_sum_2hv, max_indices_2hv = curr_sum, [day1, day2]
        if max_sum_2hv >= max_sum:
            return max_indices_2hv, max_sum_2hv
    return sorted(max_indices), max_sum  

def vernalization_params_winter_crops(plant_params,):
    vernalization_days = 0
    if 'wintercrop' in plant_params.keys():
        wintercrop = plant_params.get('wintercrop')[0].lower().strip() == 'y'
        if wintercrop:
            vernalization_period = int(plant_params.get('vernalization_effective_days')[0])
            vernalization_tmax = float(plant_params.get('vernalization_tmax')[0])
            vernalization_tmin = float(plant_params.get('vernalization_tmin')[0])
            vernalization_days = float(plant_params.get('days_to_vernalization')[0])
            if 'frost_resistance' in plant_params.keys():
                frost_resistance = float(plant_params.get('frost_resistance')[0])
                frost_resistance_period = int(plant_params.get('frost_resistance_days')[0])
                vernalization_params = [vernalization_period, vernalization_tmax, vernalization_tmin, frost_resistance, frost_resistance_period, vernalization_days]
            else:
                vernalization_params = [vernalization_period, vernalization_tmax, vernalization_tmin, 0, 0, vernalization_days]
        else:
            vernalization_params = [0, 0, 0, 0, 0, 0]
    else:
        wintercrop, vernalization_params = False, [0, 0, 0, 0, 0, 0]
        
    return wintercrop, vernalization_params, vernalization_days

def lethal_params_threshold(plant_params):
    if plant_params.get('lethal_thresholds', [0])[0] == 1:
        try:
            lethal_params = [
                int(plant_params.get('lethal_min_temp_duration', [0])[0]),
                int(plant_params.get('lethal_min_temp', [0])[0]),
                int(plant_params.get('lethal_max_temp_duration', [0])[0]),
                int(plant_params.get('lethal_max_temp', [0])[0])]
            lethal = True
            return lethal, lethal_params
        except (ValueError, TypeError, IndexError):
            return False, [0, 0, 0, 0]
    else:
        return False, [0, 0, 0, 0]


def get_photoperiod_params(plant_params, extent):
    
    if 'photoperiod' in plant_params.keys():
        photoperiod = plant_params.get('photoperiod')[0] == 1
        if photoperiod:
            photoperiod_params = [int(plant_params.get('minimum_sunlight_hours')[0]), int(plant_params.get('maximum_sunlight_hours')[0]), 
                                extent[2], extent[0]]
        else:
            photoperiod_params = [0, 24, extent[1], extent[3]]
            
        return photoperiod, photoperiod_params
    else:
        return False, [0, 0, 0, 0]
    
def get_prec_requirements(plant_params):
    if 'prec_req_after_sow' in plant_params.keys():
        return [int(plant_params.get('prec_req_after_sow')[0]), int(plant_params.get('prec_req_days')[0])]
    else:
        return [20, 15]
    
def get_lethal_min_precipitation_duration(plant_params):
    
    if 'lethal_min_prec_duration' in plant_params.keys() and 'lethal_min_prec' in plant_params.keys():
        max_consec_dry_days = int(plant_params.get('lethal_min_prec_duration', 0)[0])
        dry_day_prec = int(plant_params.get('lethal_min_prec', 0)[0])
    else:
        max_consec_dry_days = 0
        dry_day_prec = 0
        
    return max_consec_dry_days, dry_day_prec

def get_lethal_max_precipitation(plant_params):
    
    if 'lethal_max_prec' in plant_params.keys() and 'lethal_max_prec_duration' in plant_params.keys():
        max_prec_val = int(plant_params.get('lethal_max_prec', 0)[0])
        max_prec_dur = int(plant_params.get('lethal_max_prec_duration', 0)[0])
    else:
        max_prec_val = 0
        max_prec_dur = 0
        
    return max_prec_val, max_prec_dur

def get_temp_for_sowing_duration(plant_params):
    
    if 'temp_for_sow_duration' in plant_params.keys() and 'temp_for_sow' in plant_params.keys():
        dursowing = int(plant_params.get('temp_for_sow_duration', 7)[0])
        sowingtemp = int(plant_params.get('temp_for_sow', 5)[0])
    else:
        dursowing, sowingtemp = 7, 5
        
    return dursowing, sowingtemp



def get_id_list_start(dict, starts_with):
    """
    Get a list of IDs from a dictionary where the ID starts with a specified prefix.

    Parameters:
    - dictionary (dict): The input dictionary containing IDs as keys.
    - starts_with (str): The prefix to filter IDs.

    Returns:
    - list: A list of IDs from the dictionary that start with the specified prefix.

    Note:
    - The function iterates through the keys of the dictionary and includes IDs that start with the specified prefix.
    """
    lst = []
    for id, __ in dict.items():
        if str(id).startswith(starts_with):
            lst.append(id)
    return lst

def read_tif_data_to_tempprecfail_arr(ndimensions = 4, tmp_folder: str = 'temp2') -> List:
    """
    Reads temperature, precipitation, and failure suitability arrays from GeoTIFF files.

    Parameters:
    

    Returns:
    - temp_arr (np.ndarray): Array containing temperature values for each day.
    - prec_arr (np.ndarray): Array containing precipitation values for each day.
    - fail_arr (np.ndarray): Array containing failure suitability values for each day.

    Note: This function reads GeoTIFF files for each day and extracts temperature, precipitation, and failure suitability data.
    """
    
    computeddays  = [int(i[:-4]) for i in os.listdir(os.path.join(os.getcwd(), tmp_folder)) if i.endswith('.tif')]
    computeddays.sort()

    with rasterio.open(os.path.join(os.getcwd(), tmp_folder, f'{computeddays[0]}.tif')) as src:
        dtype = src.dtypes[0]
        dayshape = src.shape
    
    temp_arr = np.empty((dayshape[0],dayshape[1],len(computeddays), ndimensions), dtype=dtype)
    
    for i in range(len(computeddays)):
        day = computeddays[i]
        sys.stdout.write(f'     - reading {day}.tif                      '+'\r')
        sys.stdout.flush()
        with rasterio.open(os.path.join(os.getcwd(), tmp_folder, f'{day}.tif')) as src:
            temp_arr[..., i, :] = np.transpose(src.read(), (1, 2, 0))
    sys.stdout.write(f'   -> All files read in successfully                       '+'\r')
    sys.stdout.flush()
    gc.collect()
    return temp_arr[..., 0], temp_arr[..., 1], temp_arr[..., 2], temp_arr[..., 3]

def get_suitable_values(section, config):
    
    section_dict = {}
    parameter_list = [entry.replace('parameters.', '', 1) if entry.startswith('parameters.') else entry for entry in get_id_list_start(config, 'parameters.')]
    parameter_dictionary = {config[f'parameters.{parameter_list[parameter_id]}']['rel_member_func']: parameter_list[parameter_id] for parameter_id in range(len(parameter_list))}
    for param_name, x_vals in section.items():
        
        if '_vals' in param_name:
            y_vals = section.get(param_name.replace('_vals', '_suit'))
            if param_name.replace('_vals', '') not in parameter_dictionary:
                method = 1 if param_name in ['freqcropfail'] else 0
            else:
                parameter_name = parameter_dictionary[param_name.replace('_vals', '')]
                method = int(config[f'parameters.{parameter_name}']['interpolation_method'])
            try:
                if x_vals != sorted(x_vals):
                    x_vals = x_vals[::-1]
                    y_vals = y_vals[::-1]
            except:
                continue
            
            section_dict[param_name.replace('_vals', '')] = { 'x_vals': x_vals, 'y_vals': y_vals}
        
        
    return section_dict

from numba import njit

@njit
def interp_numba(x, old_vals, new_vals):
    return np.interp(x, old_vals, new_vals)

def interp_func(x, **kwargs):
    old_vals = kwargs.get('old_vals', None)
    new_vals = kwargs.get('new_vals', None)
    return np.interp(x, old_vals, new_vals)

def get_suitability_val_dict_xarray(xrdata, plant_params, config, form_type):
    
    #func, min_val, max_val = forms[form_type]['formula'], forms[form_type]['min_val'], forms[form_type]['max_val']
    
    val_dict = get_suitable_values(plant_params, config)[form_type]
    
    old_vals = val_dict['x_vals']
    new_vals = val_dict['y_vals']
    data = xrdata.clip(
        min=float(min(old_vals)),
        max=float(max(old_vals))
    ).astype("float32")
    
    #template = xrdata.astype(float)
    return xarray.apply_ufunc(
                interp_numba,
                data,
                kwargs={"old_vals": old_vals, "new_vals": new_vals},
                #vectorize=True,      # apply elementwise
                dask="parallelized", # allows Dask chunking
                output_dtypes=[xrdata.dtype],
                )

def mask_min_temperature_duration(xrdata, min_dur, min_tmp):
    cons_arr = xarray.zeros_like(xrdata.isel(time = 0), dtype=np.int16)
    for day in range(xrdata.sizes['time']):
        below = xrdata.isel(time=day) < (min_tmp * 10)
        cons_arr = cons_arr.where(~below, cons_arr + 1)        
        cons_arr = cons_arr.where(below | (cons_arr >= min_dur), 0)

    return cons_arr >= min_dur    

def mask_max_temperature_duration(xrdata, max_dur, max_tmp):
    cons_arr = xarray.zeros_like(xrdata.isel(time = 0), dtype=np.int16)
    for day in range(xrdata.sizes['time']):
        below = xrdata.isel(time=day) > (max_tmp * 10)
        cons_arr = cons_arr.where(~below, cons_arr + 1)        
        cons_arr = cons_arr.where(below | (cons_arr >= max_dur), 0)

    return cons_arr >= max_dur  

def mask_max_precipitation_days(xrdata, max_consec_dry_days, max_prec_val, max_prec_dur):

    cons_arr = xarray.zeros_like(xrdata.isel(time = 0), dtype=np.int16)
    for day in range(xrdata.sizes['time']):
        
        below = xrdata.isel(time=day) > (max_prec_val * 10)
        cons_arr = cons_arr.where(~below, cons_arr + 1)
        cons_arr = cons_arr.where(below | (cons_arr >= max_consec_dry_days), 0)
    
    return cons_arr >= max_prec_dur    

def mask_max_consec_dry_days(xrdata, max_consec_dry_days, dry_day_prec):
    
    cons_arr = xarray.zeros_like(xrdata.isel(time = 0), dtype=np.int16)
    for day in range(xrdata.sizes['time']):
            below = xrdata.isel(time=day) < (dry_day_prec * 10)
            cons_arr = cons_arr.where(~below, cons_arr + 1)
            cons_arr = cons_arr.where(below | (cons_arr >= max_consec_dry_days), 0)

    return cons_arr >= max_consec_dry_days

@delayed
def write_raster(xrdata, path):
    
    xrdata.rio.to_raster(path)

def process_day_climsuit_xarray(day, temperature_cycle, precipitation_cycle,
                                            land_sea_mask_clipped, config, plant_params,plant, crop_failures, extent, tmp_folder = 'temp2'):
    """_summary_

    Args:
        day (_type_): _description_
        temp_files (_type_): _description_
        prec_files (_type_): _description_
        config (_type_): _description_
        plant_params (_type_): _description_
        plant (_type_): _description_
        crop_failures (_type_): _description_
        extent (_type_): xyxy

    Raises:
        Warning: _description_

    Returns:
        _type_: _description_
    """
    
    
    #plant_params_formulas = get_plant_param_interp_forms_dict(plant_params, config)[plant]
    
    # temperature_cycle = temperature_cycle.persist()
    # precipitation_cycle = precipitation_cycle.persist()
    # land_sea_mask_clipped = land_sea_mask_clipped.persist()

    tmp = os.path.join(os.getcwd(), tmp_folder)
    len_growing_cycle = int(plant_params[plant]['growing_cycle'][0])
    is_irrigated = int(config['options']['irrigation'])
    
    
    if day + len_growing_cycle <= 365:
        timewindow = slice(day, day + len_growing_cycle)
    else:
        timewindow = [i for i in range(day, 365)] + [i for i in range(0, (day + len_growing_cycle) - 365)]
        
    if len_growing_cycle != 365:
        temperature_cycle = temperature_cycle.isel(time = timewindow)    
        if is_irrigated == 0:
            precipitation_cycle = precipitation_cycle.sel(time = timewindow)
        else:
            precipitation_cycle = xarray.zeros_like(temperature_cycle)
    try:
        mask_layer = land_sea_mask_clipped.squeeze().data ==1
        mask_layer = mask_layer.rename({'y': 'lat', 'x': 'lon'})
    except:
        mask_layer = land_sea_mask_clipped.squeeze() ==1

    temp = temperature_cycle.mean(dim='time')
    temp = temp.where(mask_layer, np.nan)
    
    #temp = temp.chunk({"lat": 100, "lon": 100})
    
    temp_ = get_suitability_val_dict_xarray(temp.data/10,  plant_params[plant], config, 'temp')*100
    temp_ = temp_.astype(np.int8)
    
    
    wintercrop, vernalization_params, vernalization_days = vernalization_params_winter_crops(plant_params=plant_params[plant])
    lethal, lethal_params = lethal_params_threshold(plant_params=plant_params[plant])
    photoperiod, photoperiod_params = get_photoperiod_params(plant_params[plant], extent)
    sowprec_params = get_prec_requirements(plant_params[plant])
    max_consec_dry_days, dry_day_prec = get_lethal_min_precipitation_duration(plant_params[plant])
    max_prec_val, max_prec_dur = get_lethal_max_precipitation(plant_params[plant])
    dursowing, sowingtemp = get_temp_for_sowing_duration(plant_params[plant])

    additional_conditions = [cond for i in range(100) if (cond := plant_params.get(f'AddCon:{i}')) is not None]
    
    
    if not wintercrop and len_growing_cycle < 365:
        sowing_day_max_cond = temperature_cycle.isel(time = slice(0,dursowing)).mean(dim = 'time') <= (sowingtemp * 10)
        temp_ = temp_.where(~sowing_day_max_cond, 0)

    crop_failures = np.zeros((365))

    if is_irrigated == 0:
        prec = precipitation_cycle.sum(dim='time')
        #prec = prec.chunk({"lat": 100, "lon": 100})
        
        prec = prec.where(mask_layer, np.nan)
        if wintercrop:
            # TODO IMPLEMENT
            #prec = ((prec * len_growing_cycle) + (vern_prec * vernalization_params[5])) / (len_growing_cycle + vernalization_params[5])
            pass
        prec_ = get_suitability_val_dict_xarray(prec.data/10,  plant_params[plant], config, 'prec')*100
        prec_ = prec_.where(mask_layer, np.nan)
        prec_ = prec_.astype(np.int8)
    else:
        prec_ = xarray.zeros_like(temp_)
    
    del temp, prec
    # TODO IMPLEMENT crop_failures.ndim 
    if crop_failures.ndim > 1:
        failure_suit = get_suitability_val_dict_xarray(crop_failures/100, plant_params[plant], config,'freqcropfail')*100
        failure_suit = failure_suit.where(mask_layer, np.nan)
        failure_suit = failure_suit.astype(np.int8)
    else:
        failure_suit = xarray.full_like(temp_, 100)
        

    if photoperiod and len_growing_cycle < 365:
        # TODO IMPLEMENT cphotoperiod
        #sunshine_hours = xarray.full_like(temp_, 100)
        raise Warning('not implemeneted yet')
    else:
        sunshine_hours = xarray.full_like(temp_, 100)
        

    if is_irrigated == 0 and len_growing_cycle < 365 and not wintercrop:
        prec_after_sowing = precipitation_cycle.isel(time = slice(0,sowprec_params[1])).sum(dim = 'time')
        sowing_day_prec_cond = prec_after_sowing < sowprec_params[1]*10
        prec_ = prec_.where(~sowing_day_prec_cond, 0)
        
        

    if lethal:
        min_dur, min_tmp = lethal_params[0], lethal_params[1]
        max_dur, max_tmp = lethal_params[2], lethal_params[3]

        if min_dur != 0:
            mask_mintmp = mask_min_temperature_duration(temperature_cycle, min_dur= min_dur, min_tmp= min_tmp)
            temp_ = temp_.where(~mask_mintmp, 0)

        if max_dur != 0:
            mask_mantmp = mask_max_temperature_duration(temperature_cycle, max_dur= max_dur, max_tmp= max_tmp)
            temp_ = temp_.where(~mask_mantmp, 0)

        if max_consec_dry_days > 0 and not is_irrigated == 1:
            if dry_day_prec != 0: dry_day_prec = 1
            mask_drydays = mask_max_consec_dry_days(precipitation_cycle, max_consec_dry_days, dry_day_prec)
            prec_ = prec_.where(~mask_drydays, 0)

        if max_prec_val > 0 and not is_irrigated == 1:
            if max_prec_dur ==0 : max_prec_dur = 3
            if max_prec_val == 0: max_prec_val = 75
            mask_precconscutive = mask_max_precipitation_days(precipitation_cycle, max_consec_dry_days, max_prec_val, max_prec_dur)
            prec_ = prec_.where(~mask_precconscutive, 0)
    #print("precipitation",list(prec_.data_vars)[0])
    #print(prec_)
    new_name = {list(prec_.data_vars)[0]: "precipitation"}
    prec_ = prec_.rename(new_name)
    # print("temperature", list(temp_.data_vars)[0])
    # print(temp_)
    new_name = {list(temp_.data_vars)[0]: "temperature"}
    temp_ = temp_.rename(new_name)
    failure_suit = failure_suit.rename_vars({'data': 'failure_suit'})
    sunshine_hours = sunshine_hours.rename_vars({'data': 'sunshine_hours'})
    
    #temp_path = os.path.join(os.getcwd(), 'temp', f'm_{day}.nc')
    
    
    alldata = xarray.merge([temp_, prec_, failure_suit, sunshine_hours]).load()
    #return xarray.merge([temp_, prec_])
    #encoding = set_encoding(alldata)
    #alldata.to_netcdf(temp_path, encoding = encoding, engine = 'netcdf4')
    # alldata = xarray.Dataset({
    #     "temperature": temp_,
    #     "precipitation": prec_,
    #     "failure_suit": failure_suit,
    #     "sunshine_hours": sunshine_hours
    # })
    temp_path = os.path.join(tmp, f'{day}.tif')
    alldata.rio.to_raster(temp_path)
    gc.collect()
    
    #return alldata
    #return delayed(write_raster)(alldata, temp_path)


def process_day_concfut_using_xarray(temperature_cycle, precipitation_cycle, land_sea_mask_clipped, extent, climate_config, plant_params, plant, day_interval = 1, tmp_folder: str = 'temp2'):
    import dask
    len_growing_cycle = int(plant_params[plant]['growing_cycle'][0])
    
    crop_failures = np.zeros((365))
    
    temperature_cycle = temperature_cycle.chunk({"time": -1, "lat": 100, "lon": 100}).persist()
    precipitation_cycle = precipitation_cycle.chunk({"time": -1, "lat": 100, "lon": 100}).persist()

    results = []
    
    
    if len_growing_cycle >= 365:
        process_day_climsuit_xarray(day = 0, temperature_cycle=temperature_cycle, precipitation_cycle=precipitation_cycle,
                        land_sea_mask_clipped = land_sea_mask_clipped.data,
                        config=climate_config, plant_params=plant_params, 
                        plant = plant, crop_failures=crop_failures, extent=extent, tmp_folder = tmp_folder)
        
    else:
        for day in tqdm(range(0, 365, int(day_interval))):
            results.append(delayed(process_day_climsuit_xarray)(day = day, temperature_cycle=temperature_cycle, precipitation_cycle=precipitation_cycle,
                        land_sea_mask_clipped = land_sea_mask_clipped,
                        config=climate_config, plant_params=plant_params, 
                        plant = plant, crop_failures=crop_failures, extent=extent, tmp_folder = tmp_folder))
    with dask.config.set(num_workers=4):
        dask.compute(*results)

    return None

def read_tif_data_to_opt_sow_date_arr(shape):
    with rasterio.open(os.path.join(os.getcwd(), 'temp', '0_osd.tif')) as src:
        dtype = src.dtypes[0]
    temp_arr = np.empty(shape, dtype=dtype)
    for day in range(365):
        sys.stdout.write(f'     - reading {day}.tif                      '+'\r')
        sys.stdout.flush()
        with rasterio.open(os.path.join(os.getcwd(), 'temp', f'{day}_osd.tif')) as src:
            temp_arr[..., day] = src.read(1)
    sys.stdout.write(f'   -> All files read in successfully                       '+'\r')
    sys.stdout.flush()
    gc.collect()
    return temp_arr


def climate_suitability_xarray(climate_config, extent, temp_files, prec_files, plant_params, results_path, area_name, day_interval = 1, tmp_folder: str = 'temp2') -> list:
    """
    Calculates climate suitability for multiple plants based on the CropSuite model.

    Parameters:
    - climate_config (dict): Configuration settings for the CLIMSUITE model.
    - extent (list): List containing the spatial extent of the analysis area in the format [min_lon, min_lat, max_lon, max_lat].
    - climatedata (np.ndarray): 4D array containing climate data in the format (X, Y, Day, Plant).
    - land_sea_mask (np.ndarray): 2D array defining land (1) and sea (0) locations.
    - plant_params (dict): Plant-specific parameters.
    - plant_params_formulas (dict): Plant-specific formulas for CLIMSUITE calculations.
    - results_path (str): Path to the directory where the results will be saved.

    Returns:
    - fuzzy_clim (np.ndarray): Array containing the fuzzy climate suitability values for each plant.
    - start_growing_cycle (np.ndarray): Array containing the optimal sowing dates for each plant.
    - length_of_growing_period (np.ndarray): Array containing the suitable sowing days for each plant.
    - limiting_factor (np.ndarray): Array containing the limiting factors for each plant.
    - multiple_cropping (np.ndarray): Array containing information about multiple cropping for each plant.
    - day_interval (int): increment of days used to calculate climate suitability conditions. Default 1

    Note: This function processes climate data for multiple plants using the CropSuite model and aggregates the results.
    """
    plant_list = [plant for plant in plant_params]
    
    ret_paths = []
    temperature_cycle = xarray.open_mfdataset(temp_files,  combine='nested', concat_dim="time")
    temperature_cycle = temperature_cycle.sel(lon = slice(min(extent[1], extent[3]), max(extent[1], extent[3])), lat = slice(max(extent[0], extent[2]), min(extent[0], extent[2])))

    precipitation_cycle = xarray.open_mfdataset(prec_files,  combine='nested', concat_dim="time") #                                       combine='by_coords')
    precipitation_cycle = precipitation_cycle.sel(lon = slice(min(extent[1], extent[3]), max(extent[1], extent[3])), lat = slice(max(extent[0], extent[2]), min(extent[0], extent[2])))

    with rio.open_rasterio(climate_config['files']['land_sea_mask']) as land_sea_mask_clipped:
        land_sea_mask_clipped = land_sea_mask_clipped.sel(x = slice(min(extent[1], extent[3]), max(extent[1], extent[3])), y = slice(max(extent[0], extent[2]), min(extent[0], extent[2])))

    # interpolate mask data
    
    ref_layer = temperature_cycle.isel(time = 0).rename({'lat': 'y', 'lon': 'x'}).set_index(y='y', x='x').squeeze()
    ref_layer.rio.write_crs("epsg:4326", inplace=True)

    land_sea_mask_clipped = BaseProcessor().mask_data(land_sea_mask_clipped.squeeze(),
                                            ref_layer, 'nearest')
    
    land_sea_mask = land_sea_mask_clipped.data.values==1
    
    for idx, plant in enumerate(plant_params):

        if climate_config['climatevariability'].get('consider_variability', False):
            ret_paths = [os.path.join(os.path.split(results_path)[0]+'_var',\
                                    os.path.split(results_path)[1]), os.path.join(os.path.split(results_path)[0]+'_novar', os.path.split(results_path)[1])]
            if os.path.exists(os.path.join(ret_paths[1], plant, 'climate_suitability.tif')):
                print(f' -> {plant} already created. Skipping')
                continue
        else:
            ret_paths = [os.path.join(os.path.split(results_path)[0]+'_novar', os.path.split(results_path)[1])]
            if os.path.exists(os.path.join(ret_paths[0], plant, 'climate_suitability.tif')):
                print(f' -> {plant} already created. Skipping')
                continue

        print(f'\nProcessing {plant} - {idx+1} out of {len(plant_list)} crops\n')
        
        
        
        process_day_concfut_using_xarray(temperature_cycle, precipitation_cycle, land_sea_mask_clipped, extent, climate_config, plant_params, plant, day_interval = day_interval, tmp_folder = tmp_folder)
        
        compute_suitability(climate_config, results_path, plant_params, plant, land_sea_mask, extent, tmp_folder = tmp_folder)
        
        gc.collect()

    print('Climate suitability calculation finished!\n\n')
    return ret_paths



def compute_suitability(climate_config, results_path, plant_params, plant, land_sea_mask, extent, tmp_folder: str = 'temp2'):
    
    no_threads = int(climate_config['options'].get('max_workers', None))
    
    tmp = os.path.join(os.getcwd(), tmp_folder)
    
    water_mask = (land_sea_mask == 1)
    
    ret_paths = []
    
    len_growing_cycle = int(plant_params[plant]['growing_cycle'][0])
    
    fuzzy_temp_growing_cycle_wdoy, fuzzy_precip_growing_cycle_wdoy, fuzzy_fail_growing_cycle_wdoy, fuzzy_photop_growing_cycle_wdoy = read_tif_data_to_tempprecfail_arr(tmp_folder = tmp_folder)
    final_shape = fuzzy_temp_growing_cycle_wdoy.shape
    print(final_shape)
    fuzzy_clim = np.zeros((final_shape[0], final_shape[1]), dtype=np.int8)
    start_growing_cycle = np.zeros((final_shape[0], final_shape[1]), dtype=np.int16)
    length_of_growing_period = np.zeros((final_shape[0], final_shape[1]), dtype=np.int16)
    
    curr_fail = np.full_like(fuzzy_temp_growing_cycle_wdoy, 100)
    res_path = os.path.join(os.path.split(results_path)[0]+'_novar', os.path.split(results_path)[1], plant)

    os.makedirs(res_path, exist_ok=True)
    ret_paths.append(res_path)
    multiple_cropping = np.zeros((final_shape[0], final_shape[1]), dtype=np.int8)
    fuzzy_clim_growing_cycle_wdoy = np.min([fuzzy_temp_growing_cycle_wdoy, fuzzy_precip_growing_cycle_wdoy, curr_fail, fuzzy_photop_growing_cycle_wdoy], axis=0)

    
    length_of_growing_period = np.zeros((final_shape[0], final_shape[1]), dtype=np.int16)
    
    print(' -> Calculating Suitable Sowing Days')
    if len_growing_cycle >= 365:
        length_of_growing_period = np.zeros_like(fuzzy_clim_growing_cycle_wdoy, dtype=np.int16)
        if climate_config['options']['output_grow_cycle_as_doy']:
            length_of_growing_period[fuzzy_clim_growing_cycle_wdoy > 0] = 365
        else:
            length_of_growing_period[fuzzy_clim_growing_cycle_wdoy > 0] = 52
    else:
        length_of_growing_period = (fuzzy_clim_growing_cycle_wdoy > 0).sum(axis=2).astype(np.int16)
    
    if len_growing_cycle >= 365:
            fuzzy_clim[water_mask] = fuzzy_clim_growing_cycle_wdoy[water_mask]
            print(' -> Calculating Optimal Sowing Date')                
            start_growing_cycle = np.full_like(fuzzy_clim_growing_cycle_wdoy, 0)
            print(' -> Calculating Limiting Factor')
            limiting_factor = np.argmin([fuzzy_temp_growing_cycle_wdoy, fuzzy_precip_growing_cycle_wdoy, curr_fail, fuzzy_photop_growing_cycle_wdoy], axis=0).astype(np.int8)
            temp_suit = fuzzy_temp_growing_cycle_wdoy
            prec_suit = fuzzy_precip_growing_cycle_wdoy
            cffq_suit = curr_fail
            photoperiod_suit = fuzzy_photop_growing_cycle_wdoy
    else:
        print(' -> Calculating Optimal Sowing Date')
        ## TODO: winter crop   
        fuzzy_clim[water_mask] = np.max(fuzzy_clim_growing_cycle_wdoy[water_mask], axis=1)
        start_growing_cycle[water_mask] = np.argmax(fuzzy_clim_growing_cycle_wdoy[water_mask, :], axis=1)

        # For determination of limiting factor:
        suit_sum = fuzzy_temp_growing_cycle_wdoy.astype(np.int16) + fuzzy_precip_growing_cycle_wdoy.astype(np.int16) + curr_fail.astype(np.int16) + fuzzy_photop_growing_cycle_wdoy.astype(np.int16)
        start_growing_cycle[water_mask & (start_growing_cycle <= 0)] = np.argmax(suit_sum[water_mask & (start_growing_cycle <= 0)], axis=1)

        print(' -> Calculating Limiting Factor')
        temp_suit = fuzzy_temp_growing_cycle_wdoy[np.arange(final_shape[0])[:,None], np.arange(final_shape[1])[None,:], start_growing_cycle] #type:ignore
        temp_suit = np.clip(temp_suit, 0,100)
        prec_suit = fuzzy_precip_growing_cycle_wdoy[np.arange(final_shape[0])[:,None], np.arange(final_shape[1])[None,:], start_growing_cycle] #type:ignore
        prec_suit = np.clip(prec_suit, 0,100)
        cffq_suit = curr_fail[np.arange(final_shape[0])[:,None], np.arange(final_shape[1])[None,:], start_growing_cycle] #type:ignore
        photoperiod_suit = fuzzy_photop_growing_cycle_wdoy[np.arange(final_shape[0])[:,None], np.arange(final_shape[1])[None,:], start_growing_cycle] #type:ignore
        limiting_factor = np.argmin([temp_suit, prec_suit, cffq_suit, photoperiod_suit], axis=0).astype(np.int8)

    start_growing_cycle[fuzzy_clim == 0] = -1

    limiting_factor[~water_mask] = -1
    gc.collect()
        
    fuzzy_clim = np.clip(fuzzy_clim, 0, 100)
    
    # Wintercrops: Growing_Cycle = Growing_cycle + Vernalisation Period
    growing_cycle_wdoy = len_growing_cycle if climate_config['options']['output_grow_cycle_as_doy'] else len_growing_cycle // 7

    
    threshold_time = 365 if climate_config['options']['output_grow_cycle_as_doy'] else 52
    # if not wintercrop: TODO: WINTER CROP
    print(' -> Calculting Potential Multiple Cropping')
    if 'multiple_cropping_turnaround_time' in climate_config['options']:
        try:
            turnaround_time = int(climate_config['options'].get('multiple_cropping_turnaround_time')) #type:ignore
        except:
            turnaround_time = 21 if climate_config['options']['output_grow_cycle_as_doy'] else 3
    else:
        turnaround_time = 21 if climate_config['options']['output_grow_cycle_as_doy'] else 3
    multiple_cropping[length_of_growing_period < 4 * growing_cycle_wdoy] = 3
    multiple_cropping[length_of_growing_period < 3 * growing_cycle_wdoy] = 2
    multiple_cropping[length_of_growing_period < 2 * growing_cycle_wdoy] = 1
    multiple_cropping[length_of_growing_period == 0] = 0
    multiple_cropping[multiple_cropping >= 3] = 3

    if (3 * growing_cycle_wdoy + 3 * turnaround_time) > threshold_time:
        multiple_cropping[multiple_cropping >= 2] = 2

    if (2 * growing_cycle_wdoy + 2 * turnaround_time) <= 365 and climate_config['options']['output_all_startdates']:
        print(' -> Calculation of Sowing Days for Multiple Cropping')
        start_days = np.empty(start_growing_cycle.shape + (4,), dtype=np.int16)       

        def process_index(idx):
            i, j = idx
            suit_vals = fuzzy_clim_growing_cycle_wdoy[i, j].astype(np.int16)
            start_idx, max_sum = find_max_sum_new(suit_vals, growing_cycle_wdoy + turnaround_time, multiple_cropping[i, j])
            
            multiple_cropping[i, j] = min(1 if np.max(suit_vals) >= np.sum(suit_vals[start_idx]) else np.sum(suit_vals[start_idx] > 1), multiple_cropping[i, j])
            if multiple_cropping[i, j] > 1:
                if len(start_idx) == 2:
                    start_days[i, j, :] = [max_sum, start_idx[0], start_idx[1], -1]
                    multiple_cropping[i, j] = 2
                else:
                    start_days[i, j, :] = [max_sum, start_idx[0], start_idx[1], start_idx[2]]
                    multiple_cropping[i, j] = 3
        
        valid_indices = np.argwhere(multiple_cropping >= 2)
        print(f' -> Processing {len(valid_indices)} pixels...')

        """
        # DEBUG
        for indices in valid_indices:
            process_index(indices)
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=no_threads) as executor:
            list(executor.map(process_index, valid_indices, chunksize=len(valid_indices)//no_threads))

        start_days[..., 1:] += 1
        start_days[multiple_cropping < 2, 1] = -1 
        start_days[multiple_cropping < 2, 2] = -1
        start_days[multiple_cropping < 3, 3] = -1 
        start_days[multiple_cropping < 2, 0] = fuzzy_clim[multiple_cropping < 2]
        start_days[land_sea_mask == 0, 0] = -1 

        if climate_config['options']['output_format'] == 'geotiff' or climate_config['options']['output_format'] == 'cog':
            write_geotiff(res_path, 'optimal_sowing_date_mc_first.tif', start_days[..., 1]*land_sea_mask, extent, nodata_value=-1, cog=climate_config['options']['output_format'].lower() == 'cog')
            write_geotiff(res_path, 'optimal_sowing_date_mc_second.tif', start_days[..., 2]*land_sea_mask, extent, nodata_value=-1, cog=climate_config['options']['output_format'].lower() == 'cog')
            if np.nanmax(start_days[..., 3]) > 0:
                write_geotiff(res_path, 'optimal_sowing_date_mc_third.tif', start_days[..., 3]*land_sea_mask, extent, nodata_value=-1, cog=climate_config['options']['output_format'].lower() == 'cog')
            write_geotiff(res_path, 'climate_suitability_mc.tif', start_days[..., 0]*land_sea_mask, extent, nodata_value=-1, cog=climate_config['options']['output_format'].lower() == 'cog')
        elif climate_config['options']['output_format'] == 'netcdf4':
            write_to_netcdf(start_days[..., 1]*land_sea_mask, os.path.join(res_path, 'optimal_sowing_date_mc_first.nc'), extent=extent, compress=True, var_name='optimal_sowing_date_mc_first', nodata_value=-1) #type:ignore
            write_to_netcdf(start_days[..., 2]*land_sea_mask, os.path.join(res_path, 'optimal_sowing_date_mc_second.nc'), extent=extent, compress=True, var_name='optimal_sowing_date_mc_second', nodata_value=-1) #type:ignore
            if np.nanmax(start_days[..., 3]) > 0:
                write_to_netcdf(start_days[..., 3]*land_sea_mask, os.path.join(res_path, 'optimal_sowing_date_mc_third.nc'), extent=extent, compress=True, var_name='optimal_sowing_date_mc_third', nodata_value=-1) #type:ignore
            write_to_netcdf(start_days[..., 0]*land_sea_mask, os.path.join(res_path, 'climate_suitability_mc.nc'), extent=extent, compress=True, var_name='climate_suitability_mc', nodata_value=-1) #type:ignore

        
    fuzzy_clim[land_sea_mask == 0] = -1
    start_growing_cycle += 1
    start_growing_cycle[land_sea_mask == 0] = -1
    multiple_cropping[land_sea_mask == 0] = -1
    length_of_growing_period[land_sea_mask == 0] = -1
    limiting_factor[land_sea_mask == 0] = -1
    temp_suit[land_sea_mask == 0] = -1
    prec_suit[land_sea_mask == 0] = -1
    cffq_suit[land_sea_mask == 0] = -1
    photoperiod_suit[land_sea_mask == 0] = -1

    #np.save(os.path.join(res_path, 'fuzzy_clim.npy'), fuzzy_clim)
    #np.save(os.path.join(res_path, 'optimal_sowing_date.npy'), start_growing_cycle)
    #np.save(os.path.join(res_path, 'multiple_cropping.npy'), multiple_cropping)
    #np.save(os.path.join(res_path, 'suitable_sowing_days.npy'), length_of_growing_period)
    #np.save(os.path.join(res_path, 'limiting_factor.npy'), limiting_factor)

    if climate_config['options']['output_all_limiting_factors']:
        all_array = np.asarray([temp_suit, prec_suit, cffq_suit, photoperiod_suit])
        all_array[np.isnan(all_array)] = -1
        all_array = all_array.astype(np.int8)
        write_geotiff(res_path, 'all_climlim_factors.tif', np.transpose(all_array, (1, 2, 0)), extent, nodata_value=-1)
        del all_array

    if climate_config['options']['output_format'] == 'geotiff' or climate_config['options']['output_format'].lower() == 'cog':
        write_geotiff(res_path, 'limiting_factor.tif', limiting_factor, extent, nodata_value=-1, cog=climate_config['options']['output_format'].lower() == 'cog')
        write_geotiff(res_path, 'climate_suitability.tif', fuzzy_clim, extent, nodata_value=-1, cog=climate_config['options']['output_format'].lower() == 'cog')
        # if wintercrop:
        #     write_geotiff(res_path, 'optimal_sowing_date_with_vernalization.tif', start_growing_cycle, extent, nodata_value=-1, cog=climate_config['options']['output_format'].lower() == 'cog')
        #     write_geotiff(res_path, 'start_growing_cycle_after_vernalization.tif', start_growing_cycle_without_vern, extent, nodata_value=-1, cog=climate_config['options']['output_format'].lower() == 'cog')
        # else:
        write_geotiff(res_path, 'optimal_sowing_date.tif', start_growing_cycle, extent, nodata_value=-1, cog=climate_config['options']['output_format'].lower() == 'cog')
        write_geotiff(res_path, 'multiple_cropping.tif', multiple_cropping, extent, nodata_value=-1, cog=climate_config['options']['output_format'].lower() == 'cog')
        write_geotiff(res_path, 'suitable_sowing_days.tif', length_of_growing_period, extent, nodata_value=-1, cog=climate_config['options']['output_format'].lower() == 'cog')
    elif climate_config['options']['output_format'] == 'netcdf4':
        write_to_netcdf(fuzzy_clim, os.path.join(res_path, 'climate_suitability.nc'), extent=extent, compress=True, var_name='climate_suitability', nodata_value=-1) #type:ignore
        write_to_netcdf(limiting_factor.astype(np.uint8)+1, os.path.join(res_path, 'limiting_factor.nc'), extent=extent, compress=True, var_name='limiting_factor', nodata_value=-1) #type:ignore
        # if wintercrop:
        #     write_to_netcdf(start_growing_cycle, os.path.join(res_path, 'optimal_sowing_date_with_vernalization.nc'), extent=extent, compress=True, var_name='optimal_sowing_date_with_vernalization', nodata_value=-1) #type:ignore
        #     write_to_netcdf(start_growing_cycle_without_vern, os.path.join(res_path, 'start_growing_cycle_after_vernalization.nc'), extent=extent, compress=True, var_name='start_growing_cycle_after_vernalization', nodata_value=-1) #type:ignore
        # else:
        write_to_netcdf(start_growing_cycle, os.path.join(res_path, 'optimal_sowing_date.nc'), extent=extent, compress=True, var_name='optimal_sowing_date', nodata_value=-1) #type:ignore
        write_to_netcdf(multiple_cropping, os.path.join(res_path, 'multiple_cropping.nc'), extent=extent, compress=True, var_name='multiple_cropping', nodata_value=-1) #type:ignore
        write_to_netcdf(length_of_growing_period, os.path.join(res_path, 'suitable_sowing_days.nc'), extent=extent, compress=True, var_name='suitable_sowing_days', nodata_value=-1) #type:ignore
    else:
        print('No output format specified.')

    length_of_growing_period = np.zeros((final_shape[0], final_shape[1]), dtype=np.int16)
    multiple_cropping = np.zeros((final_shape[0], final_shape[1]), dtype=np.int8)
    start_growing_cycle = np.zeros((final_shape[0], final_shape[1]), dtype=np.int16)
    fuzzy_clim = np.zeros((final_shape[0], final_shape[1]), dtype=np.int8)
    limiting_factor = np.zeros((final_shape[0], final_shape[1]), dtype=np.int8)
    del curr_fail

    del fuzzy_temp_growing_cycle_wdoy, fuzzy_precip_growing_cycle_wdoy, fuzzy_fail_growing_cycle_wdoy
    for i in range(365):
        try:
            os.remove(os.path.join(tmp, f'{i}.npy'))
        except:
            pass
        try:
            os.remove(os.path.join(tmp, f'{i}.tif'))
            os.remove(os.path.join(tmp, f'{i}_osd.tif'))
        except:
            pass
    gc.collect()

    ret_paths = [os.path.split(pt)[0] for pt in ret_paths]
    
    

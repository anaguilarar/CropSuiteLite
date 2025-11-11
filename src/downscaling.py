import os
from concurrent.futures import ProcessPoolExecutor
import sys

import numpy as np
import rasterio
from skimage import transform as skt

from .data_tools import (get_resolution_array, load_specified_lines, interpolate_nanmask, 
                         extract_domain_from_global_3draster, get_cpu_ram, extract_domain_from_global_raster)

from .nc_tools import (read_area_from_netcdf, get_nodata_value, get_variable_name_from_nc, write_to_netcdf)

def read_timestep(filename, extent, timestep=-1):
    if timestep == -1:
        if filename.endswith('.tif') or filename.endswith('.tiff'):
            with rasterio.open(filename) as src:
                data = np.asarray(src.read(), dtype=np.float16)
                bounds = src.bounds
                nodata = src.nodata
            data = extract_domain_from_global_raster(data, extent, raster_extent=bounds)
            return data, nodata
        else:
            data, _ = read_area_from_netcdf(filename, extent=[extent[1], extent[0], extent[3], extent[2]])
            data = np.asarray(data).transpose(2, 0, 1)
            nodata = get_nodata_value(filename)
            return data, nodata
    else:
        if filename.endswith('.tif') or filename.endswith('.tiff'):
            data, nodata = load_specified_lines(filename, extent, all_bands=timestep+1) #type:ignore
            return data, nodata
        else:
            varname = get_variable_name_from_nc(filename)
            data, _ = read_area_from_netcdf(filename, extent=extent, day_range=timestep, variable=varname)
            nodata = get_nodata_value(filename)
            return data, nodata
        


def process_tempday_interp(day, temp_file, extent, fine_resolution, land_sea_mask, output_dir, mode='nearest'):
    
    if os.path.exists(os.path.join(output_dir, f'ds_temp_{day}.tif')) or os.path.exists(os.path.join(output_dir, f'ds_temp_{day}.nc')):
        return f"Day {day + 1} skipped (already processed)."
    sys.stdout.write(f'     - Downscaling of temperature data for day #{day+1}                      '+'\r')
    sys.stdout.flush()
    temp_data, temp_nodata = read_timestep(temp_file, extent=extent, timestep=day)
    temp_data[(temp_data < -100) | (temp_data > 60) |(temp_data == temp_nodata)] = np.nan
    if mode == 'bilinear':
        temp_data = skt.resize(temp_data * 10, fine_resolution, order=1, mode='edge', anti_aliasing=False).astype(np.int16)
    elif mode == 'nearest':
        ratio = fine_resolution[0] // temp_data.shape[0]
        temp_data[(temp_data < -100) | (temp_data > 60) |(temp_data == temp_nodata)] = np.nan
        temp_data = (np.repeat(np.repeat(temp_data * 10, ratio, axis=0), ratio, axis=1)).astype(np.int16)
    else:
        raise ValueError(f'there is not an implementation for {mode}, only bilinear and nearest are supported')
    
    temp_data[np.isnan(land_sea_mask)] = -32767
    write_to_netcdf(temp_data, os.path.join(output_dir, f'ds_temp_{day}.nc'), extent=extent, compress=True, nodata_value=-32767) #type:ignore
    return ''

def process_precday_interp(day, prec_data, extent, prec_thres, fine_dem_shape, land_sea_mask, output_dir, prec_nodata, mode='nearest'):
    if os.path.exists(os.path.join(output_dir, f'ds_prec_{day}.tif')) or os.path.exists(os.path.join(output_dir, f'ds_prec_{day}.nc')):
        return f"Day {day + 1} skipped (already processed)."
    sys.stdout.write(f'     - Downscaling of precipitation data for day #{day + 1}                      ' + '\r')
    sys.stdout.flush()
    prec_data[prec_data < prec_thres] = 0
    prec_data[(prec_data < 0) | (prec_data == prec_nodata)] = np.nan
    prec_data *= 10
    if mode == 'bilinear':
        prec_data = skt.resize(prec_data, fine_dem_shape, order=1, mode='edge', anti_aliasing=False).astype(np.int16)
    elif mode == 'nearest':
        prec_data = (np.repeat(np.repeat(prec_data, int(fine_dem_shape[0] / prec_data.shape[0]), axis=0), int(fine_dem_shape[0] / prec_data.shape[0]), axis=1)).astype(np.int16)
    prec_data[np.isnan(land_sea_mask)] = -32767
    write_to_netcdf(prec_data, os.path.join(output_dir, f'ds_prec_{day}.nc'), extent=extent, compress=True, nodata_value=-32767) #type:ignore
    return ''

def _create_interpolation_folders(config_file, area_name, variable = 'precipitation'):

    interpolation_method = int(config_file['options'][f'{variable}_downscaling_method'])
    output_dir = config_file['files']['output_dir']
    if os.path.basename(output_dir) == '': output_dir = output_dir[:-1]
    output_dir = os.path.join(output_dir+'_downscaled', area_name)
    
    print('output_dir:',output_dir)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir, interpolation_method

def interpolate_precipitation(config_file, domain, area_name):
    """
        domain: [y_max, x_min, y_min, x_max]
    """
    output_dir, interpolation_method = _create_interpolation_folders(config_file, area_name, variable='precipitation')
    if interpolation_method == 0:
        prec_files = interpolate_precipitation_method(config_file, domain, output_dir, method= 'nearest')
    elif interpolation_method == 1:
        prec_files = interpolate_precipitation_method(config_file, domain, output_dir, method = 'bilinear')
    elif interpolation_method == 2:
        raise ValueError('Wodclim not implemented')
        
    return prec_files, True


def setup_initial_layers(config_file, extent, variable = 'Prec'):
    climate_data_dir = os.path.join(config_file['files']['climate_data_dir'])
    fine_resolution = get_resolution_array(config_file, extent, True)
    land_sea_mask, _ = load_specified_lines(config_file['files']['land_sea_mask'], extent, False)
    land_sea_mask = np.asarray(land_sea_mask).astype(np.float16)
    land_sea_mask[land_sea_mask == 0] = np.nan
    if land_sea_mask.shape != fine_resolution:
        land_sea_mask = interpolate_nanmask(land_sea_mask, fine_resolution)
    if not os.path.exists(os.path.join(climate_data_dir, f'{variable}_avg.tif')):
        fn = os.path.join(climate_data_dir, f'{variable}_avg.nc')
    else: fn = os.path.join(climate_data_dir, f'{variable}_avg.tif')
    
    
    return land_sea_mask, fine_resolution, fn



def interpolate_temperature_method(config_file, extent, output_dir, method = 'bilinear'):
    assert method in ['bilinear', 'nearest']
    
    land_sea_mask, fine_resolution, temp_file = setup_initial_layers(config_file, extent, variable = 'Temp')

    area = int((extent[0] - extent[2]) * (extent[3] - extent[1]))
    worker_flag = config_file['options'].get('max_workers', None)
    if worker_flag is None:
        worker = np.clip(int((get_cpu_ram()[1] / area) * 1200), 1, get_cpu_ram()[0]-1)
    else:
        worker = int(worker_flag)
    print(f'Using {worker} workers for Temperature {method} interpolation')

    with ProcessPoolExecutor(max_workers=worker) as executor:
        tasks = [executor.submit(process_tempday_interp, day, temp_file, extent, fine_resolution, land_sea_mask, output_dir, method
                                ) for day in range(365) if not os.path.exists(os.path.join(output_dir, f'ds_temp_{day}.nc'))]
        for future in tasks:
            future.result()
    return [os.path.join(output_dir, f'ds_temp_{day}.nc') for day in range(0, 365)]



def interpolate_precipitation_method(config_file, extent, output_dir, method = 'bilinear'):
    assert method in ['bilinear', 'nearest']
    
    prec_thres = float(config_file['options'].get('downscaling_precipitation_per_day_threshold', 0.75))
    land_sea_mask, fine_resolution, prec_file = setup_initial_layers(config_file, extent, variable='Prec')
    if prec_file.endswith('.nc'):
        prec_data, nodata = load_specified_lines(prec_file, extent=extent)
    else:
        with rasterio.open(prec_file, 'r') as src:
            prec_data = src.read()
            nodata = src.nodata
            pbounds = src.bounds
        prec_data = extract_domain_from_global_3draster(prec_data, [extent[1], extent[0], extent[3], extent[2]], [pbounds.left, pbounds.top, pbounds.right, pbounds.bottom])

    area = int((extent[0] - extent[2]) * (extent[3] - extent[1]))
    worker_flag = config_file['options'].get('max_workers', None)
    if worker_flag is None:
        worker = np.clip(int((get_cpu_ram()[1] / area) * 1200), 1, get_cpu_ram()[0]-1)
    else:
        worker = int(worker_flag)
    print(f'Using {worker} workers for Precipitation {method} interpolation')
    with ProcessPoolExecutor(max_workers=worker) as executor:
        tasks = [executor.submit(process_precday_interp, day, prec_data[day], extent, prec_thres, fine_resolution, land_sea_mask, output_dir, nodata, method) for day in range(365) if not os.path.exists(os.path.join(output_dir, f'ds_prec_{day}.nc'))]
        for future in tasks:
            future.result()
    
    return [os.path.join(output_dir, f'ds_prec_{day}.nc') for day in range(0, 365)]

def interpolate_temperature(config_file, domain, area_name):
    """
        domain: [y_max, x_min, y_min, x_max]
    """
    output_dir, interpolation_method = _create_interpolation_folders(config_file, area_name, variable='temperature')
    if interpolation_method == 0:
        temp_files = interpolate_temperature_method(config_file, domain, output_dir, method = 'nearest')
    elif interpolation_method == 1:
        temp_files = interpolate_temperature_method(config_file, domain, output_dir, method = 'bilinear')
    elif interpolation_method == 2:
        raise ValueError('Wodclim not implemented')
    elif interpolation_method == 3:
        raise ValueError('Temperature height not implemented')
    return temp_files, True

    

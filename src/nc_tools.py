import concurrent.futures
from datetime import datetime
import os
import shutil
import netCDF4 as nc4
from dask.diagnostics import ProgressBar 
from dask.distributed import Client
import numpy as np
import xarray as xr
from tqdm import tqdm

from pyproj import CRS

def get_variable_name_from_nc(ds):
    if isinstance(ds, str):
        ds = xr.open_dataset(ds)
    return list(ds.data_vars)[0]

def get_rows_cols(filename):
    dataset = xr.open_dataset(filename)
    return (dataset.dims['lat'], dataset.dims['lon'])


def get_nodata_value(nc_file_path, variable='data'):
    try:
        ds = xr.open_dataset(nc_file_path)
        var = ds[variable]
        nodata_value = var.attrs['_FillValue'] 
        ds.close()
    except:
        nodata_value = -9999
    return nodata_value

def get_netcdf_extent(file_path):
    """
    Get the spatial extent (min and max) of the latitude and longitude in a NetCDF file.

    Parameters:
    file_path (str): Path to the NetCDF file.

    Returns:
    dict: A dictionary with keys 'lon_min', 'lon_max', 'lat_min', and 'lat_max'.
    """
    # Open the NetCDF file
    ds = xr.open_dataset(file_path)
    
    # Check if 'latitude' and 'longitude' exist, otherwise, look for other possible names
    if 'latitude' in ds.coords and 'longitude' in ds.coords:
        lat = ds['latitude']
        lon = ds['longitude']
    elif 'lat' in ds.coords and 'lon' in ds.coords:
        lat = ds['lat']
        lon = ds['lon']
    else:
        raise ValueError("Could not find latitude and longitude coordinates in the dataset.")
    
    # Get the extent
    extent = {
        'left': float(lon.min()),
        'right': float(lon.max()),
        'bottom': float(lat.min()),
        'top': float(lat.max())
    }
    
    return extent

def merge_netcdf_files(file_list, output_file, overlap=0, nodata_value=False, info_text=False):
    """
    Merge multiple NetCDF files based on latitude and longitude coordinates

    Parameters:
    - file_list (list): List of filenames of NetCDF files covering different latitude ranges.
    - output_file (str): The filename for the merged NetCDF file covering the area from 17°N to 3°S.
    - compress (bool, optional): Whether to apply compression. Default is True.
    - complevel (int, optional): Compression level (1-9). Default is 4.

    Returns:
    - output_file (str)
    """
    
    if len(file_list) == 1:
        shutil.copy(file_list[0], output_file)
        return output_file

    print(f'Merging {os.path.basename(output_file)}')
    # Open the NetCDF files as Dask-backed datasets
    datasets = [xr.open_dataset(file, chunks={'lat': 'auto', 'lon': 'auto'}) for file in file_list]
    try:
        sorted_datasets = sorted(datasets, key=lambda ds: ds.latitude.min(), reverse=True)
    except:
        sorted_datasets = sorted(datasets, key=lambda ds: ds.lat.min(), reverse=True)

    y_max, y_min = 0, 0
    for ds in datasets:
        if y_max < ds.lat.max():
            y_max = float(ds.lat.max())
        if y_min > ds.lat.min():
            y_min = float(ds.lat.min())

    if overlap > 0:
        selected_datasets = []
        for i, ds in enumerate(sorted_datasets):
            if i == 0:
                selected_datasets.append(ds.sel(lat=slice(ds.lat.max(), ds.lat.min() + overlap/2)))
            elif i == len(sorted_datasets) - 1:
                selected_datasets.append(ds.sel(lat=slice(ds.lat.max() - overlap/2, ds.lat.min())))
            else:
                selected_datasets.append(ds.sel(lat=slice(ds.lat.max() - overlap/2, ds.lat.min() + overlap/2)))
        merged_data = xr.concat(selected_datasets, dim='lat')
    else:
        merged_data = xr.concat(sorted_datasets, dim='lat')

    if isinstance(nodata_value, (int, float)):
        merged_data.attrs['_FillValue'] = nodata_value

    ds.attrs['Institution'] = 'University of Basel, Department of Environmental Sciences'
    ds.attrs['Contact'] = 'Florian Zabel, florian.zabel@unibas.ch'
    ds.attrs['Creation_Time'] = f'{datetime.now().strftime("%d.%m.%Y - %H:%M")}'
    ds.attrs['Info'] = 'Created by CropSuite v1'
    if isinstance(info_text, str):
        ds.attrs['Info'] = info_text

    merged_data['lat'] = ('lat', np.linspace(y_max, y_min, int(merged_data.dims['lat'])))
    encoding = {var: {'zlib': True, 'complevel': 4} for var in merged_data.data_vars}

    client = Client(n_workers=12)
    
    write_job = merged_data.to_netcdf(output_file, encoding=encoding, compute=False)
    print(f"Writing to {output_file}")
    with ProgressBar():
        write_job.compute() #type:ignore

    return output_file

def read_ind_date_file(fn, extent, var_name):
    try:
        ds = xr.open_dataset(fn)
        ds_data = ds.sel(lat=slice(float(extent.get('top')), float(extent.get('bottom'))), lon=slice(float(extent.get('left')), float(extent.get('right'))))[var_name] #type:ignore
        ds.close()
        return np.asarray(ds_data)
    except:
        return None

def read_area_from_netcdf(filename, extent, variable='data', day_range=[-1, -1]):
    # extent = y_max, x_min, y_min, x_max
    ds = xr.open_dataset(filename, engine='netcdf4')

    if variable == '':
        for var_name in ds.data_vars:
            variable = var_name
            break
        if variable is None:
            raise ValueError("No data variables found in the NetCDF file.")
    try:
        ds[variable]
    except:
        variable = get_variable_name_from_nc(filename)

    dimensions = list(ds.dims.keys())

    if 'lat' in dimensions and 'lon' in dimensions:
        lat_dim, lon_dim = 'lat', 'lon'
    elif 'latitude' in dimensions and 'longitude' in dimensions:
        lat_dim, lon_dim = 'latitude', 'longitude'
    if len(dimensions) > 2:
        time_dim = [dim for dim in dimensions if dim not in ['lat', 'lon', 'latitude', 'longitude']][0]

    try:
        y_max, x_min, y_min, x_max = float(extent.get('top')), float(extent.get('left')), float(extent.get('bottom')), float(extent.get('right'))
    except:
        y_max, x_min, y_min, x_max = extent

    try:
        nodata = ds.attrs['_FillValue']
    except:
        nodata = False
    
    if isinstance(day_range, list):
        lat_slc = slice(y_max, y_min) if ds.lat[0] > ds.lat[-1] else slice(y_min, y_max)
        lon_slc = slice(x_min, x_max) if ds.lon[0] < ds.lon[-1] else slice(x_max, x_min)
        if day_range[1] > -1:
            data = ds.sel(**{lat_dim: lat_slc, lon_dim: lon_slc, time_dim: slice(day_range[0]+1, day_range[1]+1)}) #type:ignore
        else:
            data = ds.sel(**{lat_dim: lat_slc, lon_dim: lon_slc}) #type:ignore

    elif isinstance(day_range, int):
        lat_slc = slice(y_max, y_min) if ds.lat[0] > ds.lat[-1] else slice(y_min, y_max)
        lon_slc = slice(x_min, x_max) if ds.lon[0] < ds.lon[-1] else slice(x_max, x_min)
        if day_range > -1:
            data = ds.sel(**{lat_dim: lat_slc, lon_dim: lon_slc, time_dim: day_range+1}) #type:ignore
        else:
            data = ds.sel(**{lat_dim: lat_slc, lon_dim: lon_slc}) #type:ignore

    else:
        print('error')
    data = np.asarray(data[variable])
    return data, nodata

def sort_coordinatelist(filelist):
    lst = list(zip(filelist, [int(str(os.path.basename(os.path.dirname(f))).split('_')[1].split('N')[0]) for f in filelist]))
    lst.sort(key=lambda x: x[1])    
    return [f[0] for f in lst]


def read_area_from_netcdf_list(downscaled_files, overlap = False, var_name = 'data', extent = [0, 0, 0, 0], timestep=-1, dayslices=False, transp=True, workers = 0):
    """
        downscaled_files: list of netcdf files
        overlap: In Degree
        extent: [North, Left, South, Right]
    """

    if dayslices:
        downscaled_files = [f for f in downscaled_files if os.path.exists(f)]
        ds_list = []
        total = len(downscaled_files)
        try:
            extent = {'top': extent[0], 'left': extent[1], 'bottom': extent[2], 'right': extent[3]}
        except:
            pass
        
        
        if workers !=0:
            data_perday = {}
            with tqdm(total=total) as pbar:
                with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
                    future_to_day = {
                            executor.submit(read_ind_date_file, fn, extent, var_name): idx
                            for idx, fn in enumerate(downscaled_files)
                        }
                    for future in concurrent.futures.as_completed(future_to_day):
                        day = future_to_day[future]
                        npdata = future.result()
                        if npdata is not None: data_perday[str(day)] = npdata 
                        pbar.update(1)
                ## order results
            list_days = [int(i) for i in list(data_perday.keys())]
            list_days.sort()
            ds_list= [data_perday[str(i)]for i in list_days]
        else:
            for idx, fn in tqdm(enumerate(downscaled_files)):
                npdata = read_ind_date_file(fn, extent, var_name)
                if npdata is not None: ds_list.append(npdata)
        
        if transp:
            return np.transpose(np.asarray(ds_list), (1, 2, 0))
        else:
            return np.asarray(ds_list)

    else:
        if overlap:
            extent = {'top': extent[0] + overlap, 'left': extent[1], 'bottom': extent[2] - overlap, 'right': extent[3]}
        else:
            extent = {'top': extent[0], 'left': extent[1], 'bottom': extent[2], 'right': extent[3]}

        ds_list = []
        downscaled_files = sort_coordinatelist(downscaled_files)
        for fn in reversed(downscaled_files):
            current_extent = get_netcdf_extent(fn)
            if not (((current_extent['top'] >= extent['top']) and (current_extent['bottom'] <= extent['top'])) or\
                    ((current_extent['top'] >= extent['bottom']) and current_extent['bottom'] <= extent['bottom'])):
                continue

            current_area = {'top': np.min([current_extent.get('top'), extent.get('top')]), #type:ignore
                'left': np.max([current_extent.get('left'), extent.get('left')]), #type:ignore
                'bottom': np.max([current_extent.get('bottom'), extent.get('bottom')]), #type:ignore
                'right': np.min([current_extent.get('right'), extent.get('right')])} #type:ignore

            ds = xr.open_dataset(fn)
            if timestep == -1:
                ds_data = ds.sel(lat=slice(current_area.get('top'), current_area.get('bottom')), lon=slice(current_area.get('left'), current_area.get('right')))[var_name]
            else:
                ds_data = ds.sel(lat=slice(current_area.get('top'), current_area.get('bottom')), lon=slice(current_area.get('left'), current_area.get('right')), day=timestep)[var_name]
            ds.close() 
            ds_list.append(np.asarray(ds_data))

        if len(ds_list) == 1:
            return ds_list[0]
        else:
            if overlap:
                return []
            else:
                return np.concatenate(ds_list, axis=0)
            
def write_to_netcdf(data, filename, dimensions=['lat', 'lon'], extent=None, compress=False, complevel=4, info_text=False, var_name='data', nodata_value=False, unlimited=None):
    if len(dimensions) != data.ndim:
        raise ValueError(f'Specified dimensions {dimensions} do not match data dimensions {data.shape}')
    
    if extent:
        try:
            latitudes = np.linspace(float(extent.get('top')), float(extent.get('bottom')), data.shape[0])
            longitudes = np.linspace(float(extent.get('left')), float(extent.get('right')), data.shape[1])            
        except:
            try:
                extent = list(extent.values())
            except:
                pass
            latitudes = np.linspace(np.max([extent[0], extent[2]]), np.min([extent[0], extent[2]]), data.shape[0])
            longitudes = np.linspace(np.min([extent[1], extent[3]]), np.max([extent[1], extent[3]]), data.shape[1])
    else:
        latitudes, longitudes = np.arange(data.shape[0]), np.arange(data.shape[1])
    
    # Ensure correct data type
    if data.dtype == np.float16:
        data = data.astype(np.float32)
    elif data.dtype == np.int8:
        data = data.astype(np.int16)
    
    fill_value = 0
    if nodata_value:
        fill_value = nodata_value
    elif np.isnan(data).any():
        fill_value = np.nan
    elif np.min(data) == -32767:
        fill_value = -32767
    
    with nc4.Dataset(filename, 'w', format='NETCDF4') as ds: #type:ignore
        
        # Create dimensions
        ds.createDimension('lat', data.shape[0])
        ds.createDimension('lon', data.shape[1])
        for dim in range(2, data.ndim):
            ds.createDimension(dimensions[dim], data.shape[dim])
        
        # Create coordinate variables
        lat_var = ds.createVariable('lat', 'f4', ('lat',))
        lon_var = ds.createVariable('lon', 'f4', ('lon',))
        lat_var[:] = latitudes
        lon_var[:] = longitudes
        
        # Add attributes to coordinates
        lat_var.units = 'degrees_north'
        lon_var.units = 'degrees_east'
        lat_var.long_name = 'latitude'
        lon_var.long_name = 'longitude'
        lat_var.axis = 'Y'
        lon_var.axis = 'X'
        crs='EPSG:4326'
        crs_obj = CRS.from_user_input(crs)
        crs_var = ds.createVariable('crs', 'i4')
        crs_var.long_name = 'Coordinate Reference System'
        crs_var.grid_mapping_name = crs_obj.to_cf().get('grid_mapping_name', 'latitude_longitude')
        crs_var.spatial_ref = crs_obj.to_wkt()
        crs_var.EPSG_code = f"EPSG:{crs_obj.to_epsg()}" if crs_obj.to_epsg() else 'unknown'
        # link CRS to data variable
        grid_mapping_name = 'crs'
        # Create data variable
        var_dims = tuple(dimensions)
        var = ds.createVariable(var_name, data.dtype, var_dims, zlib=compress, complevel=complevel, fill_value=fill_value) #type:ignore
        var[:] = data
        
        if grid_mapping_name:
            var.grid_mapping = grid_mapping_name
        
        # Add global attributes
        ds.setncattr('Institution', 'University of Basel, Department of Environmental Sciences')
        ds.setncattr('Contact', 'Florian Zabel & Matthias Knuettel, florian.zabel@unibas.ch')
        ds.setncattr('Creation_Time', datetime.now().strftime("%d.%m.%Y - %H:%M"))
        ds.setncattr('Info', 'Created by CropSuite v1.0.1')
        if isinstance(info_text, str):
            ds.setncattr('Info', info_text)
    return filename
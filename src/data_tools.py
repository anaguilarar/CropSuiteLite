import numpy as np
import rasterio
from scipy.interpolate import RegularGridInterpolator
from multiprocessing import cpu_count
from psutil import virtual_memory
import os
import shutil

from rasterio.transform import from_bounds
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles
from scipy.ndimage import zoom

from .nc_tools import (get_variable_name_from_nc, read_area_from_netcdf)



def create_cog_from_geotiff(src_path, dst_path, profile="deflate", profile_options={}, **options):
    """Convert image to COG."""
    output_profile = cog_profiles.get(profile)
    output_profile.update(dict(BIGTIFF="IF_SAFER"))
    output_profile.update(profile_options)
    config = dict(GDAL_NUM_THREADS="ALL_CPUS", GDAL_TIFF_INTERNAL_MASK=True, GDAL_TIFF_OVR_BLOCKSIZE="128")
    cog_translate(src_path, dst_path, output_profile, config=config, in_memory=False, quiet=True, **options)

def geotiff_to_smallest_datatype(geotiff_list):
    min_val, max_val = 0, 0
    for input_file in geotiff_list:
        with rasterio.open(input_file, 'r') as src:
            data = src.read()
            min_val = np.min([min_val, np.min(data)])
            max_val = np.max([max_val, np.max(data)])
            if min_val > src.nodata:
                min_val = src.nodata
            src.close()
    if min_val >= -128 and max_val <= 128:
        dtype = np.int8
    elif min_val >= 0 and max_val <= 255:
        dtype = np.uint8
    elif min_val >= 0 and max_val <= 65535:
        dtype = np.uint16
    elif min_val >= -32768 and max_val <= 32767:
        dtype = np.int16
    elif min_val >= 0 and max_val <= 4294967295:
        dtype = np.uint32
    elif min_val >= -2147483648 and max_val <= 2147483647:
        dtype = np.int32
    else:
        dtype = np.float32
    print(f' -> Using {dtype} for {os.path.basename(geotiff_list[0])}')
    for input_file in geotiff_list:
        if not os.path.exists(input_file):
            continue

        temp_file = 'temp.tif'
        with rasterio.open(input_file) as src:
            data = src.read()
            meta = src.meta.copy()
            data_converted = data.astype(dtype)
            meta.update({'dtype': np.dtype(dtype).name})
        
        with rasterio.open(temp_file, 'w', **meta) as dst:
            dst.write(data_converted)
        shutil.move(temp_file, input_file)

def get_shape_of_raster(raster_file):
    """
    Get the shape of a raster file.

    Parameters:
    - raster_file: Numpy array or raster file object. Input raster data.

    Returns:
    - shape: Tuple. Shape of the raster data in the format (rows, columns).
    """
    return(np.shape(raster_file))


def get_resolution_array(config_dictionary, extent, only_shape=False, climate=False):
    resolution_value = int(config_dictionary['options'].get('resolution', 5))
    if climate:
        resolution_value = np.min([resolution_value, 5])
    resolution_dict = {0: 0.5, 1: 0.25, 2: 0.1, 3: 0.08333333333333, 4: 0.041666666666666, 5: 0.008333333333333, 6: 0.00208333333333333}

    try:
        y_max, y_min = float(extent.get('top')), float(extent.get('bottom'))
        x_max, x_min = float(extent.get('right')), float(extent.get('left'))
    except:
        y_max, y_min = np.max([extent[0], extent[2]]), np.min([extent[0], extent[2]])
        x_max, x_min = np.max([extent[1], extent[3]]), np.min([extent[1], extent[3]])

    resolution = resolution_dict.get(resolution_value, 0.00833333333333)

    px_y = (y_max - y_min) / resolution
    px_x = (x_max - x_min) / resolution

    if (px_y - int(px_y)) >= 0.5:
        px_y = int(px_y) + 1
    else:
        px_y = int(px_y)

    if (px_x - int(px_x)) >= 0.5:
        px_x = int(px_x) + 1
    else:
        px_x = int(px_x)

    if only_shape:
        return (px_y, px_x)
    else:
        return np.empty((px_y, px_x))
    

def extract_domain_from_global_raster(raster_dataset, domain, raster_extent=[-180, 90, 180, -90]):
    """
    Extracts a specific domain from a global raster dataset.

    Parameters:
    - raster_dataset (numpy.ndarray): Global raster dataset.
    - domain (list or numpy.ndarray): Domain to be extracted in the format [Left, Top, Right, Bottom].
    - raster_extent (list, optional): Extent of the entire global raster in the format [Left, Top, Right, Bottom].
      Defaults to [-180, 90, 180, -90].

    Returns:
    - extracted_domain (numpy.ndarray): Extracted domain from the global raster dataset.
    """

    if raster_dataset.ndim == 2:
        big_rows, big_cols = get_shape_of_raster(raster_dataset) # Großes Raster Zeilen und Spalten
    else:
        big_rows, big_cols = get_shape_of_raster(raster_dataset)[1:3]
    try:
        big_uy = raster_extent.top # Großes Raster oben
        big_by = raster_extent.bottom # Großes Raster unten
        big_lx = raster_extent.left # Großes Raster links
        big_rx = raster_extent.right # Großes Raster rechts
    except:
        big_uy = raster_extent[1] # Großes Raster oben
        big_by = raster_extent[3] # Großes Raster unten
        big_lx = raster_extent[0] # Großes Raster links
        big_rx = raster_extent[2] # Großes Raster rechts

    try:
        small_uy = domain.top # Kleines Raster oben
        small_by = domain.bottom # Kleines Raster unten
        small_lx = domain.left # Kleines Raster links
        small_rx = domain.right # Kleines Raster rechts
    except:
        small_uy = domain[1] # Kleines Raster oben
        small_by = domain[3] # Kleines Raster unten
        small_lx = domain[0] # Kleines Raster links
        small_rx = domain[2] # Kleines Raster rechts

    pix_top = int(np.round((big_rows/(big_uy - big_by)) * (small_uy-big_by)))
    pix_bottom = int(np.round((big_rows/(big_uy - big_by)) * (small_by-big_by)))

    pix_left = int(np.round((big_cols/(big_rx - big_lx)) * (small_lx - big_lx)))
    pix_right = int(np.round((big_cols/(big_rx - big_lx)) * (small_rx - big_lx)))
    
    if raster_dataset.ndim == 2:
        return raster_dataset[big_rows-pix_top:big_rows-pix_bottom, pix_left:pix_right]
    else:
        return raster_dataset[:, big_rows-pix_top:big_rows-pix_bottom, pix_left:pix_right]


def load_specified_lines(filepath, extent, all_bands = True):
    try:
        y_max, y_min = float(extent.get('top')), float(extent.get('bottom'))
        x_max, x_min = float(extent.get('right')), float(extent.get('left'))
    except:
        y_max, y_min = np.max([extent[0], extent[2]]), np.min([extent[0], extent[2]])
        x_max, x_min = np.max([extent[1], extent[3]]), np.min([extent[1], extent[3]])

    if filepath.endswith('.nc'):
        var_name = get_variable_name_from_nc(filepath)
        subset_data, nodata = read_area_from_netcdf(filepath, extent, var_name)
    else:
        with rasterio.open(filepath, 'r') as src:
            window = src.window(x_min, y_min, x_max, y_max)
            try:
                window = Window(window.col_off, np.round(window.row_off), window.width, np.round(window.height)) #type:ignore
            except:
                pass
            if isinstance(all_bands, bool):
                if all_bands:
                    if src.count == 1:
                        subset_data = src.read(window=window)
                    else:
                        shp = src.read(1, window=window).shape
                        subset_data = np.empty((src.count, shp[0], shp[1]), dtype=src.dtypes[1])
                        for i in range(src.count):
                            subset_data[i] = src.read(i+1, window=window)
                    #subset_data = src.read(window=window)
                else:
                    subset_data = src.read(1, window=window)
            elif isinstance(all_bands, int):
                subset_data = src.read(all_bands, window=window)
            nodata = src.nodata
    return subset_data, nodata

def interpolate_nanmask(array, new_shape):
    """
    Interpolate a binary mask with NaN values to a new specified shape.

    Parameters:
    - array (numpy.ndarray): Input binary mask with NaN values to be interpolated.
    - new_shape (tuple): A tuple specifying the new shape of the interpolated mask (height, width).

    Returns:
    - numpy.ndarray: Interpolated binary mask with the specified shape.

    Note:
    - The function assumes 'array' is a 2D numpy array representing a binary mask with NaN values.
    - It performs linear interpolation using the 'interp2d' function from scipy.
    - The resulting interpolated values are thresholded to create a binary mask.
    """
    """
    h, w = array.shape
    array = array.astype(float)
    interp_func = interp2d(np.arange(w), np.arange(h), array, kind='linear')
    ret = interp_func(np.linspace(0, w - 1, int(new_shape[1])), np.linspace(0, h - 1, int(new_shape[0])))
    ret = (ret >= 0.5).astype(bool)
    return ret
    """
    
    h, w = array.shape
    array = array.astype(float)
    
    # Define the grid points for the original array
    x = np.arange(w)
    y = np.arange(h)
    
    # Create the interpolation function
    interp_func = RegularGridInterpolator((y, x), array, method='linear')
    
    # Generate the new grid points for interpolation
    new_x = np.linspace(0, w - 1, int(new_shape[1]))
    new_y = np.linspace(0, h - 1, int(new_shape[0]))
    
    # Create a meshgrid for the new points
    new_grid = np.meshgrid(new_y, new_x, indexing='ij')
    new_points = np.array(new_grid).reshape(2, -1).T
    
    # Perform the interpolation
    ret = interp_func(new_points).reshape(new_shape)
    
    # Apply threshold and convert to boolean
    ret = (ret >= 0.5).astype(bool)
    
    return ret

def interpolate_array(data, target_shape, order=1):
    """
    Interpolates a 3D array to the specified target shape using spline interpolation.

    Parameters:
    - data: 3D numpy array of shape (365, 16, 18)
    - target_shape: tuple of the target shape (e.g., (365, 960, 1080))

    Returns:
    - Interpolated array with shape (365, 960, 1080)
    """
    # Calculate the zoom factors for each axis
    zoom_factors = [target_dim / original_dim for target_dim, original_dim in zip(target_shape, data.shape)]

    # Apply zoom interpolation
    interpolated_data = zoom(data, zoom_factors, order=order)  # order=1 for bilinear interpolation

    return interpolated_data


def extract_domain_from_global_3draster(raster_dataset, domain, raster_extent=[-180, 90, 180, -90], axis=0):
    """
    Extracts a specific domain from a global 3D raster dataset.

    Parameters:
    - raster_dataset (numpy.ndarray): Global 3D raster dataset.
    - domain (list or numpy.ndarray): Domain to be extracted in the format [left, top, right, bottom].
    - raster_extent (list, optional): Extent of the entire global raster in the format [left, top, right, bottom].
      Defaults to [-180, 90, 180, -90].
    - axis (int, optional): Axis along which the domain is extracted. Defaults to 0.

    Returns:
    - extracted_domain (numpy.ndarray): Extracted domain from the global 3D raster dataset.
    """
    if raster_dataset.ndim == 2:
        big_rows, big_cols = get_shape_of_raster(raster_dataset) # Großes Raster Zeilen und Spalten
    else:
        big_rows, big_cols = get_shape_of_raster(raster_dataset)[1:3]
    try:
        big_uy = raster_extent.top # Großes Raster oben
        big_by = raster_extent.bottom # Großes Raster unten
        big_lx = raster_extent.left # Großes Raster links
        big_rx = raster_extent.right # Großes Raster rechts
    except:
        big_uy = raster_extent[1] # Großes Raster oben
        big_by = raster_extent[3] # Großes Raster unten
        big_lx = raster_extent[0] # Großes Raster links
        big_rx = raster_extent[2] # Großes Raster rechts

    try:
        small_uy = domain.top # Kleines Raster oben
        small_by = domain.bottom # Kleines Raster unten
        small_lx = domain.left # Kleines Raster links
        small_rx = domain.right # Kleines Raster rechts
    except:
        small_uy = domain[1] # Kleines Raster oben
        small_by = domain[3] # Kleines Raster unten
        small_lx = domain[0] # Kleines Raster links
        small_rx = domain[2] # Kleines Raster rechts
    pix_top = int(np.round((big_rows/(big_uy - big_by)) * (small_uy-big_by)))
    pix_bottom = int(np.round((big_rows/(big_uy - big_by)) * (small_by-big_by)))
    pix_left = int(np.round((big_cols/(big_rx - big_lx)) * (small_lx - big_lx)))
    pix_right = int(np.round((big_cols/(big_rx - big_lx)) * (small_rx - big_lx)))
    if axis == 0:
        return raster_dataset[:, big_rows-pix_top:big_rows-pix_bottom, pix_left:pix_right]
    elif axis == 1:
        return raster_dataset[big_rows-pix_top:big_rows-pix_bottom, :, pix_left:pix_right]
    elif axis == 2:
        return raster_dataset[big_rows-pix_top:big_rows-pix_bottom, pix_left:pix_right, ...]
    else:
        return []

def fill_nan_nearest(array, nodata=np.nan, return_nanmask=False):
    if np.isnan(nodata):
        mask = np.isnan(array)
    else:
        mask = array == nodata
    array[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), array[~mask])
    if return_nanmask:
        return array, mask
    else:
        return array
    
def get_cpu_ram() -> list:
    """
    Get information about the CPU and available RAM.

    Returns:
    list: A list containing the number of CPU cores and the available RAM in gigabytes.

    This function uses the psutil library to retrieve information about the CPU and
    available RAM. The returned list has two elements: the number of CPU cores and
    the available RAM in gigabytes.
    """
    return [cpu_count(), virtual_memory().available/1000000000]

def get_geotiff_extent(file_path) -> tuple:
    """
    Get the spatial extent (bounding box) of a GeoTIFF file.

    Args:
    file_path (str): The path to the GeoTIFF file.

    Returns:
    tuple: A tuple (minx, miny, maxx, maxy) representing the bounding box.

    This function uses rasterio to open the GeoTIFF file and retrieve its spatial
    extent. If an error occurs, it displays an error message and exits the program.
    """
    try:
        with rasterio.open(file_path, 'r') as dataset:
            bounds = dataset.bounds
            return bounds
    except Exception as e:
        throw_exit_error(f'An error occurred while retrieving extent of {file_path}: {e}')
        return 0, 0, 0, 0

def read_tif_file_with_bands(fn) -> np.ndarray:
    """
    Read a GeoTIFF file with multiple bands into a NumPy array.

    Parameters:
    - fn (str): Path to the GeoTIFF file.

    Returns:
    numpy.ndarray: Array containing the bands from the GeoTIFF file.

    This function uses the rasterio library to read a GeoTIFF file with multiple bands
    into a NumPy array. The returned array has shape (num_bands, height, width).
    """
    with rasterio.open(fn, 'r') as src:
        data = src.read()
    return data

def read_raster_to_array(raster_file, nodata=-9999.) -> np.ndarray:
    """
    Read a raster file into a NumPy array.

    Parameters:
    - raster_file (str): Path to the raster file.
    - nodata (float): NoData value in the raster file (default is -9999.).

    Returns:
    numpy.ndarray: Array containing the raster data with NoData values replaced by NaN.

    This function uses rasterio to read a raster file into a NumPy array.
    NoData values are replaced with NaN if the data type is float (float16, float32, or float).
    """
    with rasterio.open(raster_file, 'r') as src:
        data = src.read(1)
    if data.dtype == np.float16 or data.dtype == float or data.dtype == np.float32:
        data[data == nodata] = np.nan
    return data

def resize_array_interp(array, new_shape, nodata=None, limit=(-9999, -9999), method='linear'):
    """
    Resizes a 2D array to the specified new shape using linear interpolation.

    Parameters:
    - array (np.ndarray): 2D array to be resized.
    - new_shape (tuple): Tuple representing the target shape (dimensions) in the format (rows, columns).
    - limit (tuple, optional): Tuple containing the lower and upper limits for the resized array values. Default is (-9999, -9999).

    Returns:
    - resized_array (np.ndarray): Resized array with the specified new shape using linear interpolation.

    Note: The function uses linear interpolation to resize the input array to the new shape.
    Values outside the specified limits are set to 0 if limits are provided.
    """
    if not nodata is None:
        array, nanmask = fill_nan_nearest(array, nodata, return_nanmask=True)

    h, w = array.shape

    # Create an interpolating function using RegularGridInterpolator
    interp_func = RegularGridInterpolator(
        (np.arange(h), np.arange(w)),
        array,
        method=method
    )

    # Define the target points for the new shape
    target_x = np.linspace(0, h - 1, int(new_shape[0]))
    target_y = np.linspace(0, w - 1, int(new_shape[1]))
    target_points = np.array(np.meshgrid(target_x, target_y, indexing='ij')).reshape(2, -1).T

    # Interpolate to get the resized array
    ret = interp_func(target_points).reshape(int(new_shape[0]), int(new_shape[1]))
    if limit[0] != -9999: ret[ret<=-limit[0]] = 0
    if limit[1] != -9999: ret[ret>=limit[1]] = 0
    if not nodata is None:
        nanmask = interpolate_nanmask(nanmask, new_shape) # type: ignore
        ret += np.where(nanmask, np.nan, 0.0)
    return ret, None if not nodata else np.nan

def throw_exit_error(text):
    """
    Display an error message, wait for user input, and exit the program.

    Parameters:
    - text (str): The error message to be displayed.

    Note:
    - The function displays the specified error message, waits for user input, and then exits the program.
    """
    input(text+'\nExit')
    exit()
    

def write_geotiff(filepath, filename, array, extent, crs='+proj=longlat +datum=WGS84 +no_defs +type=crs', nodata_value=-1., dtype='float', cog=False, inhibit_message=False):
    """
    Write a NumPy array to a GeoTIFF file.

    Args:
    filepath (str): Path to the directory where the GeoTIFF file will be saved.
    filename (str): Name of the GeoTIFF file.
    array (numpy.ndarray): NumPy array to be written to the GeoTIFF.
    extent (tuple or rasterio.bounds.BoundingBox): Tuple or BoundingBox containing the extent (xmin, ymin, xmax, ymax) of the data.
    crs (str): Coordinate reference system string.
    nodata_value (float): NoData value for the GeoTIFF.
    dtype (str): Data type of the array ('float', 'int', 'bool').

    Writes the array to a GeoTIFF file using rasterio.

    """
    if not inhibit_message:
        print(f' -> Writing {filename}')
    output_path = os.path.join(filepath, filename)
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    if array.ndim == 2:
        height, width = array.shape
        try:
            transform = from_bounds(extent.left, extent.bottom, extent.right, extent.top, width, height)
        except:
            try:
                transform = from_bounds(extent[1], extent[2], extent[3], extent[0], width, height)
            except:
                transform = from_bounds(float(extent['left']), float(extent['bottom']), float(extent['right']), float(extent['top']), width, height)
        if dtype == 'float':
            array = array.astype(float)
            array[np.isnan(array)] = nodata_value
            with rasterio.open(output_path, 'w', driver='GTiff', height=height, width=width, count=1, dtype=str(array.dtype), crs=crs, transform=transform, nodata=nodata_value, compress='lzw') as dst:
                dst.write(array, 1)
        if dtype == 'int':
            nodata_value = int(nodata_value)
            with rasterio.open(output_path, 'w', driver='GTiff', height=height, width=width, count=1, dtype=array.dtype, crs=crs, transform=transform, compress='lzw', nodata=nodata_value) as dst:
                dst.write(array, 1)
        if dtype == 'bool':
            with rasterio.open(output_path, 'w', driver='GTiff', nbits=1, height=height, width=width, crs=crs, count=1, dtype=np.uint8, transform=transform, compress='lzw') as dst:
                dst.write(array.astype(np.uint8), 1)
    elif array.ndim == 3:
        height, width, num_bands = array.shape
        try:
            transform = from_bounds(extent.left, extent.bottom, extent.right, extent.top, width, height)
        except:
            transform = from_bounds(extent[1], extent[2], extent[3], extent[0], width, height)
        if dtype == 'float':
            array = array.astype(float)
            array[np.isnan(array)] = nodata_value
            with rasterio.open(output_path, 'w', driver='GTiff', height=height, width=width, count=num_bands, dtype=str(array.dtype), crs=crs, transform=transform, nodata=nodata_value, compress='lzw') as dst:
                for band in range(num_bands):
                    dst.write(array[..., band], band+1)
        if dtype == 'int':
            nodata_value = int(nodata_value)
            with rasterio.open(output_path, 'w', driver='GTiff', height=height, width=width, count=num_bands, dtype=array.dtype, crs=crs, transform=transform, compress='lzw', nodata=nodata_value) as dst:
                for band in range(num_bands):
                    dst.write_band(band+1, array[..., band])
        if dtype == 'bool':
            with rasterio.open(output_path, 'w', driver='GTiff', nbits=1, height=height, width=width, count=num_bands, crs=crs, dtype=np.uint8, transform=transform, compress='lzw') as dst:
                for band in range(num_bands):
                    dst.write_band(band+1, array[..., band].astype(np.uint8))
        
    else:
        raise ValueError('Array dimensions should be 2 or 3.')
    
    if cog:
        create_cog_from_geotiff(output_path, output_path.replace('.tif', '_cog.tif'))


import os

import numpy as np
import rasterio
from matplotlib import path
from rasterio.transform import from_bounds

from .check_files import get_id_list_start
from .data_tools import (interpolate_nanmask, load_specified_lines, resize_array_interp, read_tif_file_with_bands, 
                         write_geotiff, throw_exit_error)

from .nc_tools import write_to_netcdf

def aggregate_soil_raster_lst(file_list, domain, final_shape, weighting_method = 0, conversion_factor = 1., weighting = [2., 1.5, 1., 0.75, 0.5, 0.25]):
    """
    Aggregate soil raster data from a list of files based on specified parameters.

    Parameters:
    - file_list (list): A list of file paths to the soil raster data.
    - domain (list): A list containing the geographic domain coordinates [min_longitude, min_latitude, max_longitude, max_latitude].
    - final_shape (tuple): A tuple specifying the final shape of the aggregated array (rows, columns).
    - weighting_method (int): The method used for weighting during aggregation (0, 1, 2).
    - conversion_factor (float): A conversion factor applied to the aggregated result.
    - weighting (list): A list of weighting factors used in the aggregation process.

    Returns:
    - tuple: A tuple containing the aggregated soil data array and the nodata value.

    Note:
    - The function aggregates soil raster data from a list of files based on the specified weighting method.
    - 'weighting_method' determines the aggregation strategy: 0 for single layer, 1 for mean of multiple layers, 2 for custom weighted sum.
    - The 'conversion_factor' scales the aggregated result.
    - The function returns a tuple containing the aggregated soil data array and the nodata value.
    """
    nodata = None
    
    if weighting_method == 0:
        layer, nodata = load_specified_lines(file_list[0], [domain[1], domain[0], domain[3], domain[2]], all_bands=False) #type:ignore
        if layer.shape != final_shape:
            layer, nodata = resize_array_interp(layer, final_shape, nodata=nodata, method='nearest')
        nan_mask = layer == nodata
        layer = layer / conversion_factor
        layer[nan_mask] = nodata #type:ignore
        return layer, nodata

    elif weighting_method == 1:
        file_list = file_list[:3]
        layers = []
        for layer_file in file_list:
            layer, nodata = load_specified_lines(layer_file, [domain[1], domain[0], domain[3], domain[2]], all_bands=False)
            if layer.shape != final_shape:
                nan_mask = layer == nodata
                nan_mask = interpolate_nanmask(nan_mask, final_shape)
                layer, nodata = resize_array_interp(layer, final_shape, nodata=nodata, method='nearest')
            layers.append(layer)

        nan_mask = layers[0] == nodata
        mean = np.nanmean(layers, axis=0)
        # mean[nan_mask] = nodata
        # if mean.shape != final_shape:
        #     nan_mask = dt.interpolate_nanmask(nan_mask, final_shape)
        #     mean, nodata = resize_array_interp(mean, final_shape, nodata=nodata, method='nearest')

        new_file = mean / conversion_factor
        new_file[nan_mask] = nodata  #type:ignore
        return new_file, nodata

    elif weighting_method == 2:
        layers = []
        for layer_file in file_list:
            layer, nodata = load_specified_lines(layer_file, [domain[1], domain[0], domain[3], domain[2]], all_bands=False)
            layer, nodata = resize_array_interp(layer, final_shape, nodata=nodata, method='nearest')
            layers.append(layer)

        #for idx, layer in enumerate(layers):
        #    if layer.shape != final_shape:
        #        layers[idx], nodata = resize_array_interp(layer, final_shape, nodata=nodata, method='nearest')

        nan_mask = layers[0] == nodata

        new_file = np.zeros((layers[0].shape))
        new_file = new_file.astype(float)
        new_file = layers[0] * 5 * weighting[0] # 0-5
        new_file += layers[1] * 10 * weighting[0] # 5-15
        new_file += layers[2] * 10 * weighting[0] # 15-25
        new_file += layers[2] * 5 * weighting[1] # 25-30
        new_file += layers[3] * 20 * weighting[1] # 30-50
        new_file += layers[3] * 10 * weighting[2] # 50-60
        new_file += layers[4] * 15 * weighting[2] # 60-75
        new_file += layers[4] * 25 * weighting[3] # 75-100
        new_file += layers[5] * 25 * weighting[4] # 100-125
        new_file += layers[5] * 25 * weighting[5] # 125-150
        new_file = new_file / 150 / conversion_factor  # type: ignore
        new_file[nan_mask] = nodata
        return new_file, nodata
        
    else:
        throw_exit_error('Unkown Weighting Method')
        return np.zeros(final_shape), 0


def calculate_slope(dem_path, output_shape, extent):
    if output_shape == (0, 0) or extent == [0, 0, 0, 0]:
        with rasterio.open(dem_path, 'r') as dem:
            transform = dem.transform
            cell_size = transform.a
            dem_array = dem.read(1)
    else:
        cell_size = ((extent[2] - extent[0]) / output_shape[1])
        domain = [extent[1], extent[0], extent[3], extent[2]]
        dem_array = load_specified_lines(dem_path, domain)[0][0]
        dem_array, _ = resize_array_interp(dem_array, output_shape)
    R = 6371000
    resolution_rad = np.deg2rad(cell_size)
    resolution_m = R * resolution_rad
    dz_dy, dz_dx = np.gradient(dem_array, resolution_m, edge_order=2)
    slope = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    slope_degrees = np.rad2deg(slope)
    return slope_degrees


def get_soil_data(climate_config, current_parameter, domain, shape, itype):
    """
    Get soil data based on specified climate configuration parameters.

    Parameters:
    - climate_config (dict): The climate configuration dictionary containing parameter settings.
    - current_parameter (str): The current parameter for which soil data is retrieved.
    - domain (str): The geographic domain or area of interest.
    - shape (tuple): A tuple specifying the shape of the output data (rows, columns).
    - itype: The numpy data type for the output array.

    Returns:
    - numpy.ndarray: The soil data array.

    Note:
    - The function retrieves soil data based on specified climate configuration parameters.
    - It handles various configurations, including data directories, weighting methods, conversion factors, and nodata values.
    - The retrieved soil data is aggregated using the 'aggregate_soil_raster_lst' function from your module.
    """
    try:
        current_dict = climate_config[f'parameters.{current_parameter}']
    except:
        throw_exit_error(f'{current_parameter} is not defined')
        current_dict = {}
    
    data_dir = current_dict['data_directory']
    weighting_method = int(current_dict['weighting_method'])
    weighting_factors = np.asarray(str(current_dict['weighting_factors']).split(',')).astype(float)
    conversion_factor = float(current_dict['conversion_factor'])

    param_datasets = []

    if os.path.isdir(data_dir):
        for fn in os.listdir(data_dir):
            if str(fn).lower().endswith('.tif') or str(fn).lower().endswith('.tiff'):
                param_datasets.append(fn)
    elif os.path.isfile(data_dir):
        if str(data_dir).lower().endswith('.tif') or str(data_dir).lower().endswith('.tiff'):
            param_datasets.append(os.path.basename(data_dir))
            data_dir = os.path.dirname(data_dir)

    if len(param_datasets) > 1:
        param_datasets = sorted(param_datasets, key=lambda x: int(str(x).split('_')[1].split('-')[0]))

    param_datasets = [os.path.join(data_dir, param_datasets[i]) for i in range(len(param_datasets))]

    dataset, no_data = aggregate_soil_raster_lst(param_datasets, domain, shape, weighting_method, conversion_factor, weighting_factors)
    if no_data:
        if not np.issubdtype(dataset.dtype, np.floating):
            dataset = dataset.astype(np.float32)
        dataset[dataset == no_data] = np.nan 
    return dataset.astype(itype)

def get_suitability_val_dict(forms, plant, form_type, value):
    """
    Calculate the suitability value for a given form type and value based on predefined formulas.

    Parameters:
    - forms (dict): A dictionary containing predefined formulas for different forms, plants, and form types.
    - plant (str): The plant for which the suitability value is calculated.
    - form_type (str): The type of form for which the suitability value is calculated.
    - value (float or numpy.ndarray): The input value or array of values for the specified form type.

    Returns:
    - numpy.ndarray: The suitability values calculated based on the predefined formula.

    Note:
    - The function uses predefined formulas stored in the 'forms' dictionary to calculate suitability values.
    - It handles both scalar and array input for the 'value' parameter.
    - The output is clipped between 0 and 1 to ensure it falls within the suitability range.
    - If 'value' is a numpy array, NaN values are preserved using a nanmask during the calculations.
    """
    replacements = {'base_saturation': 'base_sat', 'coarse_fragments': 'coarsefragments', 'salinity': 'elco',
                    'sodicity': 'esp', 'pH': 'ph', 'soil_organic_carbon': 'organic_carbon'}
    if form_type in replacements:
        form_type = replacements[form_type]
    func, min_val, max_val = forms[plant][form_type]['formula'], forms[plant][form_type]['min_val'], forms[plant][form_type]['max_val']
    
    if isinstance(value, np.ndarray):
        nanmask = np.isnan(value)
        value = np.clip(value, min_val, max_val)
        value[value >= max_val] = max_val
        value[value <= min_val] = min_val

        try:
            value = np.clip(func(value), 0, 1)
        except Exception as e:
            print('Error in suitability function application:', str(e))
            value = np.full_like(value, 1)
        value[nanmask] = -0.01
        return value
    else:
        value = min(max_val, max(min_val, value))
        return np.clip(func(value), 0, 1)

def getTable(fn):
    """
    Parse a file to create a table of class names and associated sand-clay boundaries.

    Parameters:
    - fn (str): File path to the input file.

    Returns:
    - dict: A dictionary where keys are class names and values are matplotlib.path.Path objects representing sand-clay boundaries.

    Note:
    - The function reads the content of the file, extracts class names, sand, and clay limits, and creates a dictionary with matplotlib.path.Path objects.
    """
    with open(fn, 'r') as f:
        x = f.readlines()
    nbClasses = int(x[0])
    classNamesRaw = x[1 + nbClasses*4 + 1: 1 + nbClasses*4 + 1 + nbClasses + 1]
    classNames = dict(item.strip().split('=') for item in classNamesRaw)
    table = {}
    for i in range(nbClasses):
        className = x[1 + i*4].strip()
        sandLimits = list(map(float, x[3 + i*4].split()))
        clayLimits = list(map(float, x[4 + i*4].split()))
        verts = np.column_stack((sandLimits, clayLimits)) # type: ignore
        table[classNames[className]] = path.Path(verts)
    return table

def get_texture_class(sand, clay, config):
    """
    Get the texture class based on sand and clay percentages using USDA soil texture boundaries.

    Parameters:
    - sand (numpy.ndarray): Array containing sand percentages.
    - clay (numpy.ndarray): Array containing clay percentages.
    - config (dict): Configuration dictionary containing file paths.

    Returns:
    - numpy.ndarray: Array containing texture class indices.

    Note:
    - The function uses USDA soil texture boundaries from a specified file to classify sand and clay percentages into texture classes.
    - The result is an array of indices corresponding to different texture classes.
    ```
    {1: 'heavy clay', 2: 'silty clay', 3: 'clay', 4: 'silty clay loam', 5: 'clay loam',
     6: 'silt', 7: 'silt loam', 8: 'sandy clay', 9: 'loam', 10: 'sandy clay loam',
     11: 'sandy loam', 12: 'loamy sand', 13: 'sand'}
    ```
    """
    usda_table = getTable(os.path.join(config['files']['texture_classes']))
    texture_dict = {'heavy clay': 1,
                    'silty clay': 2,
                    'clay': 3,
                    'silty clay loam': 4,
                    'clay loam': 5,
                    'silt': 6,
                    'silt loam': 7,
                    'sandy clay': 8,
                    'loam': 9,
                    'sandy clay loam': 10,
                    'sandy loam': 11,
                    'loamy sand': 12,
                    'sand': 13}
    sand = np.array(sand)
    clay = np.array(clay)
    texture = np.zeros_like(sand, dtype=np.int8)
    for key, p in usda_table.items():
        mask = p.contains_points(np.column_stack((sand.flatten(), clay.flatten())))
        texture[mask.reshape(sand.shape)] = int(texture_dict[str(key)])
    return texture

def get_valid_dtype(itype):
    dtype_map = {
        'uint8': 'uint8',
        'int8': 'int8',
        'uint16': 'uint16',
        'int16': 'int16',
        'uint32': 'uint32',
        'int32': 'int32',
        'float32': 'float32',
        'float64': 'float64'
    }
    try:
        return dtype_map.get(itype.type, 'float32')
    except:
        try:
            return dtype_map.get(itype.name, 'float32')
        except:
            return dtype_map.get(itype, 'float32')

def output_param_data(param_arr, param_list, output_dir, domain):
    for idx, param in enumerate(param_list):
        out_f = f'{param}_combined.tif'
        dataset = param_arr[..., idx]
        valid_dtype = get_valid_dtype(param_arr[..., idx].dtype)
        try:
            os.remove(os.path.join(output_dir, out_f))
        except:
            pass
    
        crs = 'EPSG:4326'
        transform = from_bounds(domain[0], domain[3], domain[2], domain[1], dataset.shape[1], dataset.shape[0])
        metadata = {'driver': 'GTiff', 'height': dataset.shape[0], 'width': dataset.shape[1], 'count': 1, 'dtype': valid_dtype, 'crs': crs, 'transform': transform, 'nodata': np.nan}
        with rasterio.open(os.path.join(output_dir, out_f), 'w', **metadata) as dst:
            dst.write(dataset.astype(valid_dtype), 1)

def stack_parameters_array(config, raster_ref_shape, parameter_list, domain) -> np.ndarray:
    """
    read and interpolate parameter data
    
    ------
    raster_ref_shape List: Height width
        
    :param parameter_list: dictionary with the paremeters
    :param domain: Description extent
    """
    parameter_array = np.empty((*raster_ref_shape, len(parameter_list)), dtype=np.float16)
    for counter, parameter in enumerate(parameter_list):
        print(f' -> Loading {parameter} data')
        if parameter == 'wealth':
            
            print(config['files'].get('wealth_dir'))
            wealth_array = load_specified_lines(config['files'].get('wealth_dir'), 
                                                [domain[1], domain[0], domain[3], domain[2]] # -> xyxy -> yxyx
                                                )
            
            print(f'------ domain: {domain}')
            print(f'-> Shape {wealth_array[0].shape} data')
            wealth_array, _ = resize_array_interp(wealth_array[0][0], raster_ref_shape)
            print(f'********* Wealth {wealth_array.shape}')
            
            parameter_array[..., counter] = wealth_array
            
        elif parameter == 'slope':
            m = calculate_slope(config['files'].get('fine_dem'), raster_ref_shape, domain)
            m_shape = m.shape
            print(f'********* slope {m_shape}')
            parameter_array[..., counter] = calculate_slope(config['files'].get('fine_dem'), raster_ref_shape, domain)
        else:
            msoil = get_soil_data(config, parameter, domain, raster_ref_shape, np.float16)
            
            m_shape = msoil.shape
            print(f'********* {parameter}:  {m_shape}')
            parameter_array[..., counter] = msoil
    
    return parameter_array

def calcification_map(config, parameter_list, parameter_array, extent, results_path):

    calcification_val = int(config['options'].get('simulate_calcification', '0'))
    ph_index = parameter_list.index('pH')
    if calcification_val == 1:
        calcification = 0.5
    elif calcification_val == 2:
        calcification = 1.0
    else:
        calcification = 1.5
    
    ph_data = parameter_array[..., ph_index]
    ph_gap = 6.5 - ph_data
    ph_add = np.clip(ph_gap, 0, calcification)
    parameter_array[..., ph_index] += ph_add

    top, left, bottom, right = extent
    height, width = ph_add.shape
    transform = from_bounds(left, bottom, right, top, width, height)
    with rasterio.open(os.path.join(results_path, 'ph_increase.tif'), 'w', driver='GTiff', height=height, width=width, count=1, dtype=rasterio.int8,
                        crs="EPSG:4326", transform=transform, nodata=-1,compress='LZW') as dst:
        dst.write((ph_add * 10.).astype(np.int8), 1)

    ph_write = parameter_array[..., ph_index].copy()
    ph_write[ph_write <= 0] = -.1
    ph_write = (ph_write * 10).astype(np.int8)
    with rasterio.open(os.path.join(results_path, 'ph_after_liming.tif'), 'w', driver='GTiff', height=height, width=width, count=1, dtype=rasterio.int8,
                        crs="EPSG:4326", transform=transform, nodata=-1,compress='LZW') as dst:
        dst.write(ph_write, 1)
    del ph_write

    texture_index = parameter_list.index('texture')
    texture_class = parameter_array[..., texture_index].astype(np.int8)

    # t CaCo3 / ha / 0.1 pH
    texture_caco3_dict = {1: 2.0, 2: 1.75, 3: 1.85, 4: 1.5, 5: 1.5, 6: 1.2,
                            7: 1.05, 8: 1.2, 9: 0.9, 10: 0.75, 11: 0.6, 12: 0.45, 13: 0.375}
    keys = np.array(list(texture_caco3_dict.keys()))
    lut = np.zeros(keys.max() + 1)
    lut[keys] = np.array(list(texture_caco3_dict.values()))
    caco_array = lut[texture_class] * ph_add * 10
    with rasterio.open(os.path.join(results_path, 'lime_application.tif'), 'w', driver='GTiff', height=height, width=width, count=1, dtype=rasterio.float32,
                        crs="EPSG:4326", transform=transform, nodata=-1,compress='LZW') as dst:
        dst.write(caco_array, 1)



def cropsuitability(config, clim_suit_shape, plant_formulas, plant_params, extent, land_sea_mask, results_path, multiple_cropping_sum=None):
    """
    Calculate soil suitability for crop plants based on climate suitability and soil parameters.

    Parameters:
    - config (dict): Configuration dictionary containing file paths and options.
    - clim_suit_shape (list): Array containing climate suitability shape.
    - plant_formulas (dict): Dictionary containing formulas for calculating plant suitability.
    - plant_params (dict): Dictionary containing plant parameters.
    - extent (list): List containing the geographical extent for analysis.
    - land_sea_mask (numpy.ndarray): Array representing land-sea mask.
    - results_path (str): Path to the directory where results will be stored.

    Returns:
    - None

    Note:
    - The function calculates soil suitability for each plant based on climate suitability, soil parameters, and limiting factors.
    - It processes data according to the specified configuration and stores results in the specified directory.
    """
    print('Calculate the Soil Suitability')
    domain = [-180, 90, 180, -90]
    if extent:
        domain = [extent[1], extent[0], extent[3], extent[2], 0, 0]

    plant_list = [plant for plant in plant_params]    
    if os.path.exists(os.path.join(results_path, plant_list[-1], 'crop_suitability.tif')):
        return

    parameter_list = [entry.replace('parameters.', '', 1) if entry.startswith('parameters.') else entry for entry in get_id_list_start(config, 'parameters.')] + ['slope']
    parameter_array = stack_parameters_array(config, clim_suit_shape, parameter_list, domain)
    parameter_dictionary = {parameter_list[parameter_id]: config[f'parameters.{parameter_list[parameter_id]}']['rel_member_func'] for parameter_id in range(len(parameter_list)) if f'parameters.{parameter_list[parameter_id]}' in config}
    parameter_dictionary['slope'] = 'slope'
    print(' -> Converting sand and clay content to texture class')
    parameter_array[..., parameter_list.index('sand_content')] = get_texture_class(parameter_array[..., parameter_list.index('sand_content')],\
                                                                                parameter_array[..., parameter_list.index('clay_content')],\
                                                                                config)

    parameter_array = np.delete(parameter_array, parameter_list.index('clay_content'), axis=2)
    parameter_list[parameter_list.index('sand_content')] = 'texture'
    del parameter_list[parameter_list.index('clay_content')]

    if config['options'].get('output_soil_data', 0) == 1:
        output_param_data(parameter_array, parameter_list, results_path, domain[:4])

    ##### CALCIFICATION #####
    if int(config['options'].get('simulate_calcification', '0')) > 0:
        calcification_map(config, parameter_list, parameter_array, extent, results_path)
    
    plant_list = [plant for plant in plant_params]
    formulas = [plant for plant in plant_formulas[plant_list[0]]]
    formulas = dict(zip(formulas, np.arange(0, len(formulas))))

    for plant_idx, plant in enumerate([p for p in os.listdir(results_path) if os.path.isdir(os.path.join(results_path, p))]):
        res_path = os.path.join(results_path, plant)
        if not os.path.exists(os.path.join(res_path, 'climate_suitability.tif')) and not os.path.exists(os.path.join(res_path, 'climate_suitability.nc')):
            continue
        print('\n'+f'Processing suitability for {plant}')
        
        os.makedirs(res_path, exist_ok=True)
        if os.path.exists(os.path.join(res_path, 'crop_suitability.tif')):
            continue
        
        suitability_array = np.empty((*clim_suit_shape, len(parameter_list)), dtype=np.float16)        

        max_ind_val_climate = 3

        order_file = os.path.join(res_path, 'limiting_factor.inf')
        with open(order_file, 'w') as write_file:
            write_file.write('value - limiting factor\n')
            write_file.write('0 - temperature\n')
            write_file.write('1 - precipitation\n')
            write_file.write('2 - climate variability\n')
            write_file.write('3 - photoperiod\n')
            for counter, parameter in enumerate(parameter_list):
                print(f'******** Parameter: {parameter} ***********')
                if parameter == 'texture' and 'texture' in plant_formulas[plant]:
                    suitability_array[..., counter] = (get_suitability_val_dict(plant_formulas, plant, 'texture', parameter_array[..., counter])*100).astype(np.float16)
                elif parameter not in parameter_dictionary or parameter_dictionary[parameter] not in plant_formulas[plant]:
                    print(f' -> {plant} has no parameter {parameter}. Skipping {parameter}.')
                    suitability_array[..., counter] = (np.ones_like(suitability_array[..., counter])*100).astype(np.float16)
                else:
                    suitability_array[..., counter] = (get_suitability_val_dict(plant_formulas, plant, parameter_dictionary[parameter], parameter_array[..., counter])*100).astype(np.float16)
                write_file.write(f'{counter+max_ind_val_climate+1} - {parameter}'+'\n')

        curr_climsuit = np.squeeze(load_specified_lines(
                        next(f for f in [os.path.join(results_path, plant, f'climate_suitability{ext}') for ext in ['.tif', '.nc', '.nc4']] if os.path.exists(f)),
                        extent, False
                    )[0].astype(np.int8))
        
        ## soil
        soil_suitablility = np.min(suitability_array, axis=2).astype(np.int8)
        suitability_array = np.concatenate((curr_climsuit[..., np.newaxis], suitability_array), axis=2)
        suitability = np.clip(np.min(suitability_array, axis=2), -1, 100) 
        suitability[np.isnan(land_sea_mask)] = -1
        suitability = suitability.astype(np.int8)
        ## suitability multiplication
        suitability_multi = (curr_climsuit.astype(float) * soil_suitablility.astype(float))
        suitability_multi = ((suitability_multi)/(100*100)*100).astype(np.int8)
        
        min_indices = (np.argmin(suitability_array, axis=2) + max_ind_val_climate).astype(np.int8)

        lim_factor = np.squeeze(load_specified_lines(
                        next(f for f in [os.path.join(results_path, plant, f'limiting_factor{ext}') for ext in ['.tif', '.nc', '.nc4']] if os.path.exists(f)),
                        extent, False
                    )[0].astype(np.int8))
      
        min_indices[min_indices == max_ind_val_climate] = lim_factor[min_indices == max_ind_val_climate]
        min_indices[np.isnan(land_sea_mask)] = -1
        nan_mask = suitability == -1
        min_indices[nan_mask] = -1
        
        suitability_array[np.isnan(suitability_array)] = -1
        suitability_array = suitability_array.astype(np.int8)

        if config['options']['output_all_limiting_factors'] and os.path.exists(os.path.join(res_path, 'all_climlim_factors.tif')):
            clim_lims = read_tif_file_with_bands(os.path.join(res_path, 'all_climlim_factors.tif'))
            suitability_array = np.concatenate([np.transpose(clim_lims, (1, 2, 0)), suitability_array[..., 1:]], axis=2)
            suitability_array[nan_mask, :] = -1
            write_geotiff(res_path, 'all_suitability_vals.tif', suitability_array, extent, dtype='int', nodata_value=-1, cog=config['options']['output_format'].lower() == 'cog')
            
        if config['options']['output_format'] == 'geotiff' or config['options']['output_format'] == 'cgo':
            write_geotiff(res_path, 'crop_suitability.tif', suitability, extent, nodata_value=-1, cog=config['options']['output_format'].lower() == 'cog')
            write_geotiff(res_path, 'crop_suitability_multi.tif', suitability_multi, extent, nodata_value=-1, cog=config['options']['output_format'].lower() == 'cog')
            write_geotiff(res_path, 'crop_limiting_factor.tif', min_indices, extent, nodata_value=-1, cog=config['options']['output_format'].lower() == 'cog')
            write_geotiff(res_path, 'soil_suitability.tif', soil_suitablility, extent, dtype='int', nodata_value=-1, cog=config['options']['output_format'].lower() == 'cog')
        elif config['options']['output_format'] == 'netcdf4':
            write_to_netcdf(suitability, os.path.join(res_path, 'crop_suitability.nc'), extent=extent, compress=True, var_name='crop_suitability', nodata_value=-1) #type:ignore
            write_to_netcdf(min_indices, os.path.join(res_path, 'crop_limiting_factor.nc'), extent=extent, compress=True, var_name='crop_limiting_factor', nodata_value=-1) #type:ignore
            write_to_netcdf(soil_suitablility, os.path.join(res_path, 'soil_suitability.nc'), extent=extent, compress=True, var_name='soil_suitability', nodata_value=-1) #type:ignore
        else:
            print('No output format specified.')

        try:
            del suitability_array, suitability, soil_suitablility, clim_lims, nan_mask, min_indices, curr_climsuit
            
        except:
            pass
    print('\nSuitability data created')

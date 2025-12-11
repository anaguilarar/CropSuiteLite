import sys
import os
import time
import math
import psutil
import gc
import re
import shutil
import rasterio
from typing import List, Tuple, Union

import numpy as np

from src.read_climate_ini import read_ini_file
from src.downscaling import interpolate_precipitation, interpolate_temperature
from src.data_tools import get_resolution_array, load_specified_lines, interpolate_nanmask
from src.read_plant_params import read_crop_parameterizations_files, get_plant_param_interp_forms_dict
from src.nc_tools import read_area_from_netcdf_list
from src.climate_suitability_main import climate_suitability
from src.check_files import check_all_inputs
from src.crop_suitability_main import cropsuitability
from src.crop_rotation import crop_rotation
from src.merge_geotiff import merge_outputs_no_overlap
from src.climate_suitability_main_xarray import climate_suitability_xarray


class CropSuiteLite():
    """
    Main controller for the CropSuiteLite crop suitability modeling framework.

    This class handles configuration loading, data downscaling, tiling strategy,
    and the execution of climate and crop suitability models.

    Parameters
    ----------
    config_file : str
        Path to the configuration file (.ini or .yaml).

    Attributes
    ----------
    config_file : Path
        Path object to the configuration file.
    climate_config : dict
        Dictionary containing parsed configuration parameters.
    extent : list
        The spatial extent [min_y, min_x, max_y, max_x].
    area_name : str
        Formatted string identifier for the geographic area.
    output_path : Path
        Base path for output files.
    day_interval : int
        Time step interval for processing.
    """
    def __init__(self, config_file: str) -> None:
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"The configuration file does not exist: {config_file}")
        
        self.config_file = config_file
        self.climate_config = read_ini_file(self.config_file)
        
        ## get extent
        self.extent = check_all_inputs(self.climate_config)
        self.area_name = f'Area_{int(self.extent[0])}N{int(self.extent[1])}E-{int(self.extent[2])}N{int(self.extent[3])}E'
        self.output_path = self.climate_config['files']['output_dir']
        if os.path.basename(self.output_path) == '': self.output_path = self.output_path[:-1]

        self.day_interval = self.climate_config['options'].get('day_interval', 1)
        self.plant_data()
        
    def resampling_env_data(self) -> Tuple[List[str], List[str], Union[bool, List[str]], Union[bool, List[str]]]:
        """
        Interpolates or retrieves downscaled climate data (precipitation and temperature).

        Returns
        -------
        tuple
            A tuple containing:
            - List of precipitation file paths.
            - List of temperature file paths.
            - List of daily precipitation files (or boolean).
            - List of daily temperature files (or boolean).
        """
        
        if not os.path.exists(self._output_dir):
            prec_files, prec_dailyfiles = interpolate_precipitation(self.climate_config, self.extent, self.area_name)
            temp_files, temp_dailyfiles = interpolate_temperature(self.climate_config, self.extent, self.area_name)
            return prec_files, temp_files, prec_dailyfiles, temp_dailyfiles
        
        else:
            prec_files = [os.path.join(self._output_dir, f'ds_prec_{day}.nc') for day in range(0, 365)]
            temp_files = [os.path.join(self._output_dir, f'ds_temp_{day}.nc') for day in range(0, 365)]
            prec_dailyfiles, temp_dailyfiles = True, True
            return prec_files, temp_files, True, True
        
    def split_into_tiles(self) -> List[List[float]]:
        """
        Calculates grid tiling based on available RAM to prevent memory overflow.

        Returns
        -------
        list of list of float
            A list of extents, where each extent is [min_y, min_x, max_y, max_x].
        """
        final_shape = get_resolution_array(self.climate_config, self.extent, True)
        ram = int((psutil.virtual_memory().total / (1024 ** 3)) * 0.85)
        
        no_tiles_flag = self.climate_config['options'].get('no_tiles', None)
        if no_tiles_flag is None or no_tiles_flag == 'all':
            print(f'Available ram: {ram}')
            no_tiles = int(np.clip((final_shape[0] * final_shape[1]) * 10e-6 / ram, 1, 100000)) if self.climate_config['options'].get('use_scheduler', 1) else 1
        else:
            no_tiles = int(no_tiles_flag)
        
        # Helper to align extent to resolution
        def _adjust_lower_bound(ext, res):
            return ext[2] + ((ext[0] - ext[2]) // res) * res

        if final_shape[0] % no_tiles != 0:
            no_tiles = math.ceil(final_shape[0] / (final_shape[0] // no_tiles))

        resolution = (self.extent[3] - self.extent[1]) / final_shape[1]
        # NOTE: self.extent is modified here to align with grid
        self.extent[0] = _adjust_lower_bound(self.extent, resolution)

        lst = [i * int(final_shape[0] / no_tiles) for i in range(no_tiles)] + [final_shape[0]]
        extents = [[self.extent[2] + lst[i+1] * resolution, self.extent[1], self.extent[2] + lst[i] * resolution, self.extent[3]]for i in range(no_tiles)]
        
        return extents

    def plant_data(self) -> None:
        """
        Loads crop parameterization files and interpolation formulas.
        """
        self.plant_params = read_crop_parameterizations_files(self.climate_config['files']['plant_param_dir'])
        self.plant_params_formulas = get_plant_param_interp_forms_dict(self.plant_params, self.climate_config)

    def compute_climate_suitability(self, 
                                  extent: List[float], 
                                  prec_files: List[str], 
                                  temp_files: List[str], 
                                  prec_dailyfiles: Union[bool, List[str]], 
                                  temp_dailyfiles: Union[bool, List[str]]) -> None:
        """
        Calculates climate suitability based on temperature and precipitation.

        Parameters
        ----------
        extent : list of float
            The specific tile extent [y_max, x_min, y_min, x_max].
        prec_files : list of str
            Paths to precipitation files.
        temp_files : list of str
            Paths to temperature files.
        prec_dailyfiles : bool or list
            Daily file indicators or paths.
        temp_dailyfiles : bool or list
            Daily file indicators or paths.
        """
        area_name = f'Area_{int(extent[0])}N{int(extent[1])}E-{int(extent[2])}N{int(extent[3])}E'
        
        if self.climate_config['climatevariability'].get('consider_variability', True):
            climsuits = [os.path.join(pt, area_name, crop, 'climate_suitability.tif') for pt in [os.path.split(self._temp_path)[0]+'_var', os.path.split(self._temp_path)[0]+'_novar'] for crop in list(self.plant_params.keys())]
        else:
            climsuits = [os.path.join(os.path.split(self._temp_path)[0]+'_novar', area_name, crop, 'climate_suitability.tif') for crop in list(self.plant_params.keys())]
            
        if all(os.path.exists(clims) for clims in climsuits):
            print('\nClimate Suitability Data is already existing.\n -> Using existing data.\n')
        else:
            
            # xarray test
            doit_withxarray = True
            if doit_withxarray:
                climate_suitability_xarray(self.climate_config, extent, temp_files, prec_files, self.plant_params, self._temp_path, self.area_name, day_interval = self.day_interval)
                
            else:
                print(' -> Loading required climate data to memory...')
                temperature = read_area_from_netcdf_list(temp_files, overlap=False, extent=extent, dayslices=temp_dailyfiles, workers=10)
                precipitation = read_area_from_netcdf_list(prec_files, overlap=False, extent=extent, dayslices=prec_dailyfiles, workers=10)
                fine_resolution = (temperature.shape[0], temperature.shape[1]) #type:ignore
                land_sea_mask, _ = load_specified_lines(self.climate_config['files']['land_sea_mask'], extent, False)
                if land_sea_mask.shape != fine_resolution:
                    land_sea_mask = interpolate_nanmask(land_sea_mask, fine_resolution)
                gc.collect()
                print(' -> Climate Data successfully loaded into memory')
                
                climate_suitability(self.climate_config, extent, temperature, precipitation, land_sea_mask, self.plant_params,  self._temp_path, self.area_name, day_interval = self.day_interval)
                del temperature, precipitation
            gc.collect()
                
    
    def compute_crop_suitability(self, extent: List[float]) -> None:
        """
        Combines climate suitability with soil/terrain data to calculate final crop suitability.

        Parameters
        ----------
        extent : list of float
             The specific tile extent.
        """
        
        ret_paths = [os.path.join(os.path.split(self._temp_path)[0]+'_var', os.path.split(self._temp_path)[1]), os.path.join(os.path.split(self._temp_path)[0]+'_novar', os.path.split(self._temp_path)[1])]
        for temp in ret_paths:
            if os.path.exists(temp):
                crops_dir = [croppath for croppath in os.listdir(temp) if (os.path.exists(os.path.join(temp, croppath, 'climate_suitability.tif'))
                                                or os.path.exists(os.path.join(temp, croppath, 'climate_suitability.nc')))]
                
                crop_clipath = os.path.join(temp, crops_dir[0], 'climate_suitability')

                raster_ext = '.tif' if os.path.exists(crop_clipath + '.tif') else '.nc'
                filepath =  os.path.join(crop_clipath + raster_ext)
                if raster_ext == '.tif':
                    with rasterio.open(filepath, 'r') as src:
                        raster_shape = [src.height, src.width]
                else:
                    import xarray as xr
                    with xr.open_dataset(filepath) as ds:
                        raster_shape = [ds.dims['y'], ds.dims['x']]
                        
                land_sea_mask, _ = load_specified_lines(self.climate_config['files']['land_sea_mask'], extent, False)

                if land_sea_mask.shape != raster_shape:
                    land_sea_mask = interpolate_nanmask(land_sea_mask, raster_shape)
                
                cropsuitability(self.climate_config, raster_shape, self.plant_params_formulas, self.plant_params, extent, land_sea_mask, temp)

    def merge_geodata_outputs(self, extents: List[List[float]]) -> None:
        """
        Merges tiled outputs into a single raster for the entire region.

        Parameters
        ----------
        extents : list of list of float
            List of tile extents used during processing.
        """
        try:
            out_path = self.climate_config['files']['output_dir']
            if os.path.basename(out_path) == '': out_path = out_path[:-1]
            for output_dir in [out_path+'_var', out_path+'_novar']:
                if not os.path.exists(output_dir):
                    continue
                if len(extents) > 1:
                    areas = [d for d in next(os.walk(output_dir))[1] if d.startswith('Area_')]
                    north_values = [int(value[:-1]) for item in areas for value in re.findall(r'(-?\d+N)', item)]
                    east_values = [int(value[:-1]) for item in areas for value in re.findall(r'(-?\d+E)', item)]
                    merged_result = os.path.join(output_dir, f'Area_{max(north_values)}N{min(east_values)}E-{min(north_values)}N{max(east_values)}E')
                    if not os.path.exists(merged_result):
                        merge_outputs_no_overlap(output_dir, self.climate_config)
        except:
            return None
            
    def run(self) -> None:
        """
        Executes the full CropSuiteLite pipeline.
        
        Steps:
        1. Downscale climate data.
        2. Split area into tiles based on RAM.
        3. Iterate through tiles to compute Climate and Crop suitability.
        4. Merge tiles.
        5. Clean up temporary files.
        """
        
        ## interpolate products
        print('\nDownscaling the climate data\n')
        
        self._output_dir = os.path.join(self.output_path+'_downscaled', self.area_name)

        prec_files, temp_files, prec_dailyfiles, temp_dailyfiles = self.resampling_env_data()
        
        extents = self.split_into_tiles()
        
        plants = [plant for plant in self.plant_params]
        
        for idx, extent in enumerate(extents):
            area_name = f'Area_{int(extent[0])}N{int(extent[1])}E-{int(extent[2])}N{int(extent[3])}E'
            self._temp_path = os.path.join(self.output_path, area_name)
            formatted_extent = [f"{float(e):.4f}" for e in extent]
            print(f"\nProcessing extent {formatted_extent} - {idx + 1} out of {len(extents)}\n")
            
            if os.path.exists(os.path.join(self.output_path+'_novar', area_name, plants[-1], 'crop_suitability.tif')):
                print(f'Data already existing. Skipping')
                continue
            
            ##### CLIMATE SUITABILITY #####
            self.compute_climate_suitability(extent, prec_files, temp_files, prec_dailyfiles, temp_dailyfiles)
            
            print(os.path.split(self._temp_path))
            
            ##### CROP SUITABILITY #####
            
            self.compute_crop_suitability(extent)
            ##### CROP ROTATION #####
            if self.climate_config['options'].get('consider_crop_rotation', False):
                crop_rotation(self.config_file)
        
            print('Complete Current Extent')
            gc.collect()
        
        ##### MERGING OUTPUTS #####
        self.merge_geodata_outputs(extents)
        
        ##### CLEAN UP #####
        
        if self.climate_config['options']['remove_interim_results']:
            if len(extents) > 1:
                for extent in extents:
                    for output_dir in [self.output_path+'_var', self.output_path+'_novar']:
                        if os.path.exists(output_dir):
                            shutil.rmtree(os.path.join(output_dir, f'Area_{int(extent[0])}N{int(extent[1])}E-{int(extent[2])}N{int(extent[3])}E'))

        if self.climate_config['options'].get('remove_downscaled_climate', False):
            [os.remove(os.path.join(os.path.dirname(prec_files[0]), f)) for f in os.listdir(os.path.dirname(prec_files[0])) if os.path.isfile(os.path.join(os.path.dirname(prec_files[0]), f))]

            try:
                os.removedirs(os.path.dirname(temp_files[0]))
                os.removedirs(os.path.dirname(prec_files[0]))
            except:
                pass

        gc.collect()


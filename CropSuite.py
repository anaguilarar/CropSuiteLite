import sys
import os
import time
import math
import psutil
import gc
import re
import shutil

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


class CropSuiteLite():
    
    def __init__(self, config_file):
        assert os.path.exists(config_file), 'The file does not exist'
        self.config_file = config_file
        self.climate_config = read_ini_file(self.config_file)
        
        ## get extent
        self.extent = check_all_inputs(self.climate_config)
        self.area_name = f'Area_{int(self.extent[0])}N{int(self.extent[1])}E-{int(self.extent[2])}N{int(self.extent[3])}E'
        self.output_path = self.climate_config['files']['output_dir']
        self.day_interval = self.climate_config['options'].get('day_interval', 1)
        self.plant_data()
        
    def resampling_env_data(self):
        
        if not os.path.exists(self._output_dir):
            prec_files, prec_dailyfiles = interpolate_precipitation(self.climate_config, self.extent, self.area_name)
            temp_files, temp_dailyfiles = interpolate_temperature(self.climate_config, self.extent, self.area_name)
            return prec_files, temp_files, prec_dailyfiles, temp_dailyfiles
        
        else:
            prec_files = [os.path.join(self._output_dir, f'ds_prec_{day}.nc') for day in range(0, 365)]
            temp_files = [os.path.join(self._output_dir, f'ds_temp_{day}.nc') for day in range(0, 365)]
            prec_dailyfiles, temp_dailyfiles = True, True
            return prec_files, temp_files, True, True
        
    def split_into_tiles(self):
        final_shape = get_resolution_array(self.climate_config, self.extent, True)
        ram = int((psutil.virtual_memory().total / (1024 ** 3)) * 0.85)
        
        no_tiles_flag = self.climate_config['options'].get('no_tiles', None)
        if no_tiles_flag is None or no_tiles_flag == 'all':
            print(f'Available ram: {ram}')
            no_tiles = int(np.clip((final_shape[0] * final_shape[1]) * 10e-6 / ram, 1, 100000)) if self.climate_config['options'].get('use_scheduler', 1) else 1
        else:
            no_tiles = int(no_tiles_flag)
        
        def adjust_extent_0(extent, resolution):
            return extent[2] + ((extent[0] - extent[2]) // resolution ) * resolution

        if final_shape[0] % no_tiles != 0:
            no_tiles = math.ceil(final_shape[0] / (final_shape[0] // no_tiles))

        resolution = (self.extent[3] - self.extent[1]) / final_shape[1]
        self.extent[0] = adjust_extent_0(self.extent, resolution)

        lst = [i * int(final_shape[0] / no_tiles) for i in range(no_tiles)] + [final_shape[0]]
        extents = [[self.extent[2] + lst[i+1] * resolution, self.extent[1], self.extent[2] + lst[i] * resolution, self.extent[3]]for i in range(no_tiles)]
        
        return extents

    def plant_data(self):
        self.plant_params = read_crop_parameterizations_files(self.climate_config['files']['plant_param_dir'])
        self.plant_params_formulas = get_plant_param_interp_forms_dict(self.plant_params, self.climate_config)

    def compute_climate_suitability(self, extent, prec_files, temp_files, prec_dailyfiles, temp_dailyfiles):
        area_name = f'Area_{int(extent[0])}N{int(extent[1])}E-{int(extent[2])}N{int(extent[3])}E'
        
        if self.climate_config['climatevariability'].get('consider_variability', True):
            climsuits = [os.path.join(pt, area_name, crop, 'climate_suitability.tif') for pt in [os.path.split(self._temp_path)[0]+'_var', os.path.split(self._temp_path)[0]+'_novar'] for crop in list(self.plant_params.keys())]
        else:
            climsuits = [os.path.join(os.path.split(self._temp_path)[0]+'_novar', area_name, crop, 'climate_suitability.tif') for crop in list(self.plant_params.keys())]
            
        if all(os.path.exists(clims) for clims in climsuits):
            print('\nClimate Suitability Data is already existing.\n -> Using existing data.\n')
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
                
    
    def compute_crop_suitability(self, extent):
        
        ret_paths = [os.path.join(os.path.split(self._temp_path)[0]+'_var', os.path.split(self._temp_path)[1]), os.path.join(os.path.split(self._temp_path)[0]+'_novar', os.path.split(self._temp_path)[1])]
        for _, temp in enumerate(ret_paths):
            if os.path.exists(temp):
                climsuit = np.dstack([
                    load_specified_lines(
                        next(f for f in [os.path.join(temp, c, f'climate_suitability{ext}') for ext in ['.tif', '.nc', '.nc4']] if os.path.exists(f)),
                        extent, False
                    )[0].astype(np.int8)
                    for c in os.listdir(temp)
                    if c != 'crop_rotation' and os.path.isdir(os.path.join(temp, c))
                ])

                limiting = np.dstack([
                    load_specified_lines(
                        next(f for f in [os.path.join(temp, c, f'limiting_factor{ext}') for ext in ['.tif', '.nc', '.nc4']] if os.path.exists(f)),
                        extent, False
                    )[0].astype(np.int8)
                    for c in os.listdir(temp)
                    if c != 'crop_rotation' and os.path.isdir(os.path.join(temp, c))
                ])

                land_sea_mask, _ = load_specified_lines(self.climate_config['files']['land_sea_mask'], extent, False)
                fine_resolution = (climsuit.shape[0], climsuit.shape[1])
                if land_sea_mask.shape != fine_resolution:
                    land_sea_mask = interpolate_nanmask(land_sea_mask, fine_resolution)
                
                cropsuitability(self.climate_config, climsuit, limiting, self.plant_params_formulas, self.plant_params, extent, land_sea_mask, temp)

    def merge_geodata_outputs(self, extents):
        try:
            for output_dir in [self.climate_config['files']['output_dir']+'_var', self.climate_config['files']['output_dir']+'_novar']:
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
            
    def run(self):
        
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
                    for output_dir in [self.climate_config['files']['output_dir']+'_var', self.climate_config['files']['output_dir']+'_novar']:
                        if os.path.exists(output_dir):
                            shutil.rmtree(os.path.join(output_dir, f'Area_{int(extent[0])}N{int(extent[1])}E-{int(extent[2])}N{int(extent[3])}E'))

        if self.climate_config['options'].get('remove_downscaled_climate', False):
            [os.remove(os.path.join(os.path.dirname(prec_files[0]), f)) for f in os.listdir(os.path.dirname(prec_files[0])) if os.path.isfile(os.path.join(os.path.dirname(prec_files[0]), f))]

            try:
                os.removedirs(os.path.dirname(temp_files[0]))
                os.removedirs(os.path.dirname(prec_files[0]))
            except:
                pass


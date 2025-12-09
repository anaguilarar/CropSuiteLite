import copy
import gc
import logging
import sys
import warnings

import numpy as np
import yaml
import os
from typing import List, Dict

from CropSuite import CropSuiteLite
from solutions.membership_functions import create_crop_parameters
from solutions.utils import modify_general_files, modify_soil_parameters, modify_extent
from src.read_climate_ini import read_ini_file, write_config

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s', filename='run.log')



def create_crop_suite_configuration_file(config:Dict, soil_parameters,  other_parameters_to_modify:Dict) -> List:
    output_path = other_parameters_to_modify.get('output_dir', 'results') 
    extent = other_parameters_to_modify.get('extent', None) 
    folder_name = os.path.basename(output_path).replace('-','')

    orig_cropsuite_config = read_ini_file(config['GENERAL']['config_path'])
    
    orig_cropsuite_config = modify_soil_parameters(orig_cropsuite_config, soil_parameters)
    orig_cropsuite_config = modify_general_files(orig_cropsuite_config, other_parameters_to_modify)
    if extent is not None: orig_cropsuite_config = modify_extent(orig_cropsuite_config, extent)
    config_fn = f'config_{folder_name}.ini'
    
    write_config(orig_cropsuite_config, config_fn)
    if not os.path.exists(output_path): os.makedirs(output_path)
    write_config(orig_cropsuite_config, os.path.join(output_path, config_fn))

    return config_fn

def modify_initial_cropsuite_config(config_dict):
    output_dir = config_dict['GENERAL'].get('output_path',None)
    soil_path = config_dict['GENERAL'].get('soil_grids_data', None) 
    srtm_path = config_dict['GENERAL'].get('srtm_path', None) 
    extent = config_dict['GENERAL'].get('extent', None) 
    landsea_path = config_dict['GENERAL'].get('landsea_path', None) 
    soil_dict_to_modify = {'data_directory': soil_path}
    climate_scenario = config_dict['GENERAL'].get('climate_scenarios', None)
    plant_param_dir = config_dict['GENERAL'].get('plant_params_output', None)
    max_workers = config_dict['GENERAL'].get('max_workers', 25)
    no_tiles = config_dict['GENERAL'].get('no_tiles', 15)
    crop_rotation = config_dict['GENERAL'].get('consider_crop_rotation', 'n')
    spatial_resolution = config_dict['GENERAL'].get('final_resolution', 4)
    day_interval = config_dict['GENERAL'].get('day_interval', 1)

    cs_name = os.path.basename(climate_scenario).replace('-','_')
    
    files_dict_to_modify = {'output_dir': f'{output_dir}/{cs_name}',
                            'climate_data_dir': climate_scenario,
                            'fine_dem': srtm_path,
                            'land_sea_mask': landsea_path,
                            'plant_param_dir': plant_param_dir,
                            'max_workers': max_workers,
                            'no_tiles': no_tiles,
                            'consider_crop_rotation': crop_rotation,
                            'day_interval': day_interval,
                            'resolution': spatial_resolution}
    if extent is not None:
        files_dict_to_modify.update({'extent':{'upper_left_x': extent[0],
                  'upper_left_y': extent[1],
                  'lower_right_x': extent[2],
                  'lower_right_y': extent[3]}})
        
        print(files_dict_to_modify)
        
    config_file = create_crop_suite_configuration_file(config = config_dict, soil_parameters=soil_dict_to_modify, other_parameters_to_modify = files_dict_to_modify)
    return config_file

def run_crop_suite(config_dict):
    
    logging.info(f'-------> Creating crop parameters')
    print(config_dict)
    create_crop_parameters(config_dict)

    ## modify other parameters
    logging.info(f'-------> changing config_files')
    config_file = modify_initial_cropsuite_config(config_dict)
    
    ## run model
    
    cs_lite = CropSuiteLite(config_file=config_file)
    print(f'-------> Running {config_file}')
    logging.info(f'-------> Running {config_file}')
    cs_lite.run()
    logging.info(f'-------> Finished {config_file}')
    del cs_lite
    
def main(config_path:str):
    logging.info(f'-------> Starting')
    
    with open(config_path, 'r') as file:
        config_dict = yaml.safe_load(file)

    ## if there are multiples 
    sp_resolution = config_dict['GENERAL'].get('final_resolution')
    climate_scenarios = config_dict['GENERAL'].get('climate_scenarios')
    
    if isinstance(sp_resolution, List):
        for res in sp_resolution:
            logging.info(f'-------> Resolution {res}')
            config_dict['GENERAL']['final_resolution'] = res
            config_dict['GENERAL']['output_path'] = config_dict['GENERAL']['output_path'] + '_' + str(res)
            run_crop_suite(config_dict)
            gc.collect()
    
    elif isinstance(climate_scenarios, List):
        orig_path = copy.deepcopy(config_dict['GENERAL']['output_path'])
        for cs in climate_scenarios:
            logging.info(f'-------> Climate Scenario {cs}')
            config_dict['GENERAL']['climate_scenarios'] = cs
            os.path.basename(cs)
            config_dict['GENERAL']['output_path'] = os.path.join(config_dict['GENERAL']['output_path'] , 
                                                                cs.split('/')[-1] if cs.split('/')[-1] == '/' else cs.split('/')[-2])
            run_crop_suite(config_dict)
            config_dict['GENERAL']['output_path'] = orig_path
            gc.collect()

if __name__ == '__main__':
    args = sys.argv[1:]
    config = args[args.index("-config") + 1] if "-config" in args and len(args) > args.index("-config") + 1 else None
    print(config)    
    main(config)



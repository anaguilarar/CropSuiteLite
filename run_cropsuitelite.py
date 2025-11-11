import copy

import numpy as np
import yaml
import os
from typing import List, Dict

from CropSuite import CropSuiteLite
from src.membership_functions import CropSensitivity

from src.read_climate_ini import read_ini_file, write_config
import logging
import sys
import warnings

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s', filename='run.log')



def solution_dict_query(response_function_file: str, solution_code: str):
    assert os.path.exists(response_function_file), 'the file path does not exist'
    
    dict_query = {'crop':str  ,
                'solution_type':str,
                'solution': str,
                'thresholds': List,
                }
    
    with open(response_function_file, 'r') as file:
        sol_config_dict = yaml.safe_load(file)
    
    solution_type_code, solution_code, crop_code = solution_code.split('_')
    
    print(solution_type_code, solution_code, crop_code)
    
    solution_type = sol_config_dict['SOLUTIONS_TYPE'].get(solution_type_code, None)
    solution = sol_config_dict['SOLUTIONS_CODE'].get(solution_code, None)
    crop_str = sol_config_dict['CROPS'].get(crop_code, None)
    threshold = None if solution is None else sol_config_dict[solution_type_code][solution_code][crop_code]
    
    dict_query.update({'crop': crop_str,
                    'solution_type': solution_type,
                    'solution':  solution,
                    'crop': crop_str,
                    'thresholds':threshold})
    return  dict_query

    
def change_st1_parameter(solution: str, thresholds: List, crops_params: CropSensitivity, v2 = True):
    
    new_params = None
    solution_parameter = {}
    
    for i in range(len(thresholds)):
        if 'heat' in solution.lower():
            upperlimit_temp = thresholds[i]
            
            vals_orig = crops_params.params['temp_vals']
            maxtmp = vals_orig[::-1][0]
            new_maxtemp = maxtmp * (1+(upperlimit_temp/100))

            #new_maxtemp = maxtmp + upperlimit_temp
            suit_vals, new_params = crops_params.create_new_max_parameter_values(parameter='temp',new_max=new_maxtemp, percentage = False)
            
            solution_parameter[f'temp_{i+1}'] = new_params
            
        elif 'drought' in solution.lower():
            lowerlimit_prec = thresholds[i]
            vals_orig = crops_params.params['prec_vals']
            minprec = vals_orig[0]
            new_minprec = max(minprec - (minprec * lowerlimit_prec/100), 0)
            if v2: 
                print('--> lowerlimit_prec: ', lowerlimit_prec)
                suit_vals, new_params = crops_params.multiply_suit_vals('prec', lowerlimit_prec)
                #suit_vals, new_params = crops_params.create_new_min_parameter_valuesv2(parameter='prec',new_min=lowerlimit_prec/100)
            else:
                suit_vals, new_params = crops_params.create_new_min_parameter_values(parameter='prec',new_min=new_minprec)
            
            solution_parameter[f'prec_{i+1}'] = new_params
            
    return solution_parameter, suit_vals

def modify_soil_parameters(cropsuite_configuration, parameters_to_modify:Dict):
    
    soil_parameters = {'base_saturation':'bsat', 'coarse_fragments':'cfvo', 'clay_content':'clay', 
                       'gypsum':'gyps', 'pH':'ph', 'salinity':'sal', 
                       'sand_content':'sand', 'soil_organic_carbon':'soc', 
                       'sodicity':'sod', 'soildepth':'soildepth'}
    
    for soil_param, val in soil_parameters.items():
        for k, v in parameters_to_modify.items():
            if k == 'data_directory':
                #if os.listdir(v):
                cropsuite_configuration[f'parameters.{soil_param}'][k] = os.path.join(v, val)
            else:
                cropsuite_configuration[f'parameters.{soil_param}'][k] = v

    return cropsuite_configuration

def modify_general_files(cropsuite_configuration, parameters_to_modify:Dict):
    general_config_list = list(cropsuite_configuration['files'].keys())

    for k, v in parameters_to_modify.items():
        #if not (k in general_config_list): continue
        if k in list(cropsuite_configuration['options'].keys()):
            cropsuite_configuration['options'][k] = v
        elif k in general_config_list:
            cropsuite_configuration['files'][k] = v
        else:
            warnings.warn(f"{k} is not in options either files")

    return cropsuite_configuration

def modify_extent(cropsuite_configuration, parameters_to_modify:Dict):
    extent_list = list(cropsuite_configuration['extent'].keys())

    for k, v in parameters_to_modify.items():
        #if not (k in general_config_list): continue
        if k in extent_list:
            cropsuite_configuration['extent'][k] = v
        

    return cropsuite_configuration

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

def create_crop_parameters(config_dict):
    
    solution_codes = config_dict['SOLUTIONS'].get('solution_codes')
    crop_codes = config_dict['SOLUTIONS'].get('crop_codes')
    solutions_path = config_dict['SOLUTIONS'].get('solutions_path')
    plant_params_input = config_dict['GENERAL'].get('plant_param_dir', 'plant_params/available')
    plant_params_output = config_dict['GENERAL'].get('plant_params_output', 'plant_params')
    if not os.path.exists(plant_params_output): os.mkdir(plant_params_output)
    

    if len(solution_codes):
        for crop_code in crop_codes:
         
            for i, sol_code in enumerate(solution_codes):
                
                solution_to_implement = f'ST1_{sol_code}_{crop_code}'

                solution_implementation = solution_dict_query(solutions_path, solution_to_implement)
                print(solution_implementation)
                crops_params = CropSensitivity(crop = solution_implementation['crop'], 
                                            parameters_path = plant_params_input)
                crops_params.read_crop_configuration()
                crops_params.remove_crop_lethal_conditions()
                
                sv, suit_vals = change_st1_parameter(solution_implementation['solution'],solution_implementation['thresholds'],  crops_params, v2 = True)
                crops_params.plot_solutions_profiles(sv, suit_values = suit_vals, output_fig_path=os.path.join(plant_params_output,f'{solution_to_implement}.png'))
                crops_params.export_crop_params(sv, plant_params_output, suit_vals = suit_vals, export_original = i == 0, code = solution_to_implement)
                with open(os.path.join(plant_params_output, f'{solution_to_implement}.yaml'), 'w') as file:
                    yaml.dump(solution_implementation, file, default_flow_style=False)
    else:
        for crop_code in crop_codes:
            solution_to_implement = f'ST1_null_{crop_code}'
            solution_implementation = solution_dict_query(solutions_path, solution_to_implement)
            cname = solution_implementation['crop']
            crops_params = CropSensitivity(crop = cname, 
                                            parameters_path = plant_params_input)
            crops_params.read_crop_configuration()
            crops_params.remove_crop_lethal_conditions()
            #crops_params.export_crop_params(sv, plant_params_output, suit_vals = suit_vals, export_original = i == 0, code = solution_to_implement)
            crops_params.write_configuration(os.path.join(plant_params_output, f'{cname}.inf'))

        print(solution_implementation)

def modify_initial_cropsuite_config(config_dict):
    output_dir = config_dict['GENERAL'].get('output_path',None)
    soil_path = config_dict['GENERAL'].get('soil_grids_data', None) 
    srtm_path = config_dict['GENERAL'].get('srtm_path', None) 
    extent = config_dict['GENERAL'].get('extent', None) 
    print('extent', extent)
    landsea_path = config_dict['GENERAL'].get('landsea_path', None) 
    soil_dict_to_modify = {'data_directory': soil_path}
    climate_scenario = config_dict['GENERAL'].get('climate_scenarios', None)
    plant_param_dir = config_dict['GENERAL'].get('plant_params_output', None)
    max_workers = config_dict['GENERAL'].get('max_workers', 25)
    no_tiles = config_dict['GENERAL'].get('no_tiles', 15)
    crop_rotation = config_dict['GENERAL'].get('consider_crop_rotation', 'n')
    spatial_resolution = config_dict['GENERAL'].get('final_resolution', 4)

    cs_name = os.path.basename(climate_scenario).replace('-','_')
    
    files_dict_to_modify = {'output_dir': f'{output_dir}/{cs_name}',
                            'climate_data_dir': climate_scenario,
                            'fine_dem': srtm_path,
                            'land_sea_mask': landsea_path,
                            'plant_param_dir': plant_param_dir,
                            'max_workers': max_workers,
                            'no_tiles': no_tiles,
                            'consider_crop_rotation': crop_rotation,
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


if __name__ == '__main__':
    args = sys.argv[1:]
    config = args[args.index("-config") + 1] if "-config" in args and len(args) > args.index("-config") + 1 else None
    print(config)    
    main(config)


